from typing import Dict, Optional

import torch
from torch import nn
from transformers.models.bert.modeling_bert import BertPreTrainingHeads
from transformers.models.luke.modeling_luke import EntityPredictionHead
from transformers.models.roberta.modeling_roberta import RobertaLMHead

from luke.model import LukeConfig, LukeModel
from luke.pretraining.metrics import Accuracy, Average


class CLSEntityPredictionHead(nn.Module):
    def __init__(self, config: LukeConfig):
        super().__init__()
        self.config = config
        self.decoder = nn.Linear(config.hidden_size, config.entity_emb_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.entity_emb_size))

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class LukePretrainingModel(LukeModel):
    def __init__(self, config: LukeConfig):
        super().__init__(config)

        if self.config.bert_model_name and "roberta" in self.config.bert_model_name:
            self.lm_head = RobertaLMHead(config)
            self.lm_head.decoder.weight = self.embeddings.word_embeddings.weight
        else:
            self.cls = BertPreTrainingHeads(config)
            self.cls.predictions.decoder.weight = self.embeddings.word_embeddings.weight

        self.entity_predictions = EntityPredictionHead(config)
        self.entity_predictions.decoder.weight = self.entity_embeddings.entity_embeddings.weight
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

        if self.config.cls_entity_prediction:
            self.cls_entity_predictions = CLSEntityPredictionHead(config)
            self.loss_fn_cls = nn.MSELoss()

        self.apply(self.init_weights)

        self.metrics = {
            "masked_lm_loss": Average(),
            "masked_lm_accuracy": Accuracy(),
            "masked_entity_loss": Average(),
            "masked_entity_accuracy": Accuracy(),
            "entity_prediction_loss": Average(),
        }

    def forward(
        self,
        word_ids: torch.LongTensor,
        word_segment_ids: torch.LongTensor,
        word_attention_mask: torch.LongTensor,
        entity_ids: torch.LongTensor = None,
        entity_position_ids: torch.LongTensor = None,
        entity_segment_ids: torch.LongTensor = None,
        entity_attention_mask: torch.LongTensor = None,
        masked_entity_labels: Optional[torch.LongTensor] = None,
        masked_lm_labels: Optional[torch.LongTensor] = None,
        page_id: torch.LongTensor = None,
        **kwargs
    ):
        model_dtype = next(self.parameters()).dtype  # for fp16 compatibility

        output = super().forward(
            word_ids,
            word_segment_ids,
            word_attention_mask,
            entity_ids,
            entity_position_ids,
            entity_segment_ids,
            entity_attention_mask,
        )

        if entity_ids is not None:
            word_sequence_output, entity_sequence_output = output[:2]
        else:
            word_sequence_output = output[0]

        ret = dict(loss=word_ids.new_tensor(0.0, dtype=model_dtype))

        if masked_entity_labels is not None:
            entity_mask = masked_entity_labels != -1
            if entity_mask.sum() > 0:
                target_entity_sequence_output = torch.masked_select(entity_sequence_output, entity_mask.unsqueeze(-1))
                target_entity_sequence_output = target_entity_sequence_output.view(-1, self.config.hidden_size)
                target_entity_labels = torch.masked_select(masked_entity_labels, entity_mask)

                entity_scores = self.entity_predictions(target_entity_sequence_output)
                entity_scores = entity_scores.view(-1, self.config.entity_vocab_size)

                masked_entity_loss = self.loss_fn(entity_scores, target_entity_labels)
                self.metrics["masked_entity_loss"](masked_entity_loss)
                self.metrics["masked_entity_accuracy"](
                    prediction=torch.argmax(entity_scores, dim=1), target=target_entity_labels
                )
                ret["loss"] += masked_entity_loss

        if masked_lm_labels is not None:
            masked_lm_mask = masked_lm_labels != -1
            if masked_lm_mask.sum() > 0:
                masked_word_sequence_output = torch.masked_select(word_sequence_output, masked_lm_mask.unsqueeze(-1))
                masked_word_sequence_output = masked_word_sequence_output.view(-1, self.config.hidden_size)

                if self.config.bert_model_name and "roberta" in self.config.bert_model_name:
                    masked_lm_scores = self.lm_head(masked_word_sequence_output)
                else:
                    masked_lm_scores = self.cls.predictions(masked_word_sequence_output)
                masked_lm_scores = masked_lm_scores.view(-1, self.config.vocab_size)
                masked_lm_labels = torch.masked_select(masked_lm_labels, masked_lm_mask)

                masked_lm_loss = self.loss_fn(masked_lm_scores, masked_lm_labels)

                self.metrics["masked_lm_loss"](masked_lm_loss)
                self.metrics["masked_lm_accuracy"](
                    prediction=torch.argmax(masked_lm_scores, dim=1), target=masked_lm_labels
                )
                ret["loss"] += masked_lm_loss

        if page_id is not None:
            true_entity_embeddings = torch.gather(
                self.entity_embeddings.entity_embeddings.weight,
                0,
                page_id.unsqueeze(1).repeat(1, self.config.entity_emb_size),
            )
            cls_token_embeddings = word_sequence_output[:, 0]
            pred_entity_embeddings = self.cls_entity_predictions(cls_token_embeddings)
            entity_prediction_loss = self.loss_fn_cls(pred_entity_embeddings, true_entity_embeddings)
            ret["loss"] += entity_prediction_loss
            self.metrics["entity_prediction_loss"](entity_prediction_loss)

        return ret

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {k: m.get_metric(reset=reset) for k, m in self.metrics.items()}
