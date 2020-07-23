from typing import Optional
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.modeling_bert import ACT2FN, BertLayerNorm, BertPreTrainingHeads
from transformers.modeling_roberta import RobertaLMHead

from luke.model import LukeModel, LukeConfig


class EntityPredictionHeadTransform(nn.Module):
    def __init__(self, config: LukeConfig):
        super(EntityPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.entity_emb_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = BertLayerNorm(config.entity_emb_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class EntityPredictionHead(nn.Module):
    def __init__(self, config: LukeConfig):
        super(EntityPredictionHead, self).__init__()
        self.config = config
        self.transform = EntityPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.entity_emb_size, config.entity_vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.entity_vocab_size))

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias

        return hidden_states


class LukePretrainingModel(LukeModel):
    def __init__(self, config: LukeConfig):
        super(LukePretrainingModel, self).__init__(config)

        if self.config.bert_model_name and "roberta" in self.config.bert_model_name:
            self.lm_head = RobertaLMHead(config)
            self.lm_head.decoder.weight = self.embeddings.word_embeddings.weight
        else:
            self.cls = BertPreTrainingHeads(config)
            self.cls.predictions.decoder.weight = self.embeddings.word_embeddings.weight

        self.entity_predictions = EntityPredictionHead(config)
        self.entity_predictions.decoder.weight = self.entity_embeddings.entity_embeddings.weight

        self.apply(self.init_weights)

    def forward(
        self,
        word_ids: torch.LongTensor,
        word_segment_ids: torch.LongTensor,
        word_attention_mask: torch.LongTensor,
        entity_ids: torch.LongTensor,
        entity_position_ids: torch.LongTensor,
        entity_segment_ids: torch.LongTensor,
        entity_attention_mask: torch.LongTensor,
        masked_entity_labels: Optional[torch.LongTensor] = None,
        masked_lm_labels: Optional[torch.LongTensor] = None,
        **kwargs
    ):
        model_dtype = next(self.parameters()).dtype  # for fp16 compatibility

        output = super(LukePretrainingModel, self).forward(
            word_ids,
            word_segment_ids,
            word_attention_mask,
            entity_ids,
            entity_position_ids,
            entity_segment_ids,
            entity_attention_mask,
        )
        word_sequence_output, entity_sequence_output = output[:2]

        loss_fn = CrossEntropyLoss(ignore_index=-1)
        ret = dict(loss=word_ids.new_tensor(0.0, dtype=model_dtype))

        if masked_entity_labels is not None:
            entity_mask = masked_entity_labels != -1
            if entity_mask.sum() > 0:
                target_entity_sequence_output = torch.masked_select(entity_sequence_output, entity_mask.unsqueeze(-1))
                target_entity_sequence_output = target_entity_sequence_output.view(-1, self.config.hidden_size)
                target_entity_labels = torch.masked_select(masked_entity_labels, entity_mask)

                entity_scores = self.entity_predictions(target_entity_sequence_output)
                entity_scores = entity_scores.view(-1, self.config.entity_vocab_size)

                ret["masked_entity_loss"] = loss_fn(entity_scores, target_entity_labels)
                ret["masked_entity_correct"] = (torch.argmax(entity_scores, 1).data == target_entity_labels.data).sum()
                ret["masked_entity_total"] = target_entity_labels.ne(-1).sum()
                ret["loss"] += ret["masked_entity_loss"]
            else:
                ret["masked_entity_loss"] = word_ids.new_tensor(0.0, dtype=model_dtype)
                ret["masked_entity_correct"] = word_ids.new_tensor(0, dtype=torch.long)
                ret["masked_entity_total"] = word_ids.new_tensor(0, dtype=torch.long)

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

                ret["masked_lm_loss"] = loss_fn(masked_lm_scores, masked_lm_labels)
                ret["masked_lm_correct"] = (torch.argmax(masked_lm_scores, 1).data == masked_lm_labels.data).sum()
                ret["masked_lm_total"] = masked_lm_labels.ne(-1).sum()
                ret["loss"] += ret["masked_lm_loss"]
            else:
                ret["masked_lm_loss"] = word_ids.new_tensor(0.0, dtype=model_dtype)
                ret["masked_lm_correct"] = word_ids.new_tensor(0, dtype=torch.long)
                ret["masked_lm_total"] = word_ids.new_tensor(0, dtype=torch.long)

        return ret
