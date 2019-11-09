import functools

import click
import torch
import torch.nn as nn
import torch.nn.functional as F

from luke.model import LukeModel
from luke.pretraining.model import EntityPredictionHead
from luke.utils.entity_vocab import MASK_TOKEN, UNK_TOKEN


def two_stage_model_args(func):
    @click.option('--min-context-entity-prob', default=0.0)
    @click.option('--use-softmax-average', is_flag=True)
    @click.option('--entity-softmax-temp', default=0.1)
    @click.option('--update-params-in-disambi', is_flag=True)
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


class LukeTwoStageBaseModel(LukeModel):
    def __init__(self, args):
        super(LukeTwoStageBaseModel, self).__init__(args.model_config)

        self.args = args
        self.entity_mask_id = args.entity_vocab[MASK_TOKEN]
        self.entity_unk_id = args.entity_vocab[UNK_TOKEN]

        self.entity_predictions = EntityPredictionHead(self.config)
        self.entity_predictions.decoder.weight = self.entity_embeddings.entity_embeddings.weight
        self.entity_prediction_bias = nn.Embedding(args.model_config.entity_vocab_size, 1, padding_idx=0)
        self.entity_prediction_bias.weight.data = self.entity_predictions.bias.data.view(-1, 1)

    def forward(self, word_ids, word_segment_ids, word_attention_mask, entity_candidate_ids, entity_position_ids,
                entity_segment_ids, entity_attention_mask):
        if self.args.use_softmax_average:
            extended_attention_mask = self._compute_extended_attention_mask(word_attention_mask, entity_attention_mask)
            word_embeddings = self.embeddings(word_ids, word_segment_ids)
            mask_embeddings = self.entity_embeddings(entity_attention_mask * self.entity_mask_id,
                                                     entity_position_ids, entity_segment_ids)
            encoder_outputs = self.encoder(torch.cat([word_embeddings, mask_embeddings], dim=1),
                                           extended_attention_mask, [None] * self.config.num_hidden_layers)
            mask_sequence_output = encoder_outputs[0][:, word_ids.size(1):, :]
            mask_sequence_output = self.entity_predictions.transform(mask_sequence_output)
            if self.config.entity_emb_size != self.config.hidden_size:
                mask_sequence_output = self.entity_predictions.pre_decoder_dense(mask_sequence_output)
            candidate_embeddings = self.entity_embeddings.entity_embeddings(entity_candidate_ids)

            attention_logits = (mask_sequence_output.unsqueeze(2) * candidate_embeddings).sum(-1)
            attention_bias = self.entity_prediction_bias(entity_candidate_ids).squeeze(-1)
            attention_logits = attention_logits + attention_bias
            attention_logits = attention_logits / self.args.entity_softmax_temp
            attention_logits.masked_fill_(entity_candidate_ids == 0, -10000.0)
            attention_probs = F.softmax(attention_logits, dim=-1)
            if not self.args.update_params_in_disambi:
                attention_probs = attention_probs.detach()

            entity_embeddings = self.entity_embeddings(entity_candidate_ids, entity_position_ids.unsqueeze(-2),
                                                       entity_segment_ids.unsqueeze(-1))
            entity_embeddings = (entity_embeddings * attention_probs.unsqueeze(-1)).sum(-2)

            if self.args.min_context_entity_prob != 0.0:
                mask = (attention_probs.max(2)[0] >= self.args.min_context_entity_prob).long()
                entity_attention_mask = entity_attention_mask * mask

            extended_attention_mask = self._compute_extended_attention_mask(word_attention_mask, entity_attention_mask)

            return self.encoder(torch.cat([word_embeddings, entity_embeddings], dim=1),
                                extended_attention_mask, [None] * self.config.num_hidden_layers)

        else:
            with torch.no_grad():
                mask_entity_ids = entity_attention_mask * self.entity_mask_id
                encoder_outputs = super(LukeTwoStageBaseModel, self).forward(
                    word_ids, word_segment_ids, word_attention_mask, mask_entity_ids, entity_position_ids,
                    entity_segment_ids, entity_attention_mask
                )
                logits = self.entity_predictions(encoder_outputs[1]).view(-1, self.config.entity_vocab_size)
                entity_candidate_ids = entity_candidate_ids.reshape(-1, entity_candidate_ids.size(2))
                entity_candidate_mask = logits.new_zeros(logits.size(), dtype=torch.bool)
                entity_candidate_mask.scatter_(dim=1, index=entity_candidate_ids, src=(entity_candidate_ids != 0))
                logits = logits.masked_fill(~entity_candidate_mask, -10000.0).view(mask_entity_ids.size(0), -1,
                                                                                   self.config.entity_vocab_size)

                predicted_entity_ids = logits.argmax(2) * entity_attention_mask
                predicted_entity_ids = predicted_entity_ids * (predicted_entity_ids != self.entity_unk_id).long()
                entity_attention_mask = (predicted_entity_ids != 0).long()

                if self.args.min_context_entity_prob != 0.0:
                    entity_probs = F.softmax(logits, dim=2)
                    mask = (entity_probs.max(2)[0] >= self.args.min_context_entity_prob).long()
                    predicted_entity_ids = predicted_entity_ids * mask
                    entity_attention_mask = entity_attention_mask * mask

            return super(LukeTwoStageBaseModel, self).forward(
                word_ids, word_segment_ids, word_attention_mask, predicted_entity_ids, entity_position_ids,
                entity_segment_ids, entity_attention_mask)
