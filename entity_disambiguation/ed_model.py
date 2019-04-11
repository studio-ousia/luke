# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from model import LukeConfig, LukeModel, EntityPredictionHead


class LukeConfigForEntityDisambiguation(LukeConfig):
    def __init__(self, prior_prob_bin_size, entity_prior_bin_size, *args, **kwargs):
        super(LukeConfigForEntityDisambiguation, self).__init__(*args, **kwargs)
        self.prior_prob_bin_size = prior_prob_bin_size
        self.entity_prior_bin_size = entity_prior_bin_size


class LukeForEntityDisambiguation(LukeModel):
    def __init__(self, config):
        super(LukeForEntityDisambiguation, self).__init__(config)

        self.entity_predictions = EntityPredictionHead(config,
            self.entity_embeddings.entity_embeddings.weight)
        if config.prior_prob_bin_size != 0:
            self.prior_prob_bias_embeddings = nn.Embedding(config.prior_prob_bin_size, 1)
        if config.entity_emb_size != 0:
            self.entity_prior_bias_embeddings = nn.Embedding(config.entity_prior_bin_size, 1)

        self.apply(self.init_weights)
        self.prior_prob_bias_embeddings.weight.data.fill_(0)
        self.entity_prior_bias_embeddings.weight.data.fill_(0)

    def forward(self, word_ids, word_segment_ids, word_attention_mask, entity_ids,
                entity_position_ids, entity_segment_ids, entity_attention_mask,
                entity_candidate_ids, entity_prior_prob_ids, entity_prior_ids, entity_label=None):
        (encoded_layers, _) = super(LukeForEntityDisambiguation, self).forward(
            word_ids, word_segment_ids, word_attention_mask, entity_ids, entity_position_ids,
            entity_segment_ids, entity_attention_mask, output_all_encoded_layers=False)

        entity_output = encoded_layers[1][:, 0]

        logits = self.entity_predictions(entity_output).view(-1, self.config.entity_vocab_size)
        entity_candidate_mask = logits.new_full(logits.size(), 0, dtype=torch.uint8)
        entity_candidate_mask.scatter_(dim=1, index=entity_candidate_ids,
                                       src=(entity_candidate_ids != 0))
        if self.config.prior_prob_bin_size != 0:
            prior_prob_emb = self.prior_prob_bias_embeddings(entity_prior_prob_ids).squeeze(-1)
            prior_prob_bias = logits.new_full(logits.size(), 0)
            prior_prob_bias.scatter_(dim=1, index=entity_candidate_ids, src=prior_prob_emb)
            logits = logits + prior_prob_bias

        if self.config.entity_prior_bin_size != 0:
            entity_prior_emb = self.entity_prior_bias_embeddings(entity_prior_ids).squeeze(-1)
            entity_prior_bias = logits.new_full(logits.size(), 0)
            entity_prior_bias.scatter_(dim=1, index=entity_candidate_ids, src=entity_prior_emb)
            logits = logits + entity_prior_bias

        masked_logits = logits.masked_fill((1 - entity_candidate_mask), -1e32)

        if entity_label is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(masked_logits, entity_label)
            return loss
        else:
            return masked_logits
