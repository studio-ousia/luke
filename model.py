# -*- coding: utf-8 -*-

import logging
import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from model_common import BaseConfig, BaseModel, WordEmbeddings, LayerNorm, Encoder, Pooler,\
    PredictionHeadTransform, BertPreTrainingHeads

logger = logging.getLogger(__name__)


class LukeConfig(BaseConfig):
    def __init__(self, vocab_size, entity_vocab_size, hidden_size, entity_emb_size,
                 num_hidden_layers, num_attention_heads, intermediate_size, hidden_dropout_prob,
                 attention_probs_dropout_prob, max_position_embeddings, type_vocab_size,
                 initializer_range, **kwargs):
        self.vocab_size = vocab_size
        self.entity_vocab_size = entity_vocab_size
        self.hidden_size = hidden_size
        self.entity_emb_size = entity_emb_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range


class EntityEmbeddings(nn.Module):
    def __init__(self, config):
        super(EntityEmbeddings, self).__init__()
        self.config = config

        self.entity_embeddings = nn.Embedding(config.entity_vocab_size, config.entity_emb_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        if config.entity_emb_size != config.hidden_size:
            self.entity_dense = nn.Linear(config.entity_emb_size, config.hidden_size, bias=False)

        self.LayerNorm = LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, entity_ids, position_ids, token_type_ids):
        entity_embeddings = self.entity_embeddings(entity_ids)
        if self.config.entity_emb_size != self.config.hidden_size:
            entity_embeddings = self.entity_dense(entity_embeddings)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = entity_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class EntityPredictionHead(nn.Module):
    def __init__(self, config, entity_embedding_weights):
        super(EntityPredictionHead, self).__init__()
        self.transform = PredictionHeadTransform(config, entity_embedding_weights.size(1))

        self.decoder = nn.Linear(entity_embedding_weights.size(1),
                                 entity_embedding_weights.size(0),
                                 bias=False)
        self.decoder.weight = entity_embedding_weights
        self.bias = nn.Parameter(torch.zeros(entity_embedding_weights.size(0)))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        return self.decoder(hidden_states) + self.bias


class LukeModel(BaseModel):
    def __init__(self, config):
        super(LukeModel, self).__init__(config)

        self.encoder = Encoder(config)
        self.pooler = Pooler(config)
        self.embeddings = WordEmbeddings(config)
        self.entity_embeddings = EntityEmbeddings(config)

    def forward(self, word_ids, word_segment_ids, word_attention_mask, entity_ids,
                entity_position_ids, entity_segment_ids, entity_attention_mask,
                output_all_encoded_layers=True):
        attention_mask = torch.cat([word_attention_mask, entity_attention_mask], dim=1)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        word_embedding_output = self.embeddings(word_ids, word_segment_ids)
        entity_embedding_output = self.entity_embeddings(entity_ids, entity_position_ids,
                                                         entity_segment_ids)
        encoded_layers = self.encoder(word_embedding_output, entity_embedding_output,
                                      extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers)
        word_sequence_output = encoded_layers[-1][0]
        pooled_output = self.pooler(word_sequence_output)

        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]

        return (encoded_layers, pooled_output)


class LukePretrainingModel(LukeModel):
    def __init__(self, config):
        super(LukePretrainingModel, self).__init__(config)

        self.cls = BertPreTrainingHeads(config, self.embeddings.word_embeddings.weight)
        self.entity_predictions = EntityPredictionHead(config,
            self.entity_embeddings.entity_embeddings.weight)

        self.apply(self.init_weights)

    def forward(self, word_ids, word_segment_ids, word_attention_mask, entity_ids,
                entity_position_ids, entity_segment_ids, entity_attention_mask,
                masked_lm_labels, masked_entity_labels, is_random_next=None, **kwargs):
        (encoded_layers, pooled_output) = super(LukePretrainingModel, self).forward(
            word_ids, word_segment_ids, word_attention_mask, entity_ids, entity_position_ids,
            entity_segment_ids, entity_attention_mask, output_all_encoded_layers=False
        )
        word_sequence_output = encoded_layers[0]
        entity_sequence_output = encoded_layers[1]

        loss_fn = CrossEntropyLoss(ignore_index=-1)
        ret = {}

        masked_lm_mask = (masked_lm_labels != -1)
        masked_word_sequence_output = torch.masked_select(
            word_sequence_output, masked_lm_mask.unsqueeze(-1)).view(-1, self.config.hidden_size)
        (masked_lm_scores, nsp_score) = self.cls(masked_word_sequence_output, pooled_output)

        masked_lm_scores = masked_lm_scores.view(-1, self.config.vocab_size)
        masked_lm_labels = torch.masked_select(masked_lm_labels, masked_lm_mask)
        # masked_lm_labels = masked_lm_labels.view(-1)
        ret['masked_lm_loss'] = loss_fn(masked_lm_scores, masked_lm_labels)
        ret['masked_lm_correct'] = (torch.argmax(masked_lm_scores, 1).data ==
                                    masked_lm_labels.data).sum()
        ret['masked_lm_total'] = masked_lm_labels.ne(-1).sum()
        ret['loss'] = ret['masked_lm_loss']

        if is_random_next is not None:
            ret['nsp_loss'] = loss_fn(nsp_score, is_random_next)
            ret['nsp_correct'] = (torch.argmax(nsp_score, 1).data == is_random_next.data).sum()
            ret['nsp_total'] = ret['nsp_correct'].new_tensor(word_ids.size(0))
            ret['loss'] += ret['nsp_loss']

        entity_mask = (masked_entity_labels != -1)
        masked_entity_sequence_output = torch.masked_select(
            entity_sequence_output, entity_mask.unsqueeze(-1)).view(-1, self.config.hidden_size)
        entity_scores = self.entity_predictions(masked_entity_sequence_output)

        entity_scores = entity_scores.view(-1, self.config.entity_vocab_size)
        entity_labels = torch.masked_select(masked_entity_labels, entity_mask)
        ret['masked_entity_loss'] = loss_fn(entity_scores, entity_labels)
        ret['masked_entity_correct'] = (torch.argmax(entity_scores, 1).data == entity_labels.data).sum()
        ret['masked_entity_total'] = entity_labels.ne(-1).sum()
        ret['loss'] += ret['masked_entity_loss']

        return ret
