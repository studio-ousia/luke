# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from luke.model import LukeModel, LukeConfig

class LukeForRerankingConfig(LukeConfig):
    def __init__(self, scalar_mix, **kwargs):
        super(LukeForRerankingConfig, self).__init__(**kwargs)
        self.scalar_mix = scalar_mix


class LukeForReranking(LukeModel):
    def __init__(self, config):
        super(LukeForReranking, self).__init__(config)

        self.output_layer = nn.Linear(config.hidden_size, 2)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        if config.scalar_mix:
            self.scalar_parameters = nn.ParameterList(
                [nn.Parameter(torch.FloatTensor([0.0])) for i in range(config.num_hidden_layers)])
            self.gamma = nn.Parameter(torch.FloatTensor([1.0]))

        self.apply(self.init_weights)

    def forward(self, word_ids, word_segment_ids, word_attention_mask, entity_ids,
                entity_position_ids, entity_segment_ids, entity_attention_mask, label=None):
        (encoded_layers, _) = super(LukeForReranking, self).forward(
            word_ids, word_segment_ids, word_attention_mask, entity_ids, entity_position_ids,
            entity_segment_ids, entity_attention_mask, output_all_encoded_layers=True)

        if self.config.scalar_mix:
            cls_states = torch.cat([w[:, :1] for (w, e) in encoded_layers], dim=1)
            normed_weights = nn.functional.softmax(torch.cat([p for p in self.scalar_parameters]),
                                                   dim=0)
            cls_states = cls_states * normed_weights.unsqueeze(0).unsqueeze(2)
            cls_states = self.gamma * torch.sum(cls_states, dim=1, keepdim=True)
            pooled_output = self.pooler(cls_states)

        else:
            word_sequence_output = encoded_layers[-1][0]
            pooled_output = self.pooler(word_sequence_output)

        pooled_output = self.dropout(pooled_output)
        logits = self.output_layer(pooled_output)

        if label is not None:
            ret = {}
            loss_fct = CrossEntropyLoss()
            ret['loss'] = loss_fct(logits, label)
            ret['correct'] = (torch.argmax(logits, 1).data == label.data).sum()
            ret['total'] = ret['correct'].new_tensor(word_ids.size(0))
            return ret

        else:
            return logits
