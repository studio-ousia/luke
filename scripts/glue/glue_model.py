# -*- coding: utf-8 -*-

from torch import nn
from torch.nn import CrossEntropyLoss

from luke.model import LukeModel


class LukeForSequenceClassification(LukeModel):
    def __init__(self, config, num_labels):
        super(LukeForSequenceClassification, self).__init__(config)

        self.num_labels = num_labels
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

        self.apply(self.init_weights)

    def forward(self, word_ids, word_segment_ids, word_attention_mask, entity_ids,
                entity_position_ids, entity_segment_ids, entity_attention_mask, labels=None):
        # print('')
        # print('')
        # print(word_ids, word_segment_ids, word_attention_mask)
        # print(entity_ids, entity_position_ids, entity_segment_ids, entity_attention_mask)
        # print('')
        # print('')
        (_, pooled_output) = super(LukeForSequenceClassification, self).forward(
            word_ids, word_segment_ids, word_attention_mask, entity_ids, entity_position_ids,
            entity_segment_ids, entity_attention_mask, output_all_encoded_layers=False)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss

        else:
            return logits