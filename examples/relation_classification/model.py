import torch
import torch.nn as nn
import torch.nn.functional as F

from ..word_entity_model import LukeWordEntityAttentionModel


class LukeForRelationClassification(LukeWordEntityAttentionModel):
    def __init__(self, args, num_labels):
        super(LukeForRelationClassification, self).__init__(args)

        self.args = args

        if self.args.use_difference_feature:
            feature_size = args.model_config.hidden_size * 3
        else:
            feature_size = args.model_config.hidden_size * 2

        self.num_labels = num_labels
        self.dropout = nn.Dropout(args.dropout_prob)
        self.classifier = nn.Linear(feature_size, num_labels, False)

        self.apply(self.init_weights)

    def forward(self, word_ids, word_segment_ids, word_attention_mask, entity_ids, entity_position_ids,
                entity_segment_ids, entity_attention_mask, label=None):
        encoder_outputs = super(LukeForRelationClassification, self).forward(
            word_ids, word_segment_ids, word_attention_mask, entity_ids, entity_position_ids, entity_segment_ids,
            entity_attention_mask)

        feature_vector = torch.cat([encoder_outputs[1][:, 0, :], encoder_outputs[1][:, 1, :]], dim=1)

        if self.args.use_difference_feature:
            diff_feature_vector = torch.abs(encoder_outputs[1][:, 0, :] - encoder_outputs[1][:, 1, :])
            feature_vector = torch.cat([feature_vector, diff_feature_vector], dim=1)

        feature_vector = self.dropout(feature_vector)
        logits = self.classifier(feature_vector)
        if label is None:
            return logits

        return (F.cross_entropy(logits, label),)
