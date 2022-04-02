import torch
import torch.nn as nn
import torch.nn.functional as F

from luke.model import LukeEntityAwareAttentionModel


class LukeForRelationClassification(LukeEntityAwareAttentionModel):
    def __init__(self, args, num_labels):
        super(LukeForRelationClassification, self).__init__(args.model_config)

        self.args = args

        self.num_labels = num_labels
        self.dropout = nn.Dropout(args.model_config.hidden_dropout_prob)
        self.classifier = nn.Linear(args.model_config.hidden_size * 2, num_labels, False)

        self.apply(self.init_weights)

    def forward(
        self,
        word_ids,
        word_segment_ids,
        word_attention_mask,
        entity_ids,
        entity_position_ids,
        entity_segment_ids,
        entity_attention_mask,
        label=None,
    ):
        encoder_outputs = super(LukeForRelationClassification, self).forward(
            word_ids,
            word_segment_ids,
            word_attention_mask,
            entity_ids,
            entity_position_ids,
            entity_segment_ids,
            entity_attention_mask,
        )

        feature_vector = torch.cat([encoder_outputs[1][:, 0, :], encoder_outputs[1][:, 1, :]], dim=1)
        feature_vector = self.dropout(feature_vector)

        logits = self.classifier(feature_vector)
        if label is None:
            return logits

        return (F.cross_entropy(logits, label),)
