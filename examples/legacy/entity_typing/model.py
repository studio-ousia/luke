import torch.nn as nn
import torch.nn.functional as F

from luke.model import LukeEntityAwareAttentionModel


class LukeForEntityTyping(LukeEntityAwareAttentionModel):
    def __init__(self, args, num_labels):
        super(LukeForEntityTyping, self).__init__(args.model_config)

        self.num_labels = num_labels
        self.dropout = nn.Dropout(args.model_config.hidden_dropout_prob)
        self.typing = nn.Linear(args.model_config.hidden_size, num_labels)

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
        labels=None,
    ):
        encoder_outputs = super(LukeForEntityTyping, self).forward(
            word_ids,
            word_segment_ids,
            word_attention_mask,
            entity_ids,
            entity_position_ids,
            entity_segment_ids,
            entity_attention_mask,
        )

        feature_vector = encoder_outputs[1][:, 0, :]
        feature_vector = self.dropout(feature_vector)
        logits = self.typing(feature_vector)
        if labels is None:
            return logits

        return (F.binary_cross_entropy_with_logits(logits.view(-1), labels.view(-1).type_as(logits)),)
