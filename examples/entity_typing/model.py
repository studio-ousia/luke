import torch.nn as nn
import torch.nn.functional as F
from luke.pretraining.model import EntityPredictionHead

from ..word_entity_model import LukeWordEntityAttentionModel


class LukeForEntityTyping(LukeWordEntityAttentionModel):
    def __init__(self, args, num_labels):
        super(LukeForEntityTyping, self).__init__(args)

        self.args = args
        if self.args.use_entity_head:
            feature_size = args.model_config.entity_emb_size
        else:
            feature_size = args.model_config.hidden_size

        self.num_labels = num_labels
        self.dropout = nn.Dropout(args.dropout_prob)
        self.typing = nn.Linear(feature_size, num_labels, False)

        if args.use_entity_head:
            self.entity_predictions = EntityPredictionHead(args.model_config)

        if args.use_hidden_layer:
            self.dense = nn.Linear(feature_size, feature_size)
            self.activation = nn.Tanh()

        self.apply(self.init_weights)

    def forward(self, word_ids, word_segment_ids, word_attention_mask, entity_ids, entity_position_ids,
                entity_segment_ids, entity_attention_mask, labels=None):
        encoder_outputs = super(LukeForEntityTyping, self).forward(
            word_ids, word_segment_ids, word_attention_mask, entity_ids, entity_position_ids, entity_segment_ids,
            entity_attention_mask)

        feature_vector = encoder_outputs[1][:, 0, :]
        if self.args.use_entity_head:
            feature_vector = self.entity_predictions.transform(feature_vector)

        if self.args.use_hidden_layer:
            feature_vector = self.dense(feature_vector)
            feature_vector = self.activation(feature_vector)

        feature_vector = self.dropout(feature_vector)
        logits = self.typing(feature_vector)
        if labels is None:
            return logits

        return (F.binary_cross_entropy_with_logits(logits.view(-1, self.num_labels),
                                                   labels.view(-1, self.num_labels).type_as(logits)),)
