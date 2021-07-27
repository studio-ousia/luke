import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from luke.model import LukeEntityAwareAttentionModel


class LukeForNamedEntityRecognition(LukeEntityAwareAttentionModel):
    def __init__(self, args, num_labels):
        super(LukeForNamedEntityRecognition, self).__init__(args.model_config)
        self.args = args
        self.num_labels = num_labels

        self.dropout = nn.Dropout(args.model_config.hidden_dropout_prob)
        if args.no_word_feature:
            self.classifier = nn.Linear(args.model_config.hidden_size, num_labels)
        elif args.no_entity_feature:
            self.classifier = nn.Linear(args.model_config.hidden_size * 2, num_labels)
        else:
            self.classifier = nn.Linear(args.model_config.hidden_size * 3, num_labels)

        self.apply(self.init_weights)

    def forward(
        self,
        word_ids,
        word_segment_ids,
        word_attention_mask,
        entity_start_positions,
        entity_end_positions,
        entity_ids,
        entity_position_ids,
        entity_segment_ids,
        entity_attention_mask,
        labels=None,
    ):
        encoder_outputs = super(LukeForNamedEntityRecognition, self).forward(
            word_ids,
            word_segment_ids,
            word_attention_mask,
            entity_ids,
            entity_position_ids,
            entity_segment_ids,
            entity_attention_mask,
        )

        word_hidden_states, entity_hidden_states = encoder_outputs[:2]
        hidden_size = word_hidden_states.size()[-1]

        if self.args.no_word_feature:
            feature_vector = entity_hidden_states

        else:
            entity_start_positions = entity_start_positions.unsqueeze(-1).expand(-1, -1, hidden_size)
            start_states = torch.gather(word_hidden_states, -2, entity_start_positions)
            entity_end_positions = entity_end_positions.unsqueeze(-1).expand(-1, -1, hidden_size)
            end_states = torch.gather(word_hidden_states, -2, entity_end_positions)
            if self.args.no_entity_feature:
                feature_vector = torch.cat([start_states, end_states], dim=2)
            else:
                feature_vector = torch.cat([start_states, end_states, entity_hidden_states], dim=2)

        feature_vector = self.dropout(feature_vector)
        logits = self.classifier(feature_vector)

        if labels is None:
            return logits

        loss_fn = CrossEntropyLoss(ignore_index=-1)
        return (loss_fn(logits.view(-1, self.num_labels), labels.view(-1)),)
