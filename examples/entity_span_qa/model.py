import torch
import torch.nn as nn
import torch.nn.functional as F

from luke.model import LukeEntityAwareAttentionModel


class LukeForEntitySpanQA(LukeEntityAwareAttentionModel):
    def __init__(self, args):
        super(LukeForEntitySpanQA, self).__init__(args.model_config)
        self.args = args

        self.dropout = nn.Dropout(args.model_config.hidden_dropout_prob)
        self.scorer = nn.Linear(args.model_config.hidden_size * 2, 1)

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
        encoder_outputs = super(LukeForEntitySpanQA, self).forward(
            word_ids,
            word_segment_ids,
            word_attention_mask,
            entity_ids,
            entity_position_ids,
            entity_segment_ids,
            entity_attention_mask,
        )

        entity_hidden_states = encoder_outputs[1]
        doc_entity_emb = entity_hidden_states[:, 1:, :]
        placeholder_emb = entity_hidden_states[:, :1, :]

        feature_vector = torch.cat([placeholder_emb.expand_as(doc_entity_emb), doc_entity_emb], dim=2)
        feature_vector = self.dropout(feature_vector)
        logits = self.scorer(feature_vector)

        doc_entity_mask = entity_attention_mask[:, 1:]
        if labels is None:
            return logits.squeeze(-1) + ((doc_entity_mask - 1) * 10000).type_as(logits)

        loss = F.binary_cross_entropy_with_logits(logits.view(-1), labels.view(-1).type_as(logits), reduce=False)
        loss = loss.masked_select(doc_entity_mask.reshape(-1).bool()).sum()
        loss = loss / doc_entity_mask.sum().type_as(loss)
        return (loss,)
