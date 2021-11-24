import torch
import torch.nn.functional as F
from torch import nn

from luke.model import LukeModel
from luke.pretraining.model import EntityPredictionHead


class EntityEmbeddings(nn.Module):
    def __init__(self, config):
        super(EntityEmbeddings, self).__init__()
        self.config = config

        self.entity_embeddings = nn.Embedding(config.entity_vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.mask_embedding = nn.Parameter(torch.zeros(1, config.hidden_size))

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, entity_ids, position_ids, token_type_ids):
        entity_embeddings = self.entity_embeddings(entity_ids)
        entity_embeddings.masked_scatter_(
            (entity_ids == 1).unsqueeze(-1), self.mask_embedding.expand_as(entity_embeddings)
        )

        position_embeddings = self.position_embeddings(position_ids.clamp(min=0))
        position_embedding_mask = (position_ids != -1).type_as(position_embeddings).unsqueeze(-1)
        position_embeddings = position_embeddings * position_embedding_mask
        position_embeddings = torch.sum(position_embeddings, dim=-2)
        position_embeddings = position_embeddings / position_embedding_mask.sum(dim=-2).clamp(min=1e-7)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = entity_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class LukeForEntityDisambiguation(LukeModel):
    def __init__(self, config):
        super(LukeForEntityDisambiguation, self).__init__(config)

        self.entity_embeddings = EntityEmbeddings(config)
        self.entity_predictions = EntityPredictionHead(config)
        self.entity_predictions.decoder.weight = self.entity_embeddings.entity_embeddings.weight

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
        entity_candidate_ids=None,
        entity_labels=None,
    ):
        encoder_output = super(LukeForEntityDisambiguation, self).forward(
            word_ids,
            word_segment_ids,
            word_attention_mask,
            entity_ids,
            entity_position_ids,
            entity_segment_ids,
            entity_attention_mask,
        )
        logits = self.entity_predictions(encoder_output[1]).view(-1, self.config.entity_vocab_size)

        if entity_candidate_ids is not None:
            entity_candidate_ids = entity_candidate_ids.view(-1, entity_candidate_ids.size(-1))
            entity_candidate_mask = logits.new_zeros(logits.size(), dtype=torch.bool)
            entity_candidate_mask.scatter_(dim=1, index=entity_candidate_ids, src=(entity_candidate_ids != 0))
            logits = logits.masked_fill(~entity_candidate_mask, -1e32)

        logits = logits.view(entity_ids.size(0), -1, self.config.entity_vocab_size)

        if entity_labels is not None:
            loss = F.cross_entropy(
                logits.view(entity_labels.view(-1).size(0), -1), entity_labels.view(-1), ignore_index=-1
            )
            return loss, logits

        return (logits,)
