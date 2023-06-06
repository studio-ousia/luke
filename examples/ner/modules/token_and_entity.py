from typing import Dict

import torch
from allennlp.modules.token_embedders import TokenEmbedder

from .feature_extractor import NERFeatureExtractor


@NERFeatureExtractor.register("token+entity")
class TokenEntityNERFeatureExtractor(NERFeatureExtractor):
    def __init__(
        self, embedder: TokenEmbedder,
    ):
        super().__init__()
        self.embedder = embedder

    def get_output_dim(self):
        return self.embedder.get_output_dim() * 3

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        entity_start_positions: torch.LongTensor,
        entity_end_positions: torch.LongTensor,
        entity_ids: torch.LongTensor = None,
        entity_position_ids: torch.LongTensor = None,
        entity_attention_mask: torch.LongTensor = None,
    ):

        inputs["entity_ids"] = entity_ids
        inputs["entity_position_ids"] = entity_position_ids
        inputs["entity_attention_mask"] = entity_attention_mask

        token_embeddings, entity_embeddings = self.embedder(**inputs)
        embedding_size = token_embeddings.size(-1)

        entity_start_positions = entity_start_positions.unsqueeze(-1).expand(-1, -1, embedding_size)
        start_embeddings = torch.gather(token_embeddings, -2, entity_start_positions)

        entity_end_positions = entity_end_positions.unsqueeze(-1).expand(-1, -1, embedding_size)
        end_embeddings = torch.gather(token_embeddings, -2, entity_end_positions)

        feature_vector = torch.cat([start_embeddings, end_embeddings, entity_embeddings], dim=2)

        return feature_vector
