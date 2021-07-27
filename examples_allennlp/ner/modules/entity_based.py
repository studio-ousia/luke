from typing import Dict
import torch
from allennlp.modules.token_embedders import TokenEmbedder

from .feature_extractor import NERFeatureExtractor


@NERFeatureExtractor.register("entity")
class EntityBasedNERFeatureExtractor(NERFeatureExtractor):
    def __init__(
        self, embedder: TokenEmbedder,
    ):
        super().__init__()
        self.embedder = embedder

    def get_output_dim(self):
        return self.embedder.get_output_dim()

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        entity_start_positions: torch.LongTensor,
        entity_end_positions: torch.LongTensor,
        entity_ids: torch.LongTensor = None,
        entity_position_ids: torch.LongTensor = None,
    ):

        inputs["entity_ids"] = entity_ids
        inputs["entity_position_ids"] = entity_position_ids
        inputs["entity_attention_mask"] = entity_ids != 0

        entity_embeddings = self.embedder(**inputs)
        return entity_embeddings
