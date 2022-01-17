from typing import Dict
import torch

from allennlp.modules.token_embedders import TokenEmbedder

from examples_allennlp.utils.span_utils import get_span_max_length, span_to_position_ids
from .feature_extractor import RCFeatureExtractor


@RCFeatureExtractor.register("entity")
class EntityBasedRCFeatureExtractor(RCFeatureExtractor):
    def __init__(
        self, embedder: TokenEmbedder,
    ):
        super().__init__()
        self.embedder = embedder

    def get_output_dim(self):
        return self.embedder.get_output_dim() * 2

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        entity1_span: torch.LongTensor,
        entity2_span: torch.LongTensor,
        entity_ids: torch.LongTensor,
    ):
        max_position_length = max(get_span_max_length(entity1_span), get_span_max_length(entity2_span))
        entity_position_ids = torch.stack(
            [
                span_to_position_ids(entity1_span, max_position_length),
                span_to_position_ids(entity2_span, max_position_length),
            ],
            dim=1,
        )
        inputs["entity_position_ids"] = entity_position_ids
        inputs["entity_attention_mask"] = torch.ones_like(entity_ids)
        inputs["entity_ids"] = entity_ids
        embedder_outputs = self.embedder(**inputs)
        batch_size = embedder_outputs.size(0)

        # (batch_size, 2, feature_size) -> (batch_size, 2 * feature_size)
        return embedder_outputs.view(batch_size, -1)
