from typing import Dict
import torch

from allennlp.modules.token_embedders import TokenEmbedder

from examples_allennlp.utils.span_utils import span_to_position_ids
from .feature_extractor import ETFeatureExtractor


@ETFeatureExtractor.register("entity")
class EntityBasedETFeatureExtractor(ETFeatureExtractor):
    def __init__(
        self, embedder: TokenEmbedder,
    ):
        super().__init__()
        self.embedder = embedder

    def get_output_dim(self):
        return self.embedder.get_output_dim()

    def forward(
        self, inputs: Dict[str, torch.Tensor], entity_span: torch.LongTensor, entity_ids: torch.LongTensor,
    ):
        inputs["entity_position_ids"] = span_to_position_ids(entity_span).unsqueeze(1)
        inputs["entity_attention_mask"] = torch.ones_like(entity_ids)
        inputs["entity_ids"] = entity_ids
        embedder_output = self.embedder(**inputs)

        # (batch_size, num_entities=1, embedding_size) -> (batch_size, embedding_size)
        return embedder_output.squeeze(1)
