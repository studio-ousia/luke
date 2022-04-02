from typing import Dict

import torch
from allennlp.modules.token_embedders import TokenEmbedder

from examples.utils.span_utils import (
    extract_span_start,
    get_span_max_length,
    span_pooling,
    span_to_position_ids,
)

from .feature_extractor import RCFeatureExtractor


@RCFeatureExtractor.register("token+entity")
class TokenEntityBasedRCFeatureExtractor(RCFeatureExtractor):
    def __init__(
        self, embedder: TokenEmbedder, feature_type: str = "entity_start",
    ):
        super().__init__()
        self.embedder = embedder
        self.feature_type = feature_type

    def get_output_dim(self):
        if self.feature_type == "cls_token":
            word_feature_dim = self.embedder.get_output_dim()
        else:
            word_feature_dim = self.embedder.get_output_dim() * 2
        entity_feature_dim = self.embedder.get_output_dim() * 2
        return word_feature_dim + entity_feature_dim

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
        token_embeddings, entity_embeddings = self.embedder(**inputs)

        if self.feature_type == "cls_token":
            word_feature_vector = token_embeddings[:, 0]
        elif self.feature_type == "mention_pooling":
            entity_1_features = span_pooling(token_embeddings, entity1_span)
            entity_2_features = span_pooling(token_embeddings, entity2_span)
            word_feature_vector = torch.cat([entity_1_features, entity_2_features], dim=1)
        elif self.feature_type == "entity_start":
            entity_1_features = extract_span_start(token_embeddings, entity1_span)
            entity_2_features = extract_span_start(token_embeddings, entity2_span)
            word_feature_vector = torch.cat([entity_1_features, entity_2_features], dim=1)
        else:
            raise ValueError(f"Invalid feature_type: {self.feature_type}")

        batch_size = entity_embeddings.size(0)

        # (batch_size, 2, feature_size) -> (batch_size, 2 * feature_size)
        entity_feature_vector = entity_embeddings.view(batch_size, -1)

        return torch.cat([word_feature_vector, entity_feature_vector], dim=1)
