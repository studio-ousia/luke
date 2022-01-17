from typing import Dict
import torch

from allennlp.modules.token_embedders import TokenEmbedder
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.modules.seq2seq_encoders import PassThroughEncoder

from examples_allennlp.utils.span_utils import span_pooling, extract_span_start
from .feature_extractor import RCFeatureExtractor


@RCFeatureExtractor.register("token")
class TokenBasedRCFeatureExtractor(RCFeatureExtractor):
    def __init__(
        self, embedder: TokenEmbedder, encoder: Seq2SeqEncoder = None, feature_type: str = "entity_start",
    ):
        super().__init__()
        self.embedder = embedder
        self.encoder = encoder or PassThroughEncoder(input_dim=self.embedder.get_output_dim())

        self.feature_type = feature_type

    def get_output_dim(self):
        if self.feature_type == "cls_token":
            return self.encoder.get_output_dim()
        else:
            return self.encoder.get_output_dim() * 2

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        entity1_span: torch.LongTensor,
        entity2_span: torch.LongTensor,
        entity_ids: torch.LongTensor = None,
    ):
        token_embeddings = self.embedder(**inputs)
        token_embeddings = self.encoder(token_embeddings)

        if self.feature_type == "cls_token":
            feature_vector = token_embeddings[:, 0]
        elif self.feature_type == "mention_pooling":
            entity_1_features = span_pooling(token_embeddings, entity1_span)
            entity_2_features = span_pooling(token_embeddings, entity2_span)
            feature_vector = torch.cat([entity_1_features, entity_2_features], dim=1)
        elif self.feature_type == "entity_start":
            entity_1_features = extract_span_start(token_embeddings, entity1_span)
            entity_2_features = extract_span_start(token_embeddings, entity2_span)
            feature_vector = torch.cat([entity_1_features, entity_2_features], dim=1)
        else:
            raise ValueError(f"Invalid feature_type: {self.feature_type}")

        return feature_vector
