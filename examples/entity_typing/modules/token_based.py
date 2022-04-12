from typing import Dict

import torch
from allennlp.modules.seq2seq_encoders import PassThroughEncoder, Seq2SeqEncoder
from allennlp.modules.token_embedders import TokenEmbedder

from examples.utils.span_utils import extract_span_start, span_pooling

from .feature_extractor import ETFeatureExtractor


@ETFeatureExtractor.register("token")
class TokenBasedETFeatureExtractor(ETFeatureExtractor):
    def __init__(
        self, embedder: TokenEmbedder, encoder: Seq2SeqEncoder = None, feature_type: str = "entity_start",
    ):
        super().__init__()
        self.embedder = embedder
        self.encoder = encoder or PassThroughEncoder(input_dim=self.embedder.get_output_dim())

        self.feature_type = feature_type

    def get_output_dim(self):
        return self.encoder.get_output_dim()

    def forward(self, inputs: Dict[str, torch.Tensor], entity_span: torch.LongTensor, entity_ids: torch.LongTensor):
        token_embeddings = self.embedder(**inputs)
        token_embeddings = self.encoder(token_embeddings)

        if self.feature_type == "cls_token":
            feature_vector = token_embeddings[:, 0]
        elif self.feature_type == "mention_pooling":
            feature_vector = span_pooling(token_embeddings, entity_span)
        elif self.feature_type == "entity_start":
            feature_vector = extract_span_start(token_embeddings, entity_span)
        else:
            raise ValueError(f"Invalid feature_type: {self.feature_type}")

        return feature_vector
