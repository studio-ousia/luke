from typing import Dict

import torch
from allennlp.modules.seq2seq_encoders import PassThroughEncoder, Seq2SeqEncoder
from allennlp.modules.token_embedders import TokenEmbedder

from .feature_extractor import NERFeatureExtractor


@NERFeatureExtractor.register("token")
class TokenBasedNERFeatureExtractor(NERFeatureExtractor):
    def __init__(
        self, embedder: TokenEmbedder, encoder: Seq2SeqEncoder = None,
    ):
        super().__init__()
        self.embedder = embedder
        self.encoder = encoder or PassThroughEncoder(input_dim=self.embedder.get_output_dim())

    def get_output_dim(self):
        return self.encoder.get_output_dim() * 2

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        entity_start_positions: torch.LongTensor,
        entity_end_positions: torch.LongTensor,
        entity_ids: torch.LongTensor = None,
        entity_position_ids: torch.LongTensor = None,
        entity_attention_mask: torch.LongTensor = None,
    ):
        token_embeddings = self.embedder(**inputs)
        token_embeddings = self.encoder(token_embeddings)

        embedding_size = token_embeddings.size(-1)

        entity_start_positions = entity_start_positions.unsqueeze(-1).expand(-1, -1, embedding_size)
        start_embeddings = torch.gather(token_embeddings, -2, entity_start_positions)

        entity_end_positions = entity_end_positions.unsqueeze(-1).expand(-1, -1, embedding_size)
        end_embeddings = torch.gather(token_embeddings, -2, entity_end_positions)

        feature_vector = torch.cat([start_embeddings, end_embeddings], dim=2)

        return feature_vector
