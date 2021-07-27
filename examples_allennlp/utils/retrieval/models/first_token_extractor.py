import torch
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.modules import TextFieldEmbedder

from .seq2vec_encoder import Seq2VecEncoder


@Seq2VecEncoder.register("first_token")
class FirstTokenExtractor(Seq2VecEncoder):
    def __init__(self, vocab: Vocabulary, embedder: TextFieldEmbedder) -> None:
        super().__init__(vocab=vocab)
        self.embedder = embedder

    def forward(self, tokens: TextFieldTensors) -> torch.Tensor:
        embeddings = self.embedder(tokens)
        return embeddings[:, 0]
