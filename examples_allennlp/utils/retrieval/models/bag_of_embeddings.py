from typing import List
import torch
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.modules import TextFieldEmbedder
from allennlp.nn import util

from .seq2vec_encoder import Seq2VecEncoder


def masked_mean_pooling(embedding_sequence: torch.Tensor, mask: torch.Tensor):
    embedding_sequence = embedding_sequence * mask.unsqueeze(-1).float()

    summed_embeddings = embedding_sequence.sum(dim=1)  # shape: (batch_size, embedding_dim)

    lengths = mask.sum(dim=1)  # shape: (batch_size, )
    length_mask = lengths > 0

    # Set any length 0 to 1, to avoid dividing by zero.
    lengths = torch.max(lengths, lengths.new_ones(1))

    mean_pooled_embeddings = summed_embeddings / lengths.unsqueeze(-1).float()

    # mask embeddings with length 0
    mean_pooled_embeddings = mean_pooled_embeddings * (length_mask > 0).float().unsqueeze(-1)

    return mean_pooled_embeddings


def get_last_indices_from_mask(mask: torch.Tensor) -> List[int]:
    last_index = []
    for m in mask:
        zero_indices = (m == 0).nonzero(as_tuple=True)[0]
        if len(zero_indices) == 0:
            index = -1
        else:
            index = (zero_indices[0] - 1).item()
        last_index.append(index)
    return last_index


@Seq2VecEncoder.register("boe")
class BoeEncoder(Seq2VecEncoder):
    def __init__(
        self, vocab: Vocabulary, embedder: TextFieldEmbedder, averaged: bool = False, mask_first_and_last: bool = False
    ) -> None:
        super().__init__(vocab=vocab)
        self.embedder = embedder
        self.averaged = averaged
        self.mask_first_and_last = mask_first_and_last

    def forward(self, tokens: TextFieldTensors) -> torch.Tensor:
        embedding_sequence = self.embedder(tokens)
        mask = util.get_text_field_mask(tokens)

        if self.mask_first_and_last:
            last_indices = get_last_indices_from_mask(mask)
            batch_size = mask.size(0)
            mask[range(batch_size), last_indices] = 0
            mask[:, 0] = 0

        if self.averaged:
            return masked_mean_pooling(embedding_sequence, mask)
        else:
            embedding_sequence = embedding_sequence * mask.unsqueeze(-1).float()
            summed = embedding_sequence.sum(dim=1)  # shape: (batch_size, embedding_dim)
            return summed
