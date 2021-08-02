import tqdm

import torch
from allennlp.common import Registrable


def sharding(iterable, sharding_size: int = 1):
    l = len(iterable)
    for ndx in range(0, l, sharding_size):
        yield iterable[ndx : min(ndx + sharding_size, l)]


class ScoringFunction(Registrable):
    def __init__(self, sharding_size: int = 32):
        self.sharding_size = sharding_size

    def __call__(self, query_embeddings: torch.Tensor, target_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute similarity scores spliting target_embedding into shards.
        """
        score_shards = []
        for query_shard in tqdm.tqdm(sharding(query_embeddings, self.sharding_size)):
            score_shard = self._compute_score(query_shard, target_embeddings).detach().cpu()
            score_shards.append(score_shard)

        scores = torch.cat(score_shards, dim=0)

        return scores

    def _compute_score(self, query_embeddings: torch.Tensor, target_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute similarity scores between query and target embeddings.

        Parameters
        ----------
        query_embeddings: (num_query_embeddings, embedding_size)
        target_embeddings: (num_target_embeddings, embedding_size)

        Returns
        -------
        scores: (num_query_embeddings, num_target_embeddings)
        """
        raise NotImplementedError
