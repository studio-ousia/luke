from typing import Tuple
import torch
from allennlp.common import Registrable


class Retriever(Registrable):
    def _post_process_scores(self, scores: torch.Tensor):
        raise NotImplementedError

    def __call__(self, scores: torch.Tensor) -> Tuple[torch.LongTensor, torch.LongTensor]:
        """
        Parameters
        ----------
        scores: torch.Tensor (num_queries, num_targets)
        """
        scores = self._post_process_scores(scores)
        max_scores, indices = torch.max(scores, dim=1)
        return max_scores, indices
