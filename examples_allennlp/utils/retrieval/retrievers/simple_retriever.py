import torch
from .retriever import Retriever


@Retriever.register("simple")
class SimpleRetriever(Retriever):
    """
    Simply retrieve the target with the highest score.
    """
    def _post_process_scores(self, scores: torch.Tensor):
        return scores
