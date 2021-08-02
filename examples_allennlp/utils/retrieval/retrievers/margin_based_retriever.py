import torch
from .retriever import Retriever


@Retriever.register("margin")
class MarginBasedRetriever(Retriever):
    """
    Penalize the scores of the queries and targets with many neighbors.

    Thie is proposed in ``Margin-based Parallel Corpus Mining with Multilingual Sentence Embeddings``
    (https://www.aclweb.org/anthology/P19-1309)
    """

    def __init__(self, k: int = 4, method: str = "ratio"):
        self.k = k

        assert method in {"ratio", "distance"}
        self.method = method

    def _post_process_scores(self, scores: torch.Tensor):
        query_topk_average_scores = torch.topk(scores, k=self.k, dim=1)[0].mean(dim=1)
        target_topk_average_scores = torch.topk(scores, k=self.k, dim=0)[0].mean(dim=0)
        margin = (query_topk_average_scores.unsqueeze(1) + target_topk_average_scores) / 2
        if self.method == "ratio":
            margin_based_scores = scores / margin
        elif self.method == "distance":
            margin_based_scores = scores - margin
        else:
            raise Exception(f"Unexpected method: {self.method}")
        return margin_based_scores
