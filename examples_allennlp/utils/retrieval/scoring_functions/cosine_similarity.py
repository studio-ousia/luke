import torch

from .scoring_function import ScoringFunction


@ScoringFunction.register("cosine")
class CosineSimilarity(ScoringFunction):
    def _compute_score(self, query_embeddings: torch.Tensor, target_embeddings: torch.Tensor) -> torch.Tensor:
        query_normalized = query_embeddings / query_embeddings.norm(dim=1)[:, None]
        target_normalized = target_embeddings / target_embeddings.norm(dim=1)[:, None]

        similarity = torch.mm(query_normalized, target_normalized.transpose(0, 1))
        return similarity
