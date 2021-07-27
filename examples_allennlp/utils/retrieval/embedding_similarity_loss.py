import torch
from allennlp.common import Registrable


class EmbeddingSimilarityLoss(Registrable):
    def forward(self, similarity_matrix: torch.Tensor):
        raise NotImplementedError


@EmbeddingSimilarityLoss.register("in-batch_softmax")
class InBatchSoftmax(EmbeddingSimilarityLoss):
    def __init__(self):
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, similarity_matrix: torch.Tensor):
        batch_size = similarity_matrix.size(0)

        label = torch.arange(batch_size).to(similarity_matrix.device)
        return self.criterion(similarity_matrix, label)
