from typing import Dict
import torch
import torch.nn as nn
from allennlp.common.registrable import Registrable


class ETFeatureExtractor(Registrable, nn.Module):
    """
    Feature extractor for entity typing.
    """

    def __init__(self):
        super().__init__()

    def get_output_dim(self) -> int:
        raise NotImplementedError

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        entity_span: torch.LongTensor,
        entity_ids: torch.LongTensor,
    ) -> torch.Tensor:
        raise NotImplementedError
