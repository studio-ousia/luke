from typing import Dict
import torch
import torch.nn as nn
from allennlp.common.registrable import Registrable


class NERFeatureExtractor(Registrable, nn.Module):
    """
    Feature extractor for named entity recognition.
    """

    def __init__(self):
        super().__init__()

    def get_output_dim(self) -> int:
        raise NotImplementedError

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        entity_start_positions: torch.LongTensor,
        entity_end_positions: torch.LongTensor,
        entity_ids: torch.LongTensor = None,
        entity_position_ids: torch.LongTensor = None,
    ) -> torch.Tensor:
        raise NotImplementedError
