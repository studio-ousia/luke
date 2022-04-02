from typing import Dict

import torch
import torch.nn as nn
from allennlp.common.registrable import Registrable


class RCFeatureExtractor(Registrable, nn.Module):
    """
    Feature extractor for relation classification.
    """

    def __init__(self):
        super().__init__()

    def get_output_dim(self) -> int:
        raise NotImplementedError

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        entity1_span: torch.LongTensor,
        entity2_span: torch.LongTensor,
        entity_ids: torch.LongTensor,
    ) -> torch.Tensor:
        raise NotImplementedError
