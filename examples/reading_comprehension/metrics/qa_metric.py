from typing import Dict, List

import torch
from allennlp.training.metrics import Metric


class QAMetric(Metric):
    def __call__(self, output_dict: Dict[str, torch.Tensor], metadata_list: List[Dict]):
        raise NotImplementedError
