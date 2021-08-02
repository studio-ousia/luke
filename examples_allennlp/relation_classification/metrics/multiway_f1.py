from typing import List, Dict
import numpy as np
import torch
from allennlp.training.metrics import Metric
from collections import Counter

DEFAULT_IGNORED_LABEL = ["no_relation"]


def normalize_kbp37_label(label: str) -> str:
    label = label.replace("(e1,e2)", "")
    label = label.replace("(e2,e1)", "")
    return label


class MultiwayF1(Metric):
    """
    Calculate the F1 score by taking directionality into account.
    ```SemEval-2010 Task 8: Multi-Way Classification of Semantic Relations between Pairs of Nominals```
    (https://www.aclweb.org/anthology/S10-1006/)
    """

    def __init__(self, ignored_labels: List[str] = None, label_normalize_scheme: str = "kbp37"):
        self._num_gold_by_label = Counter()
        self._num_prediction_by_label = Counter()
        self._num_correct_by_label = Counter()

        self.ignored_labels = set(ignored_labels or DEFAULT_IGNORED_LABEL)

        self.label_normalize_scheme = label_normalize_scheme

    def __call__(
        self,
        prediction: torch.LongTensor,
        label: torch.LongTensor,
        prediction_labels: List[str],
        gold_labels: List[str],
    ):

        prediction, label = self.detach_tensors(prediction, label)
        correct_tensor = prediction == label

        if self.label_normalize_scheme == "kbp37":
            normalized_prediction_labels = [normalize_kbp37_label(l) for l in prediction_labels]
            normalized_gold_labels = [normalize_kbp37_label(l) for l in gold_labels]
        else:
            normalized_prediction_labels = prediction_labels
            normalized_gold_labels = gold_labels

        for correct, pred_label, gold_label in zip(
            correct_tensor, normalized_prediction_labels, normalized_gold_labels
        ):
            if correct:
                self._num_correct_by_label[pred_label] += 1
            self._num_gold_by_label[gold_label] += 1
            self._num_prediction_by_label[pred_label] += 1

    def reset(self):
        self._num_gold_by_label = Counter()
        self._num_correct_by_label = Counter()
        self._num_prediction_by_label = Counter()

    def get_f1_by_label(self, label: str) -> float:
        if self._num_correct_by_label[label] == 0:
            return 0

        precision = self._num_correct_by_label[label] / self._num_prediction_by_label[label]
        recall = self._num_correct_by_label[label] / self._num_gold_by_label[label]
        f1 = 2 * precision * recall / (precision + recall)
        return f1

    def get_micro_fscore(self) -> float:
        num_correct_predictions = self._get_sum_wihout_ignored_labels(self._num_correct_by_label)
        if num_correct_predictions == 0:
            return 0
        precision = num_correct_predictions / self._get_sum_wihout_ignored_labels(self._num_prediction_by_label)
        recall = num_correct_predictions / self._get_sum_wihout_ignored_labels(self._num_gold_by_label)
        f1 = 2 * precision * recall / (precision + recall)
        return f1

    def _get_sum_wihout_ignored_labels(self, counter: Counter) -> int:
        return sum([value for key, value in counter.items() if key not in self.ignored_labels])

    def get_metric(self, reset: bool) -> Dict[str, float]:
        all_labels = (
            set(self._num_prediction_by_label.keys())
            | set(self._num_gold_by_label.keys())
            | set(self._num_correct_by_label.keys())
        )
        macro_fscore_score = np.mean(
            [self.get_f1_by_label(label) for label in all_labels if label not in self.ignored_labels]
        )
        micro_fscore_score = self.get_micro_fscore()
        if reset:
            self.reset()
        return {"macro_fscore": macro_fscore_score, "micro_fscore": micro_fscore_score}
