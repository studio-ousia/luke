from typing import Iterable

import torch


class Metric:
    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def get_metric(self, reset: bool = False) -> float:
        raise NotImplementedError

    @staticmethod
    def detach_tensors(*tensors: torch.Tensor) -> Iterable[torch.Tensor]:
        return (x.detach() if isinstance(x, torch.Tensor) else x for x in tensors)


class Average(Metric):
    def __init__(self):
        self._count = 0
        self._sum = 0

    def __call__(self, value: torch.Tensor):
        (value,) = self.detach_tensors(value)
        self._count += 1
        self._sum += value

    def get_metric(self, reset: bool = False):
        if self._count == 0:
            value = 0
        else:
            value = self._sum / self._count

        if reset:
            self._count = 0
            self._sum = 0
        return value


class Accuracy(Metric):
    def __init__(self, ignored_label: int = -1):
        self._total = 0
        self._correct = 0
        self._ignored_label = ignored_label

    def __call__(self, prediction: torch.Tensor, target: torch.Tensor):
        prediction, target = self.detach_tensors(prediction, target)
        assert len(prediction) == len(target)
        self._total += target.ne(self._ignored_label).sum()
        self._correct += (prediction == target).sum()

    def get_metric(self, reset: bool = False):
        if self._total == 0:
            value = 0
        else:
            value = self._correct / self._total

        if reset:
            self._total = 0
            self._correct = 0
        return value
