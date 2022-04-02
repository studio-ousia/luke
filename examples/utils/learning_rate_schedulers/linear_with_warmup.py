import torch
from allennlp.training.learning_rate_schedulers import PolynomialDecay
from allennlp.training.learning_rate_schedulers.learning_rate_scheduler import LearningRateScheduler


@LearningRateScheduler.register("custom_linear_with_warmup")
class CustomLinearWithWarmup(PolynomialDecay):
    """
    Implements a learning rate scheduler that increases the learning rate to `lr` during the first
    `warmup_steps` steps, and then decreases it to zero over the rest of the training steps.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        num_epochs: int,
        num_steps_per_epoch: int,
        warmup_ratio: float,
        last_epoch: int = -1,
    ) -> None:
        super().__init__(
            optimizer,
            num_epochs,
            num_steps_per_epoch,
            power=1.0,
            warmup_steps=int(num_epochs * num_steps_per_epoch * warmup_ratio),
            end_learning_rate=0.0,
            last_epoch=last_epoch,
        )
