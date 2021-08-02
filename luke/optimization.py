# This code is based on the Transformers' AdamW
# https://github.com/huggingface/transformers/blob/6be7cdda66f3f4bd3ba4073274bf73be0843c5f9/src/transformers/optimization.py

from typing import Callable, Dict
import torch
from transformers.optimization import AdamW


class LukeAdamW(AdamW):
    def __init__(self, params, *args, grad_avg_device=None, **kwargs):
        super(LukeAdamW, self).__init__(params, *args, **kwargs)
        if grad_avg_device is None:
            self.grad_avg_device = self.param_groups[0]["params"][0].device
        else:
            self.grad_avg_device = grad_avg_device

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data, device=self.grad_avg_device)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data, device=self.grad_avg_device)

                exp_avg, exp_avg_sq = state["exp_avg"].to(p.device), state["exp_avg_sq"].to(p.device)
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                p.data.addcdiv_(exp_avg, denom, value=-group["lr"])

                state["exp_avg"] = exp_avg.to(self.grad_avg_device)
                state["exp_avg_sq"] = exp_avg_sq.to(self.grad_avg_device)

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                # Add weight decay at the end (fixed version)
                if group["weight_decay"] > 0.0:
                    p.data.add_(p.data, alpha=-group["lr"] * group["weight_decay"])

        return loss

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        super(LukeAdamW, self).load_state_dict(state_dict)

        for state in self.state.values():
            if "exp_avg" in state:
                state["exp_avg"] = state["exp_avg"].to(self.grad_avg_device)
                state["exp_avg_sq"] = state["exp_avg_sq"].to(self.grad_avg_device)
