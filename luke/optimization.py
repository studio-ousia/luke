# -*- coding: utf-8 -*-

from copy import deepcopy
from itertools import chain
from collections import defaultdict
from torch._six import container_abcs

import torch
from torch.optim.optimizer import Optimizer
from torch.nn.utils import clip_grad_norm_


def warmup_linear(x, warmup=0.002, decay=False):
    if x < warmup:
        return x/warmup
    if decay:
        return 1.0 - x
    else:
        return 1.0


class BertAdam(Optimizer):
    """Implements BERT version of Adam algorithm with weight decay fix.
    Params:
        lr: learning rate
        warmup: portion of t_total for the warmup, -1  means no warmup. Default: -1
        t_total: total number of training steps for the learning
            rate schedule, -1  means constant learning rate. Default: -1
        b1: Adams b1. Default: 0.9
        b2: Adams b2. Default: 0.999
        e: Adams epsilon. Default: 1e-6
        weight_decay_rate: Weight decay. Default: 0.01
        max_grad_norm: Maximum norm for the gradients (-1 means no clipping). Default: 1.0
    """
    def __init__(self, params, lr, warmup=-1, lr_decay=False, t_total=-1, b1=0.9, b2=0.999, e=1e-6,
                 weight_decay_rate=0.01, max_grad_norm=1.0, device=None):
        if not lr >= 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= warmup < 1.0 and not warmup == -1:
            raise ValueError("Invalid warmup: {} - should be in [0.0, 1.0[ or -1".format(warmup))
        if not 0.0 <= b1 < 1.0:
            raise ValueError("Invalid b1 parameter: {} - should be in [0.0, 1.0[".format(b1))
        if not 0.0 <= b2 < 1.0:
            raise ValueError("Invalid b2 parameter: {} - should be in [0.0, 1.0[".format(b2))
        if not e >= 0.0:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(e))
        defaults = dict(lr=lr, lr_decay=lr_decay, warmup=warmup, t_total=t_total, b1=b1, b2=b2, e=e,
                        weight_decay_rate=weight_decay_rate, max_grad_norm=max_grad_norm)

        super(BertAdam, self).__init__(params, defaults)

        if device is None:
            self.device = self.param_groups[0]['params'][0].device
        else:
            self.device = device

    def get_lr(self):
        lr = []
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if len(state) == 0:
                    return [0]
                if group['t_total'] != -1:
                    lr_scheduled = group['lr'] * warmup_linear(state['step'] / group['t_total'],
                                                               group['warmup'], group.get('lr_decay', False))
                else:
                    lr_scheduled = group['lr']
                lr.append(lr_scheduled)
        return lr

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['next_m'] = torch.zeros_like(p.data, device=self.device)
                    # Exponential moving average of squared gradient values
                    state['next_v'] = torch.zeros_like(p.data, device=self.device)

                next_m, next_v = state['next_m'], state['next_v']
                beta1, beta2 = group['b1'], group['b2']

                # Add grad clipping
                if group['max_grad_norm'] > 0:
                    clip_grad_norm_(p, group['max_grad_norm'])

                if group['t_total'] != -1:
                    lr_scheduled = group['lr'] * warmup_linear(state['step'] / group['t_total'],
                                                            group['warmup'], group.get('lr_decay', False))
                else:
                    lr_scheduled = group['lr']

                if grad.is_sparse:
                    grad = grad.coalesce()  # the update is non-linear so indices must be unique

                    p_data = p.data.sparse_mask(grad).to(self.device)
                    grad = grad.to(self.device)

                    grad_indices = grad._indices()
                    grad_values = grad._values()
                    size = grad.size()

                    def make_sparse(values):
                        constructor = grad.new
                        if grad_indices.dim() == 0 or values.dim() == 0:
                            return constructor().resize_as_(grad)
                        return constructor(grad_indices, values, size)

                    next_m, next_v = state['next_m'], state['next_v']
                    beta1, beta2 = group['b1'], group['b2']

                    old_next_m_values = next_m.sparse_mask(grad)._values()
                    next_m_update_values = grad_values.sub(old_next_m_values).mul_(1 - beta1)
                    next_m.add_(make_sparse(next_m_update_values))
                    old_next_v_values = next_v.sparse_mask(grad)._values()
                    next_v_update_values = grad_values.pow(2).sub_(old_next_v_values).mul_(1 - beta2)
                    next_v.add_(make_sparse(next_v_update_values))

                    # Dense addition again is intended, avoiding another sparse_mask
                    numer = next_m_update_values.add_(old_next_m_values)
                    next_v_update_values.add_(old_next_v_values)
                    denom = next_v_update_values.sqrt_().add_(group['e'])
                    del next_m_update_values, next_v_update_values

                    update_values = numer / denom

                    if group['weight_decay_rate'] > 0.0:
                        update_values += group['weight_decay_rate'] * p_data._values()

                    update_with_lr = -lr_scheduled * update_values
                    update_with_lr = make_sparse(update_with_lr).to(p.device)

                else:
                    grad = grad.to(self.device)

                    # Decay the first and second moment running average coefficient
                    # In-place operations to update the averages at the same time
                    next_m.mul_(beta1).add_(1 - beta1, grad)
                    next_v.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                    update = next_m / (next_v.sqrt() + group['e'])

                    update = update.to(p.device)

                    if group['weight_decay_rate'] > 0.0:
                        update += group['weight_decay_rate'] * p.data

                    update_with_lr = -lr_scheduled * update

                p.data.add_(update_with_lr)
                state['step'] += 1

        return loss

    def load_state_dict(self, state_dict):
        # originally obtained from: https://github.com/pytorch/pytorch/blob/7956e9718b72e6399f89a2b9cdaf489df22cacc4/torch/optim/optimizer.py#L95

        # deepcopy, to be consistent with module API
        state_dict = deepcopy(state_dict)
        # Validate the state_dict
        groups = self.param_groups
        saved_groups = state_dict['param_groups']

        if len(groups) != len(saved_groups):
            raise ValueError("loaded state dict has a different number of "
                             "parameter groups")
        param_lens = (len(g['params']) for g in groups)
        saved_lens = (len(g['params']) for g in saved_groups)
        if any(p_len != s_len for p_len, s_len in zip(param_lens, saved_lens)):
            raise ValueError("loaded state dict contains a parameter group "
                             "that doesn't match the size of optimizer's group")

        # Update the state
        id_map = {old_id: p for old_id, p in
                  zip(chain(*(g['params'] for g in saved_groups)),
                      chain(*(g['params'] for g in groups)))}

        def cast(param, value):
            r"""Make a deep copy of value, casting all tensors to device of param."""
            if isinstance(value, torch.Tensor):
                # Floating-point types are a bit special here. They are the only ones
                # that are assumed to always match the type of params.
                if param.is_floating_point():
                    value = value.to(param.dtype)
                value = value.to(self.device)
                # value = value.to(param.device)
                return value
            elif isinstance(value, dict):
                return {k: cast(param, v) for k, v in value.items()}
            elif isinstance(value, container_abcs.Iterable):
                return type(value)(cast(param, v) for v in value)
            else:
                return value

        # Copy state assigned to params (and cast tensors to appropriate types).
        # State that is not assigned to params is copied as is (needed for
        # backward compatibility).
        state = defaultdict(dict)
        for k, v in state_dict['state'].items():
            if k in id_map:
                param = id_map[k]
                state[param] = cast(param, v)
            else:
                state[k] = v

        # Update parameter groups, setting their 'params' value
        def update_group(group, new_group):
            new_group['params'] = group['params']
            return new_group
        param_groups = [
            update_group(g, ng) for g, ng in zip(groups, saved_groups)]
        self.__setstate__({'state': state, 'param_groups': param_groups})
