import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim.optimizer import required
from pytorch_pretrained_bert.optimization import BertAdam


class BertDenseSparseAdam(BertAdam):
    def __init__(self, params, lr=required, warmup=-1, t_total=-1, schedule='warmup_linear', b1=0.9, b2=0.999, e=1e-6,
                 weight_decay=0.01, max_grad_norm=1.0, device=None, **kwargs):
        super(BertDenseSparseAdam, self).__init__(params, lr, warmup=warmup, t_total=t_total, schedule=schedule, b1=b1,
                                                  b2=b2, e=e, weight_decay=weight_decay, max_grad_norm=max_grad_norm,
                                                  **kwargs)
        if device is None:
            self.device = self.param_groups[0]['params'][0].device
        else:
            self.device = device

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

                lr_scheduled = group['lr']
                lr_scheduled *= group['schedule'].get_lr(state['step'])

                if grad.is_sparse:
                    grad = orig_grad = grad.coalesce()  # the update is non-linear so indices must be unique
                    grad = grad.to(self.device)
                    grad_values = grad._values()

                    next_m, next_v = state['next_m'], state['next_v']
                    beta1, beta2 = group['b1'], group['b2']

                    old_next_m_values = next_m.sparse_mask(grad)._values()
                    next_m_update_values = grad_values.sub(old_next_m_values).mul_(1 - beta1)
                    next_m.add_(self._make_sparse(next_m_update_values, grad))
                    old_next_v_values = next_v.sparse_mask(grad)._values()
                    next_v_update_values = grad_values.pow(2).sub_(old_next_v_values).mul_(1 - beta2)
                    next_v.add_(self._make_sparse(next_v_update_values, grad))

                    # Dense addition again is intended, avoiding another sparse_mask
                    numer = next_m_update_values.add_(old_next_m_values)
                    next_v_update_values.add_(old_next_v_values)
                    denom = next_v_update_values.sqrt_().add_(group['e'])
                    del next_m_update_values, next_v_update_values

                    update_values = numer / denom
                    update_values = update_values.to(p.device)

                    if group['weight_decay'] > 0.0:
                        update_values += group['weight_decay'] * p.data.sparse_mask(orig_grad)._values()

                    update_with_lr = -lr_scheduled * update_values
                    update_with_lr = self._make_sparse(update_with_lr, orig_grad)

                else:
                    grad = grad.to(self.device)

                    # Decay the first and second moment running average coefficient
                    # In-place operations to update the averages at the same time
                    next_m.mul_(beta1).add_(1 - beta1, grad)
                    next_v.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                    update = next_m / (next_v.sqrt() + group['e'])

                    update = update.to(p.device)

                    if group['weight_decay'] > 0.0:
                        update += group['weight_decay'] * p.data

                    update_with_lr = -lr_scheduled * update

                p.data.add_(update_with_lr)
                state['step'] += 1

        return loss

    @staticmethod
    def _make_sparse(values, sparse_tensor):
        indices = sparse_tensor._indices()
        if indices.dim() == 0 or values.dim() == 0:
            return sparse_tensor.new().resize_as_(sparse_tensor)
        return sparse_tensor.new(indices, values, sparse_tensor.size())

    def load_state_dict(self, state_dict):
        super(BertDenseSparseAdam, self).load_state_dict(state_dict)

        for state in self.state.values():
            if 'next_m' in state:
                state['next_m'] = state['next_m'].to(self.device)
                state['next_v'] = state['next_v'].to(self.device)
