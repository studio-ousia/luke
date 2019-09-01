import torch
from torch.optim.lr_scheduler import LambdaLR
from pytorch_transformers.optimization import AdamW


class LukeDenseSparseAdam(AdamW):
    def __init__(self, params, *args, grad_avg_device=None, **kwargs):
        super(LukeDenseSparseAdam, self).__init__(params, *args, **kwargs)
        if grad_avg_device is None:
            self.grad_avg_device = self.param_groups[0]['params'][0].device
        else:
            self.grad_avg_device = grad_avg_device

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
                    state['exp_avg'] = torch.zeros_like(p.data, device=self.grad_avg_device)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data, device=self.grad_avg_device)

                state['step'] += 1

                exp_avg, exp_avg_sq = state['exp_avg'].to(p.device), state['exp_avg_sq'].to(p.device)
                beta1, beta2 = group['betas']

                if grad.is_sparse:
                    grad = orig_grad = grad.coalesce()  # the update is non-linear so indices must be unique
                    grad_values = grad._values()

                    old_next_m_values = exp_avg.sparse_mask(grad)._values()
                    next_m_update_values = grad_values.sub(old_next_m_values).mul_(1 - beta1)
                    exp_avg.add_(self._make_sparse(next_m_update_values, grad))
                    old_next_v_values = exp_avg_sq.sparse_mask(grad)._values()
                    next_v_update_values = grad_values.pow(2).sub_(old_next_v_values).mul_(1 - beta2)
                    exp_avg_sq.add_(self._make_sparse(next_v_update_values, grad))

                    # Dense addition again is intended, avoiding another sparse_mask
                    numer = next_m_update_values.add_(old_next_m_values)
                    next_v_update_values.add_(old_next_v_values)
                    denom = next_v_update_values.sqrt_().add_(group['eps'])
                    del next_m_update_values, next_v_update_values

                    update_values = numer / denom

                    if group['weight_decay'] > 0.0:
                        update_values += group['weight_decay'] * p.data.sparse_mask(orig_grad)._values()

                    update_with_lr = -group['lr'] * update_values
                    update_with_lr = self._make_sparse(update_with_lr, orig_grad)

                else:
                    exp_avg.mul_(beta1).add_(1.0 - beta1, grad)
                    exp_avg_sq.mul_(beta2).addcmul_(1.0 - beta2, grad, grad)
                    update = exp_avg / (exp_avg_sq.sqrt() + group['eps'])

                    if group['weight_decay'] > 0.0:
                        update += group['weight_decay'] * p.data

                    update_with_lr = -group['lr'] * update

                state['exp_avg'] = exp_avg.to(self.grad_avg_device)
                state['exp_avg_sq'] = exp_avg_sq.to(self.grad_avg_device)

                p.data.add_(update_with_lr)

        return loss

    @staticmethod
    def _make_sparse(values, sparse_tensor):
        indices = sparse_tensor._indices()
        if indices.dim() == 0 or values.dim() == 0:
            return sparse_tensor.new().resize_as_(sparse_tensor)
        return sparse_tensor.new(indices, values, sparse_tensor.size())

    def load_state_dict(self, state_dict):
        super(LukeDenseSparseAdam, self).load_state_dict(state_dict)

        for state in self.state.values():
            if 'exp_avg' in state:
                state['exp_avg'] = state['exp_avg'].to(self.grad_avg_device)
                state['exp_avg_sq'] = state['exp_avg_sq'].to(self.grad_avg_device)


class WarmupInverseSquareRootSchedule(LambdaLR):
    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        super(WarmupInverseSquareRootSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return step / self.warmup_steps
        return (step - self.warmup_steps + 1) ** -0.5
