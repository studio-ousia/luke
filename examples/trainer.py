import contextlib
import logging

import torch
from tqdm import tqdm
from transformers import WarmupConstantSchedule, WarmupLinearSchedule

from luke.optimization import LukeDenseSparseAdam

logger = logging.getLogger(__name__)


class Trainer(object):
    def __init__(
            self, model, dataloader, device, num_train_steps, learning_rate, weight_decay=0.01, max_grad_norm=1.0,
            adam_b1=0.9, adam_b2=0.999, adam_eps=1e-6, lr_schedule='warmup_linear', warmup_proportion=0.1,
            gradient_accumulation_steps=1, grad_avg_on_cpu=False, local_rank=-1, fp16=False, fp16_opt_level='O2',
            fp16_min_loss_scale=1, fp16_max_loss_scale=None):
        self.model = model
        self._dataloader = dataloader
        self._device = device
        self._num_train_steps = num_train_steps
        self._max_grad_norm = max_grad_norm
        self._gradient_accumulation_steps = gradient_accumulation_steps
        self._local_rank = local_rank
        self._fp16 = fp16
        self._fp16_opt_level = fp16_opt_level
        self._fp16_min_loss_scale = fp16_min_loss_scale
        self._fp16_max_loss_scale = fp16_max_loss_scale

        self.optimizer = self._create_optimizer(model, device, learning_rate, weight_decay, adam_b1, adam_b2, adam_eps,
                                                grad_avg_on_cpu)
        self.scheduler = self._create_scheduler(self.optimizer, lr_schedule, num_train_steps, warmup_proportion)

    def train(self):
        model = self.model
        optimizer = self.optimizer

        if self._fp16:
            from apex import amp
            model, optimizer = amp.initialize(model, optimizer, opt_level=self._fp16_opt_level,
                                              min_loss_scale=self._fp16_min_loss_scale,
                                              max_loss_scale=self._fp16_max_loss_scale)

        if self._local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[self._local_rank],
                                                              output_device=self._local_rank,
                                                              find_unused_parameters=True)

        epoch = 0
        global_step = 0
        tr_loss = 0.0

        num_workers = torch.cuda.device_count()

        def maybe_no_sync(step):
            if hasattr(model, 'no_sync') and num_workers > 1 and\
                (step + 1) % self._gradient_accumulation_steps != 0:
                return model.no_sync()
            else:
                return contextlib.ExitStack()

        model.train()

        with tqdm(total=self._num_train_steps, disable=self._local_rank not in [-1, 0]) as pbar:
            while True:
                for step, batch in enumerate(self._dataloader):
                    inputs = {k: v.to(self._device) for k, v in self._create_model_arguments(batch).items()}
                    outputs = model(**inputs)
                    loss = outputs[0]
                    if self._gradient_accumulation_steps > 1:
                        loss = loss / self._gradient_accumulation_steps

                    with maybe_no_sync(step):
                        if self._fp16:
                            with amp.scale_loss(loss, optimizer) as scaled_loss:
                                scaled_loss.backward()
                        else:
                            loss.backward()

                    tr_loss += loss.item()
                    if (step + 1) % self._gradient_accumulation_steps == 0:
                        if self._max_grad_norm != 0.0:
                            if self._fp16:
                                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), self._max_grad_norm)
                            else:
                                torch.nn.utils.clip_grad_norm_(model.parameters(), self._max_grad_norm)

                        self.optimizer.step()
                        self.scheduler.step()
                        model.zero_grad()
                        pbar.set_description('epoch: %d loss: %.7f lr: %.7f' %
                                             (epoch, loss.item(), max(self.scheduler.get_lr())))
                        pbar.update()
                        global_step += 1
                        if global_step == self._num_train_steps:
                            break
                if global_step == self._num_train_steps:
                    break
                epoch += 1

        return model, global_step, tr_loss / global_step

    def _create_optimizer(self, model, device, learning_rate, weight_decay, b1, b2, eps, grad_avg_on_cpu):
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        grad_avg_device = torch.device(device)
        if grad_avg_on_cpu:
            grad_avg_device = torch.device('cpu')

        return LukeDenseSparseAdam(optimizer_parameters, lr=learning_rate, betas=(b1, b2), eps=eps,
                                   grad_avg_device=grad_avg_device)

    def _create_scheduler(self, optimizer, schedule, num_train_steps, warmup_proportion):
        warmup_steps = int(num_train_steps * warmup_proportion)
        if schedule == 'warmup_linear':
            return WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=num_train_steps)
        if schedule == 'warmup_constant':
            return WarmupConstantSchedule(optimizer, warmup_steps=warmup_steps)

        raise RuntimeError('Unsupported scheduler: ' + schedule)

    def _create_model_arguments(self, batch):
        return batch
