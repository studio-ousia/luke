# -*- coding: utf-8 -*-

import logging
import os
import time
import joblib
import numpy as np
import torch
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
from pytorch_pretrained_bert.modeling import BertForPreTraining
from tensorboardX import SummaryWriter
from tqdm import tqdm

from batch_generator import BatchGenerator
from model import LukeConfig, LukeModel
from wiki_corpus import WikiCorpus

logger = logging.getLogger(__name__)


def train(corpus_data_file, entity_vocab, run_name, output_dir, log_dir, mmap, batch_size,
          learning_rate, warmup_steps, gradient_accumulation_steps, fp16,
          static_loss_scale, fp16_emb, max_seq_length, max_entity_length, short_seq_prob,
          masked_lm_prob, max_predictions_per_seq, num_train_steps, num_page_split,
          entity_emb_size, bert_model_name, global_step=0, page_indices_list=[],
          model_file=None, optimizer_file=None):
    train_args = locals()

    logger.info('run name: %s', run_name)

    device = torch.device('cuda')
    n_gpu = torch.cuda.device_count()

    logger.info('device: {} n_gpu: {}, 16-bits training: {}'.format(device, n_gpu, fp16))

    batch_size = int(batch_size / gradient_accumulation_steps)

    corpus = WikiCorpus(corpus_data_file, mmap_mode='r')

    bert_model = BertForPreTraining.from_pretrained(bert_model_name)
    logger.info('loaded %s model', bert_model_name)

    config = LukeConfig(entity_vocab_size=entity_vocab.size,
                        entity_emb_size=entity_emb_size,
                        max_entity_position_embeddings=bert_model.config.max_position_embeddings,
                        fp16_emb=fp16_emb,
                        **bert_model.config.to_dict())

    logger.info('configuration: %s', config)

    model = LukeModel(config)
    if model_file is None:
        model.load_bert_weights(bert_model.state_dict())
    else:
        model.load_state_dict(torch.load(model_file))
    del bert_model

    if fp16:
        model = model.half()
    model.to(device)

    if n_gpu > 1:
        # model = torch.nn.DataParallel(model, device_ids=range(1, n_gpu), output_device=0)
        model = torch.nn.DataParallel(model)

    model.train()

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for (n, p) in param_optimizer if not any(nd in n for nd in no_decay) and
                    p.requires_grad],
         'weight_decay': 0.01},
        {'params': [p for (n, p) in param_optimizer if any(nd in n for nd in no_decay) and
                    p.requires_grad],
         'weight_decay': 0.0}
    ]
    warmup_proportion = warmup_steps / num_train_steps
    if fp16:
        from apex.optimizers import FP16_Optimizer
        from apex.optimizers import FusedAdam

        optimizer = FusedAdam(optimizer_grouped_parameters, lr=learning_rate,
                              bias_correction=False, max_grad_norm=1.0)
        if static_loss_scale != 0:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=static_loss_scale)
        else:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)

    else:
        optimizer = BertAdam(optimizer_grouped_parameters, lr=learning_rate,
                             warmup=warmup_proportion, t_total=num_train_steps)

    if optimizer_file is not None:
        optimizer.load_state_dict(torch.load(optimizer_file))

    run_output_dir = os.path.join(output_dir, run_name)
    os.makedirs(run_output_dir, exist_ok=True)
    run_log_dir = os.path.join(log_dir, run_name)
    os.makedirs(run_log_dir, exist_ok=True)

    logger.info("***** Running training *****")
    batch_generator = BatchGenerator(
        corpus_data_file=corpus_data_file,
        entity_vocab=entity_vocab,
        batch_size=batch_size,
        max_seq_length=max_seq_length,
        max_entity_length=max_entity_length,
        short_seq_prob=short_seq_prob,
        masked_lm_prob=masked_lm_prob,
        max_predictions_per_seq=max_predictions_per_seq,
        mmap=mmap)

    summary_writer = SummaryWriter(run_log_dir)
    pbar = tqdm(total=num_train_steps)

    while True:
        if not page_indices_list:
            page_indices_list = np.array_split(np.random.permutation(corpus.page_size),
                                               num_page_split)
        page_indices = page_indices_list.pop()

        tr_loss = 0
        results = []
        prev_time = time.time()

        for (step, batch) in enumerate(batch_generator.generate_batches(
            page_indices=page_indices)):
            batch = {k: torch.from_numpy(v).to(device) for (k, v) in batch.items()}
            result = model(**batch)
            results.append(result)

            loss = result['loss']

            if n_gpu > 1:
                loss = loss.mean()

            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps
            if fp16:
                optimizer.backward(loss)
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % gradient_accumulation_steps == 0:
                lr_this_step = learning_rate * warmup_linear(global_step / num_train_steps,
                                                             warmup_proportion)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step

                optimizer.step()
                optimizer.zero_grad()

                summary = {}
                summary['learning_rate'] = lr_this_step
                summary['loss'] = tr_loss
                tr_loss = 0

                current_time = time.time()
                summary['batch_run_time'] = current_time - prev_time
                prev_time = current_time

                for name in ('masked_lm', 'nsp', 'entity'):
                    summary[name + '_loss'] = torch.cat(
                        [r[name + '_loss'].view(-1) for r in results]).mean().item()
                    correct = torch.cat(
                        [r[name + '_correct'].view(-1) for r in results]).sum().item()
                    total = torch.cat(
                        [r[name + '_total'].view(-1) for r in results]).sum().item()
                    if total > 0:
                        summary[name + '_acc'] = correct / total

                results = []

                for (name, value) in summary.items():
                    summary_writer.add_scalar(name, value, global_step)

                if global_step == num_train_steps:
                    break

                global_step += 1
                pbar.update(1)

        logger.info('saving the model to %s', run_output_dir)
        suffix = 'step%07d' % (global_step,)

        if n_gpu > 1:
            torch.save(model.module.state_dict(),
                        os.path.join(run_output_dir, 'model_%s.bin' % suffix))
        else:
            torch.save(model.state_dict(),
                        os.path.join(run_output_dir, 'model_%s.bin' % suffix))

        torch.save(optimizer.state_dict(), os.path.join(run_output_dir,
                                                        'optimizer_%s.bin' % suffix))

        data = train_args
        data['global_step'] = global_step
        data['page_indices_list'] = page_indices_list
        joblib.dump(data, os.path.join(run_output_dir, 'data_%s.pkl' % suffix))

        if global_step == num_train_steps:
            break

    summary_writer.close()
