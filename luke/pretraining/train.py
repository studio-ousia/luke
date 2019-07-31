import gc
import inspect
import json
import logging
import math
import os
import time
import joblib
import numpy as np
import torch
from pytorch_transformers.modeling_bert import BertForPreTraining
from pytorch_transformers.optimization import ConstantLRSchedule, WarmupLinearSchedule, WarmupConstantSchedule
from tensorboardX import SummaryWriter
from tqdm import tqdm

from luke.model import LukeConfig, LukeE2EConfig
from luke.pretraining.batch_generator import LukePretrainingBatchGenerator, LukeE2EPretrainingBatchGenerator
from luke.pretraining.model import LukePretrainingModel, LukeE2EPretrainingModel
from luke.optimization import LukeDenseSparseAdam
from luke.utils.entity_vocab import EntityVocab
from luke.utils.wiki_corpus import WikiCorpus

logger = logging.getLogger(__name__)


def run_training(corpus_file, entity_vocab_file, output_dir, bert_model_name, single_sentence, max_seq_length,
                 max_entity_length, max_mention_length, short_seq_prob, masked_lm_prob, masked_entity_prob,
                 batch_size, gradient_accumulation_steps, learning_rate, lr_schedule, warmup_steps, fix_bert_weights,
                 grad_avg_on_cpu, num_train_steps, num_page_chunks, fp16=True, fp16_opt_level='O1', local_rank=-1,
                 whole_word_masking=False, log_dir=None, model_file=None, optimizer_file=None, epoch=0, global_step=0,
                 page_chunks=[], **kwargs):
    train_args = {}
    for arg in inspect.getfullargspec(run_training).args:
        train_args[arg] = locals()[arg]

    entity_vocab = EntityVocab(entity_vocab_file)
    bert_model = BertForPreTraining.from_pretrained(bert_model_name)

    config = LukeConfig(entity_vocab_size=entity_vocab.size, **bert_model.config.to_dict())
    logger.info('Model configuration: %s', config)

    model = LukePretrainingModel(config)
    if model_file is None:
        model.load_bert_weights(bert_model.state_dict())
    else:
        model.load_state_dict(torch.load(model_file, map_location='cpu'))

    if fix_bert_weights:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.entity_embeddings.parameters():
            param.requires_grad = True
        for param in model.entity_predictions.parameters():
            param.requires_grad = True

    batch_generator = LukePretrainingBatchGenerator(
        corpus_file=corpus_file,
        entity_vocab=entity_vocab,
        batch_size=int(batch_size / gradient_accumulation_steps),
        max_seq_length=max_seq_length,
        max_entity_length=max_entity_length,
        max_mention_length=max_mention_length,
        short_seq_prob=short_seq_prob,
        masked_lm_prob=masked_lm_prob,
        masked_entity_prob=masked_entity_prob,
        whole_word_masking=whole_word_masking,
        single_sentence=single_sentence)

    _run_training(model, batch_generator, train_args, corpus_file, output_dir, gradient_accumulation_steps,
                  learning_rate, lr_schedule, warmup_steps, grad_avg_on_cpu, fp16, fp16_opt_level, local_rank,
                  num_train_steps, num_page_chunks, log_dir, optimizer_file, epoch, global_step, page_chunks)


def run_e2e_training(corpus_file, entity_vocab_file, output_dir, bert_model_name, single_sentence, max_seq_length,
                     max_entity_length, max_mention_length, max_candidate_length, short_seq_prob, masked_lm_prob,
                     masked_entity_prob, min_candidate_prior_prob, num_el_hidden_layers, entity_selector_softmax_temp,
                     batch_size, gradient_accumulation_steps, learning_rate, lr_schedule, warmup_steps,
                     fix_bert_weights, grad_avg_on_cpu, num_train_steps, num_page_chunks, fp16=False,
                     fp16_opt_level='O1', local_rank=-1, whole_word_masking=False, fix_word_emb=False, log_dir=None,
                     model_file=None, optimizer_file=None, epoch=0, global_step=0, page_chunks=[], **kwargs):
    train_args = {}
    for arg in inspect.getfullargspec(run_e2e_training).args:
        train_args[arg] = locals()[arg]

    entity_vocab = EntityVocab(entity_vocab_file)
    bert_model = BertForPreTraining.from_pretrained(bert_model_name)

    config = LukeE2EConfig(entity_vocab_size=entity_vocab.size,
                           num_el_hidden_layers=num_el_hidden_layers,
                           entity_selector_softmax_temp=entity_selector_softmax_temp,
                           **bert_model.config.to_dict())
    logger.info('Model configuration: %s', config)

    model = LukeE2EPretrainingModel(config)
    if model_file is None:
        bert_state_dict = bert_model.state_dict()
        for key in tuple(bert_state_dict.keys()):
            if key.startswith('bert.encoder.layer.') and int(key.split('.')[3]) < num_el_hidden_layers:
                bert_state_dict['el_encoder.' + key[13:]] = bert_state_dict[key]
        model.load_bert_weights(bert_state_dict)
    else:
        model_state_dict = torch.load(model_file, map_location='cpu')
        for key in tuple(model_state_dict.keys()):
            if key.startswith('bert.encoder.layer.') and int(key.split('.')[3]) < num_el_hidden_layers:
                model_state_dict['el_encoder.' + key[13:]] = model_state_dict[key]
        model.load_state_dict(model_state_dict, strict=False)

    if fix_bert_weights:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.entity_embeddings.parameters():
            param.requires_grad = True
        for param in model.entity_predictions.parameters():
            param.requires_grad = True
        for param in model.entity_selector.parameters():
            param.requires_grad = True
        for param in model.el_encoder.parameters():
            param.requires_grad = True

    elif fix_word_emb:
        model.embeddings.word_embeddings.weight.requires_grad = False

    batch_generator = LukeE2EPretrainingBatchGenerator(
        corpus_file=corpus_file,
        entity_vocab=entity_vocab,
        batch_size=int(batch_size / gradient_accumulation_steps),
        max_seq_length=max_seq_length,
        max_entity_length=max_entity_length,
        max_mention_length=max_mention_length,
        max_candidate_length=max_candidate_length,
        short_seq_prob=short_seq_prob,
        masked_lm_prob=masked_lm_prob,
        masked_entity_prob=masked_entity_prob,
        whole_word_masking=whole_word_masking,
        single_sentence=single_sentence,
        min_candidate_prior_prob=min_candidate_prior_prob)

    _run_training(model, batch_generator, train_args, corpus_file, output_dir, gradient_accumulation_steps,
                  learning_rate, lr_schedule, warmup_steps, grad_avg_on_cpu, fp16, fp16_opt_level, local_rank,
                  num_train_steps, num_page_chunks, log_dir, optimizer_file, epoch, global_step, page_chunks)


def _run_training(model, batch_generator, train_args, corpus_file, output_dir, gradient_accumulation_steps,
                  learning_rate, lr_schedule, warmup_steps, grad_avg_on_cpu, fp16, fp16_opt_level, local_rank,
                  num_train_steps, num_page_chunks, log_dir, optimizer_file, epoch, global_step, page_chunks):
    if local_rank == -1:
        device = torch.device('cuda')
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
        n_gpu = 1
        torch.distributed.init_process_group(backend='nccl')
        world_size = torch.distributed.get_world_size()

    model.to(device)

    if local_rank != -1:
        num_train_steps = num_train_steps // world_size

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    grad_avg_device = device
    if grad_avg_on_cpu:
        grad_avg_device = torch.device('cpu')

    optimizer = LukeDenseSparseAdam(optimizer_parameters, lr=learning_rate, grad_avg_device=grad_avg_device)
    if lr_schedule == 'warmup_constant':
        scheduler = WarmupConstantSchedule(optimizer, warmup_steps=warmup_steps)
    elif lr_schedule == 'warmup_linear':
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=num_train_steps)
    else:
        scheduler = ConstantLRSchedule(optimizer)

    if fp16:
        from apex import amp
        model, optimizer = amp.initialize(model, optimizer, opt_level=fp16_opt_level)

    if optimizer_file is not None:
        optimizer.load_state_dict(torch.load(optimizer_file, map_location='cpu'))

    if local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank,
                                                          find_unused_parameters=True)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    model.train()

    def save_model(model, suffix, epoch, global_step, page_chunks):
        if n_gpu > 1 or local_rank != -1:
            model = model.module

        torch.save(model.state_dict(), os.path.join(output_dir, f'model_{suffix}.bin'))
        torch.save(optimizer.state_dict(), os.path.join(output_dir, f'optimizer_{suffix}.bin'))

        config_dict = model.config.to_dict()

        json_data = dict(model_config=config_dict, epoch=epoch, global_step=global_step)
        with open(os.path.join(output_dir, f'model_{suffix}.json'), 'w') as f:
            json.dump(json_data, f, indent=2, sort_keys=True)

        joblib.dump(dict(args=train_args, epoch=epoch, global_step=global_step, page_chunks=page_chunks,
                         model_config=config_dict), os.path.join(output_dir, f'model_{suffix}.pkl'))

    if local_rank in (0, -1):
        summary_writer = SummaryWriter(log_dir)
        pbar = tqdm(total=num_train_steps, initial=global_step)

    total_page_size = WikiCorpus(corpus_file, mmap_mode='r').page_size

    while True:
        if not page_chunks:
            logger.info('Creating new page chunks (global_step=%d)', global_step)
            np.random.seed(epoch)
            page_chunks = np.array_split(np.random.permutation(total_page_size), num_page_chunks)

        page_indices = page_chunks.pop()
        if local_rank != -1:
            num_indices_per_gpu = math.ceil(page_indices.size / world_size)
            pad_size = num_indices_per_gpu - page_indices.size % num_indices_per_gpu
            page_indices = np.concatenate([page_indices, np.random.randint(total_page_size, size=pad_size)])
            assert page_indices.size == num_indices_per_gpu * world_size
            page_indices = page_indices.reshape(world_size, num_indices_per_gpu)[torch.distributed.get_rank()]

        step = 0
        tr_loss = 0
        error_count = 0
        results = []
        prev_time = time.time()

        for batch in batch_generator.generate_batches(page_indices=page_indices):
            try:
                batch = {k: torch.from_numpy(v).to(device) for (k, v) in batch.items()}
                result = model(**batch)
                loss = result['loss']
                result = {k: v.cpu().detach().numpy() for (k, v) in result.items()}

                if n_gpu > 1:
                    loss = loss.mean()

                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps

                if fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 1.0)
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            except RuntimeError:
                error_count += 1
                if error_count == 3:
                    logger.exception('Three consecutive errors have been observed. Exiting...')
                    raise

                logger.exception('An unexpected error has occurred. Skipping a batch...')
                loss = None
                gc.collect()
                torch.cuda.empty_cache()
                continue

            step += 1
            error_count = 0
            tr_loss += loss.item()
            loss = None
            results.append(result)

            if step == gradient_accumulation_steps:
                scheduler.step()
                optimizer.step()
                model.zero_grad()
                step = 0

                summary = {}
                summary['learning_rate'] = max(scheduler.get_lr())
                summary['loss'] = tr_loss
                tr_loss = 0

                current_time = time.time()
                summary['batch_run_time'] = current_time - prev_time
                prev_time = current_time

                for name in ('masked_lm', 'nsp', 'masked_entity', 'entity_selector'):
                    try:
                        summary[name + '_loss'] = np.concatenate(
                            [r[name + '_loss'].flatten() for r in results]).mean()
                        correct = np.concatenate(
                            [r[name + '_correct'].flatten() for r in results]).sum()
                        total = np.concatenate(
                            [r[name + '_total'].flatten() for r in results]).sum()
                        if total > 0:
                            summary[name + '_acc'] = correct / total
                    except KeyError:
                        continue

                results = []

                if local_rank in (0, -1):
                    for (name, value) in summary.items():
                        summary_writer.add_scalar(name, value, global_step)
                    pbar.update()

                if global_step == num_train_steps:
                    break

                global_step += 1

        if local_rank in (0, -1):
            save_model(model, f'step{global_step:07}', epoch, global_step, page_chunks)
            if not page_chunks:
                save_model(model, f'epoch{epoch:03}', epoch, global_step, page_chunks)
                epoch += 1

        if global_step == num_train_steps:
            break

    if local_rank in (0, -1):
        summary_writer.close()
