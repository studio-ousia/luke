# -*- coding: utf-8 -*-

import gc
import inspect
import json
import logging
import os
import time
import joblib
import numpy as np
import torch
from pytorch_pretrained_bert.modeling import BertForPreTraining
from tensorboardX import SummaryWriter
from tqdm import tqdm

from luke.batch_generator import LukeBatchGenerator, LukeE2EBatchGenerator
from luke.model_common import LayerNorm
from luke.model import LukeConfig, LukePretrainingModel
from luke.model_e2e import LukeE2EConfig, LukeE2EPretrainingModel
from luke.optimization import BertAdam, SparseBertAdam
from luke.utils.vocab import EntityVocab
from luke.wiki_corpus import WikiCorpus

logger = logging.getLogger(__name__)


def run_training(corpus_data_file, entity_vocab_file, mmap, single_sentence, batch_size,
    gradient_accumulation_steps, max_seq_length, max_entity_length, short_seq_prob, masked_lm_prob,
    max_predictions_per_seq, masked_entity_prob, max_entity_predictions_per_seq, update_all_weights,
    entity_emb_size, bert_model_name, single_token_per_mention=False, max_mention_length=100,
    model_file=None, allocate_gpu_for_optimizer=True, **train_kwargs):
    train_args = train_kwargs.copy()
    for arg in inspect.getfullargspec(run_training).args:
        train_args[arg] = locals()[arg]

    entity_vocab = EntityVocab(entity_vocab_file)
    bert_model = BertForPreTraining.from_pretrained(bert_model_name)
    config = LukeConfig(entity_vocab_size=entity_vocab.size,
                        entity_emb_size=entity_emb_size,
                        **bert_model.config.to_dict())

    logger.info('Model configuration: %s', config)
    model = LukePretrainingModel(config)
    if model_file is None:
        model.load_bert_weights(bert_model.state_dict())
    else:
        model.load_state_dict(torch.load(model_file, map_location='cpu'))

    del bert_model

    if not update_all_weights:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.entity_embeddings.parameters():
            param.requires_grad = True
        for param in model.entity_predictions.parameters():
            param.requires_grad = True

    batch_generator = LukeBatchGenerator(
        corpus_data_file=corpus_data_file,
        entity_vocab=entity_vocab,
        batch_size=batch_size / gradient_accumulation_steps,
        max_seq_length=max_seq_length,
        max_entity_length=max_entity_length,
        max_mention_length=max_mention_length,
        short_seq_prob=short_seq_prob,
        masked_lm_prob=masked_lm_prob,
        max_predictions_per_seq=max_predictions_per_seq,
        masked_entity_prob=masked_entity_prob,
        max_entity_predictions_per_seq=max_entity_predictions_per_seq,
        single_sentence=single_sentence,
        single_token_per_mention=single_token_per_mention,
        mmap=mmap)

    _train(model, batch_generator, train_args, corpus_data_file, gradient_accumulation_steps,
           allocate_gpu_for_optimizer=allocate_gpu_for_optimizer, **train_kwargs)


def run_e2e_training(corpus_data_file, entity_vocab_file, mmap, single_sentence, batch_size,
    gradient_accumulation_steps, max_seq_length, max_entity_length, short_seq_prob, masked_lm_prob,
    max_predictions_per_seq, link_prob_bin_size, prior_prob_bin_size, entity_emb_size,
    entity_classification, bert_model_name, single_token_per_mention=False, max_mention_length=100,
    model_file=None, pretrained_model_file=None, allocate_gpu_for_optimizer=False, **train_kwargs):
    train_args = train_kwargs.copy()
    for arg in inspect.getfullargspec(run_e2e_training).args:
        train_args[arg] = locals()[arg]

    entity_vocab = EntityVocab(entity_vocab_file)
    bert_model = BertForPreTraining.from_pretrained(bert_model_name)
    config = LukeE2EConfig(entity_vocab_size=entity_vocab.size,
                           entity_emb_size=entity_emb_size,
                           link_prob_bin_size=link_prob_bin_size,
                           prior_prob_bin_size=prior_prob_bin_size,
                           entity_classification=entity_classification,
                           **bert_model.config.to_dict())

    logger.info('Model configuration: %s', config)
    model = LukeE2EPretrainingModel(config)
    if model_file is None:
        if pretrained_model_file is None:
            model.load_bert_weights(bert_model.state_dict())
        else:
            model.load_state_dict(torch.load(pretrained_model_file, map_location='cpu'),
                                  strict=False)
    else:
        model.load_state_dict(torch.load(model_file, map_location='cpu'))

    del bert_model

    batch_generator = LukeE2EBatchGenerator(
        corpus_data_file=corpus_data_file,
        entity_vocab=entity_vocab,
        batch_size=batch_size / gradient_accumulation_steps,
        max_seq_length=max_seq_length,
        max_entity_length=max_entity_length,
        max_mention_length=max_mention_length,
        short_seq_prob=short_seq_prob,
        masked_lm_prob=masked_lm_prob,
        max_predictions_per_seq=max_predictions_per_seq,
        link_prob_bin_size=link_prob_bin_size,
        prior_prob_bin_size=prior_prob_bin_size,
        single_sentence=single_sentence,
        single_token_per_mention=single_token_per_mention,
        mmap=mmap)

    _train(model, batch_generator, train_args, corpus_data_file, gradient_accumulation_steps,
           allocate_gpu_for_optimizer=allocate_gpu_for_optimizer, **train_kwargs)


def _train(model, batch_generator, train_args, corpus_data_file, gradient_accumulation_steps,
           output_dir, log_dir, learning_rate, lr_decay, warmup_steps, num_train_steps,
           num_page_chunks, save_every, allocate_gpu_for_optimizer, epoch=0, global_step=0,
           page_chunks=[], optimizer_file=None, sparse_optimizer_file=None):
    device = torch.device('cuda:0')
    n_gpu = torch.cuda.device_count()

    model.to(device)
    if n_gpu > 1:
        if allocate_gpu_for_optimizer:
            model = torch.nn.DataParallel(model, device_ids=list(range(n_gpu - 1)))
        else:
            model = torch.nn.DataParallel(model, device_ids=list(range(n_gpu)))

    parameters = {'params': [], 'weight_decay': 0.01}
    sparse_parameters = {'params': [], 'weight_decay': 0.01}
    no_decay_parameters = {'params': [], 'weight_decay': 0.0}

    params_set = set()

    for module in model.modules():
        if isinstance(module, torch.nn.Embedding) and module.sparse:
            sparse_parameters['params'].extend(
                [p for p in module.parameters(recurse=False) if p.requires_grad])
        elif isinstance(module, LayerNorm):
            no_decay_parameters['params'].extend(
                [p for p in module.parameters(recurse=False) if p.requires_grad])
        else:
            for (name, param) in module.named_parameters(recurse=False):
                if param in params_set:
                    continue
                params_set.add(param)

                if param.requires_grad:
                    if 'bias' in name:
                        no_decay_parameters['params'].append(param)
                    else:
                        parameters['params'].append(param)

    warmup_proportion = warmup_steps / num_train_steps
    if allocate_gpu_for_optimizer:
        optimizer_device = torch.device('cuda:' + str(n_gpu - 1))
    else:
        optimizer_device = torch.device('cuda:0')

    optimizer = BertAdam([parameters, no_decay_parameters], lr=learning_rate, lr_decay=lr_decay,
                         device=optimizer_device, warmup=warmup_proportion, t_total=num_train_steps)
    if optimizer_file is not None:
        optimizer.load_state_dict(torch.load(optimizer_file, map_location='cpu'))

    if sparse_parameters['params']:
        sparse_optimizer = SparseBertAdam([sparse_parameters], lr=learning_rate, lr_decay=lr_decay,
                                          device=optimizer_device, warmup=warmup_proportion,
                                          t_total=num_train_steps)
        if sparse_optimizer_file is not None:
            sparse_optimizer.load_state_dict(torch.load(sparse_optimizer_file, map_location='cpu'))
    else:
        sparse_optimizer = None

    gc.collect()
    torch.cuda.empty_cache()

    model.train()

    def save_model(model, suffix, epoch, global_step, page_chunks):
        if n_gpu > 1:
            torch.save(model.module.state_dict(), os.path.join(output_dir, 'model_%s.bin' % suffix))
            config_dict = model.module.config.to_dict()
        else:
            torch.save(model.state_dict(), os.path.join(output_dir, 'model_%s.bin' % suffix))
            config_dict = model.config.to_dict()

        optimizer_file = 'optimizer_%s.bin' % suffix
        torch.save(optimizer.state_dict(), os.path.join(output_dir, optimizer_file))

        if sparse_optimizer is not None:
            sparse_optimizer_file = 'sparse_optimizer_%s.bin' % suffix
            torch.save(sparse_optimizer.state_dict(),
                       os.path.join(output_dir, sparse_optimizer_file))

        json_data = dict(model_config=config_dict, epoch=epoch, global_step=global_step)
        with open(os.path.join(output_dir, 'model_%s.json' % suffix), 'w') as f:
            json.dump(json_data, f, indent=2, sort_keys=True)

        data = {}
        data['args'] = train_args
        data['epoch'] = epoch
        data['global_step'] = global_step
        data['page_chunks'] = page_chunks
        data['config'] = config_dict
        joblib.dump(data, os.path.join(output_dir, 'data_%s.pkl' % suffix))

    page_size = WikiCorpus(corpus_data_file, mmap_mode='r').page_size
    summary_writer = SummaryWriter(log_dir)
    pbar = tqdm(total=num_train_steps, initial=global_step)

    while True:
        if not page_chunks:
            logger.info('Creating new page chunks (step %d)', global_step)
            page_chunks = np.array_split(np.random.permutation(page_size), num_page_chunks)

        page_indices = page_chunks.pop()

        step = 0
        tr_loss = 0
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

                loss.backward()

            except RuntimeError as e:
                if 'out of memory' in str(e):
                    logger.exception('Out of memory error has occurred. Skipping a batch...')
                    loss = None
                    gc.collect()
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise

            step += 1
            tr_loss += loss.item()
            loss = None
            results.append(result)

            if step == gradient_accumulation_steps:
                optimizer.step()
                optimizer.zero_grad()
                if sparse_optimizer is not None:
                    sparse_optimizer.step()
                    sparse_optimizer.zero_grad()

                step = 0

                summary = {}
                summary['learning_rate'] = optimizer.get_lr()[0]

                summary['loss'] = tr_loss
                tr_loss = 0

                current_time = time.time()
                summary['batch_run_time'] = current_time - prev_time
                prev_time = current_time

                for name in ('masked_lm', 'nsp', 'entity', 'masked_entity'):
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

                for (name, value) in summary.items():
                    summary_writer.add_scalar(name, value, global_step)

                if global_step == num_train_steps:
                    break

                if global_step != 0 and global_step % save_every == 0:
                    save_model(model, 'step%07d' % (global_step,), epoch, global_step, page_chunks)

                global_step += 1
                pbar.update(1)

        save_model(model, 'step%07d' % (global_step,), epoch, global_step, page_chunks)
        if not page_chunks:
            save_model(model, 'epoch%03d' % (epoch,), epoch, global_step, page_chunks)
            epoch += 1

        if global_step == num_train_steps:
            break

    summary_writer.close()
