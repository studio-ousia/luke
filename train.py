# -*- coding: utf-8 -*-

import gc
import importlib
import logging
import os
import time
import joblib
import numpy as np
import torch
from pytorch_pretrained_bert.modeling import BertForPreTraining
from tensorboardX import SummaryWriter
from tqdm import tqdm
from wikipedia2vec import Wikipedia2Vec

from optimization import BertAdam
from wiki_corpus import WikiCorpus
from .batch_generator import BatchGenerator

logger = logging.getLogger(__name__)


def train(corpus_data_file, entity_vocab, run_name, output_dir, log_dir, mmap, batch_size,
          learning_rate, warmup_steps, gradient_accumulation_steps, max_seq_length,
          max_entity_length, short_seq_prob, masked_lm_prob, max_predictions_per_seq,
          num_train_steps, num_page_split, entity_emb_size, link_prob_bin_size, prior_prob_bin_size,
          mask_title_words, bert_model_name, entity_emb_file, global_step=0, page_indices_list=[],
          model_file=None, optimizer_file=None, model_type='model', **kwargs):
    train_args = locals()

    model_module = importlib.import_module(model_type)
    LukeConfig = getattr(model_module, 'LukeConfig')
    LukeModel = getattr(model_module, 'LukeModel')
    LayerNorm = getattr(model_module, 'LayerNorm')

    logger.info('run name: %s', run_name)

    device = torch.device('cuda:0')
    n_gpu = torch.cuda.device_count()

    logger.info('device: {} n_gpu: {}'.format(device, n_gpu))

    batch_size = int(batch_size / gradient_accumulation_steps)

    bert_model = BertForPreTraining.from_pretrained(bert_model_name)
    logger.info('loaded BERT model: %s', bert_model_name)

    config = LukeConfig(entity_vocab_size=entity_vocab.size,
                        entity_emb_size=entity_emb_size,
                        link_prob_bin_size=link_prob_bin_size,
                        prior_prob_bin_size=prior_prob_bin_size,
                        **bert_model.config.to_dict())

    logger.info('Model configuration: %s', config)
    model = LukeModel(config)

    if model_file is None:
        model.load_bert_weights(bert_model.state_dict())
        if entity_emb_file is not None:
            entity_emb = Wikipedia2Vec.load(entity_emb_file)
            entity_emb_weights = model.entity_embeddings.entity_embeddings.weight.data.numpy()
            for title in entity_vocab:
                try:
                    vec = entity_emb.get_entity_vector(title)
                    entity_emb_weights[entity_vocab[title]] = vec
                except KeyError:
                    pass
            entity_emb_weights = torch.nn.Parameter(torch.FloatTensor(entity_emb_weights))
            model.entity_embeddings.entity_embeddings.weight = entity_emb_weights

    else:
        model.load_state_dict(torch.load(model_file, map_location='cpu'))

    del bert_model

    if n_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(0, n_gpu - 1)))

    model.to(device)
    model.train()

    parameters = {'params': [], 'weight_decay': 0.01}
    no_decay_parameters = {'params': [], 'weight_decay': 0.0}

    for module in model.modules():
        if isinstance(module, LayerNorm):
            no_decay_parameters['params'].extend(
                [p for p in module.parameters(recurse=False) if p.requires_grad])
        else:
            for (name, param) in module.named_parameters(recurse=False):
                if not param.requires_grad:
                    continue
                if 'bias' in name:
                    no_decay_parameters['params'].append(param)
                else:
                    parameters['params'].append(param)

    warmup_proportion = warmup_steps / num_train_steps

    opt_device = torch.device('cuda:' + str(n_gpu - 1))

    optimizer = BertAdam([parameters, no_decay_parameters], lr=learning_rate, device=opt_device,
                         warmup=warmup_proportion, t_total=num_train_steps)

    if optimizer_file is not None:
        optimizer.load_state_dict(torch.load(optimizer_file, map_location='cpu'))

    run_output_dir = os.path.join(output_dir, run_name)
    os.makedirs(run_output_dir, exist_ok=True)
    run_log_dir = os.path.join(log_dir, run_name)
    os.makedirs(run_log_dir, exist_ok=True)

    page_size = WikiCorpus(corpus_data_file, mmap_mode='r').page_size

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
        link_prob_bin_size=link_prob_bin_size,
        prior_prob_bin_size=prior_prob_bin_size,
        mask_title_words=mask_title_words,
        mmap=mmap)

    summary_writer = SummaryWriter(run_log_dir)
    pbar = tqdm(total=num_train_steps, initial=global_step)

    gc.collect()
    torch.cuda.empty_cache()

    while True:
        if not page_indices_list:
            page_indices_list = np.array_split(np.random.permutation(page_size), num_page_split)
        page_indices = page_indices_list.pop()

        tr_loss = 0
        results = []
        prev_time = time.time()

        # suffix = 'step%07d' % (global_step,)
        # torch.save(model.module.state_dict(),
        #             os.path.join(run_output_dir, 'model_%s.bin' % suffix))
        # data = train_args
        # data['global_step'] = global_step
        # data['page_indices_list'] = page_indices_list
        # joblib.dump(data, os.path.join(run_output_dir, 'data_%s.pkl' % suffix))

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

            loss.backward()

            tr_loss += loss.item()
            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

                summary = {}
                summary['learning_rate'] = optimizer.get_lr()[0]
                summary['loss'] = tr_loss
                tr_loss = 0

                current_time = time.time()
                summary['batch_run_time'] = current_time - prev_time
                prev_time = current_time

                for name in ('masked_lm', 'nsp', 'entity', 'page_entity'):
                    try:
                        summary[name + '_loss'] = torch.cat(
                            [r[name + '_loss'].view(-1) for r in results]).mean().item()
                        correct = torch.cat(
                            [r[name + '_correct'].view(-1) for r in results]).sum().item()
                        total = torch.cat(
                            [r[name + '_total'].view(-1) for r in results]).sum().item()
                        if total > 0:
                            summary[name + '_acc'] = correct / total
                    except KeyError:
                        continue

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

        optimizer_file = 'optimizer_%s.bin' % suffix
        torch.save(optimizer.state_dict(), os.path.join(run_output_dir, optimizer_file))

        data = train_args
        data['global_step'] = global_step
        data['page_indices_list'] = page_indices_list
        joblib.dump(data, os.path.join(run_output_dir, 'data_%s.pkl' % suffix))

        if global_step == num_train_steps:
            break

    summary_writer.close()
