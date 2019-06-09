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

from luke.batch_generator import LukePretrainingBatchGenerator, LukeE2EPretrainingBatchGenerator
from luke.model import LukeConfig, LukePretrainingModel, LukeE2EConfig, LukeE2EPretrainingModel
from luke.optimization import BertDenseSparseAdam
from luke.utils.entity_vocab import EntityVocab
from luke.utils.wiki_corpus import WikiCorpus

logger = logging.getLogger(__name__)


def run_training(corpus_file, entity_vocab_file, output_dir, bert_model_name, single_sentence, max_seq_length,
                 max_entity_length, max_mention_length, short_seq_prob, masked_lm_prob, masked_entity_prob,
                 batch_size, gradient_accumulation_steps, learning_rate, lr_schedule, warmup_steps, fix_bert_weights,
                 optimizer_on_cpu, num_train_steps, num_page_chunks, log_dir=None, model_file=None,
                 optimizer_file=None, epoch=0, global_step=0, page_chunks=[]):
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

    if masked_entity_prob == 0.0:
        model.entity_embeddings.entity_embeddings.sparse = True

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
        single_sentence=single_sentence)

    _run_training(model, batch_generator, train_args, corpus_file, output_dir, gradient_accumulation_steps,
                  learning_rate, lr_schedule, warmup_steps, optimizer_on_cpu, num_train_steps, num_page_chunks, log_dir,
                  optimizer_file, epoch, global_step, page_chunks)


def run_e2e_training(corpus_file, entity_vocab_file, output_dir, bert_model_name, single_sentence, max_seq_length,
                     max_entity_length, max_mention_length, max_candidate_length, short_seq_prob, masked_lm_prob,
                     masked_entity_prob, min_candidate_prior_prob, num_t_hidden_layers, entity_selector_softmax_temp,
                     batch_size, gradient_accumulation_steps, learning_rate, lr_schedule, warmup_steps,
                     fix_bert_weights, optimizer_on_cpu, num_train_steps, num_page_chunks, log_dir=None,
                     model_file=None, optimizer_file=None, epoch=0, global_step=0, page_chunks=[]):
    train_args = {}
    for arg in inspect.getfullargspec(run_training).args:
        train_args[arg] = locals()[arg]

    entity_vocab = EntityVocab(entity_vocab_file)
    bert_model = BertForPreTraining.from_pretrained(bert_model_name)

    config = LukeE2EConfig(entity_vocab_size=entity_vocab.size,
                           num_t_hidden_layers=num_t_hidden_layers,
                           entity_selector_softmax_temp=entity_selector_softmax_temp,
                           **bert_model.config.to_dict())
    logger.info('Model configuration: %s', config)

    model = LukeE2EPretrainingModel(config)
    if model_file is None:
        model.load_bert_weights(bert_model.state_dict())
    else:
        model.load_state_dict(torch.load(model_file, map_location='cpu'))

    if masked_entity_prob == 0.0:
        model.entity_embeddings.entity_embeddings.sparse = True
        model.entity_selector.embeddings.sparse = True

    if fix_bert_weights:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.entity_embeddings.parameters():
            param.requires_grad = True
        for param in model.entity_predictions.parameters():
            param.requires_grad = True
        for param in model.entity_selector.parameters():
            param.requires_grad = True

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
        single_sentence=single_sentence,
        min_candidate_prior_prob=min_candidate_prior_prob)

    _run_training(model, batch_generator, train_args, corpus_file, output_dir, gradient_accumulation_steps,
                  learning_rate, lr_schedule, warmup_steps, optimizer_on_cpu, num_train_steps, num_page_chunks, log_dir,
                  optimizer_file, epoch, global_step, page_chunks)


def _run_training(model, batch_generator, train_args, corpus_file, output_dir, gradient_accumulation_steps,
                  learning_rate, lr_schedule, warmup_steps, optimizer_on_cpu, num_train_steps, num_page_chunks, log_dir,
                  optimizer_file, epoch, global_step, page_chunks):
    n_gpu = torch.cuda.device_count()
    device = torch.device('cuda')
    model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(n_gpu)))

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    warmup_proportion = warmup_steps / num_train_steps

    if optimizer_on_cpu:
        optimizer_device = torch.device('cpu')
    else:
        optimizer_device = torch.device('cuda')

    optimizer = BertDenseSparseAdam(optimizer_parameters, lr=learning_rate, warmup=warmup_proportion,
                                    t_total=num_train_steps, schedule=lr_schedule, device=optimizer_device)
    if optimizer_file is not None:
        optimizer.load_state_dict(torch.load(optimizer_file, map_location='cpu'))

    model.train()

    def save_model(model, suffix, epoch, global_step, page_chunks):
        if n_gpu > 1:
            model = model.module

        torch.save(model.state_dict(), os.path.join(output_dir, f'model_{suffix}.bin'))
        torch.save(optimizer.state_dict(), os.path.join(output_dir, f'optimizer_{suffix}.bin'))

        config_dict = model.config.to_dict()

        json_data = dict(model_config=config_dict, epoch=epoch, global_step=global_step)
        with open(os.path.join(output_dir, f'model_{suffix}.json'), 'w') as f:
            json.dump(json_data, f, indent=2, sort_keys=True)

        joblib.dump(dict(args=train_args, epoch=epoch, global_step=global_step, page_chunks=page_chunks,
                         model_config=config_dict), os.path.join(output_dir, f'model_{suffix}.pkl'))

    summary_writer = SummaryWriter(log_dir)
    pbar = tqdm(total=num_train_steps, initial=global_step)

    while True:
        if not page_chunks:
            logger.info('Creating new page chunks (global_step=%d})', global_step)
            page_chunks = np.array_split(np.random.permutation(WikiCorpus(corpus_file, mmap_mode='r').page_size),
                                         num_page_chunks)

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
                step = 0

                summary = {}
                summary['learning_rate'] = max(optimizer.get_lr())
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

                for (name, value) in summary.items():
                    summary_writer.add_scalar(name, value, global_step)

                if global_step == num_train_steps:
                    break

                global_step += 1
                pbar.update()

        save_model(model, f'step{global_step:07}', epoch, global_step, page_chunks)
        if not page_chunks:
            save_model(model, f'epoch{epoch:03}', epoch, global_step, page_chunks)
            epoch += 1

        if global_step == num_train_steps:
            break

    summary_writer.close()
