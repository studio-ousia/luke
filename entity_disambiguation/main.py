# -*- coding: utf-8 -*-

import collections
import inspect
import json
import logging
import os
import pickle
import random
import joblib
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange

from batch_generator import create_word_data
from model import LukeConfig
from model_common import LayerNorm
from optimization import BertAdam
from utils import clean_text
from utils.vocab import EntityVocab, MASK_TOKEN, PAD_TOKEN
from wiki_corpus import WikiCorpus

from .dataset import EntityDisambiguationDataset
from .model import LukeForEntityDisambiguation, LukeConfigForEntityDisambiguation

DATASET_CACHE_FILE = 'entity_disambiguation_dataset.pkl'

logger = logging.getLogger(__name__)


def generate_features(documents, tokenizer, entity_vocab, max_seq_length, max_entity_length,
                      max_candidate_size, min_context_prior_prob, prior_prob_bin_size,
                      single_token_per_mention, max_mention_length):
    ret = []
    max_num_tokens = max_seq_length - 2
    for document in tqdm(documents):
        orig_words = [clean_text(w) for w in document.words]
        mention_start_map = {m.span[0]: m for m in document.mentions}
        mention_end_map = {m.span[1]: m for m in document.mentions}

        tokens = []
        for (orig_pos, orig_word) in enumerate(orig_words):
            if orig_pos in mention_start_map:
                mention_start_map[orig_pos].start = len(tokens)

            if orig_pos in mention_end_map:
                mention_end_map[orig_pos].end = len(tokens)

            tokens.extend(tokenizer.tokenize(orig_word))

        for target_mention in document.mentions:
            target_span = target_mention.span

            mention_size = target_span[1] - target_span[0]
            half_context_size = int((max_num_tokens - mention_size) / 2)

            left_token_size = target_span[0]
            right_token_size = len(tokens) - target_span[1]
            if left_token_size < right_token_size:
                left_context_size = min(left_token_size, half_context_size)
                right_context_size = min(right_token_size,
                                         max_num_tokens - left_context_size - mention_size)
            else:
                right_context_size = min(right_token_size, half_context_size)
                left_context_size = min(left_token_size,
                                        max_num_tokens - right_context_size - mention_size)

            token_start = target_span[0] - left_context_size
            token_end = target_span[1] + right_context_size
            target_tokens = tokens[token_start:target_span[0]]
            target_tokens += tokens[target_span[0]:target_span[1]]
            target_tokens += tokens[target_span[1]:token_end]

            word_data = create_word_data(target_tokens, None, tokenizer.vocab, max_seq_length)

            entity_ids = np.zeros(max_entity_length, dtype=np.int)
            entity_ids[0] = entity_vocab[MASK_TOKEN]

            if single_token_per_mention:
                entity_position_ids = np.full((max_entity_length, max_mention_length), -1,
                                              dtype=np.int)
                entity_position_ids[0][:mention_size] = range(left_context_size + 1,
                                                              left_context_size + mention_size + 1)
            else:
                entity_position_ids = np.zeros(max_entity_length, dtype=np.int)
                entity_position_ids[0] = left_context_size + 1

            entity_index = 1
            for mention in document.mentions:
                if mention == target_mention:
                    continue

                if entity_index == max_entity_length:
                    break

                start = mention.start - token_start
                end = mention.end - token_start

                if start < 0 or end > max_num_tokens:
                    continue

                mention_size = end - start
                for candidate in mention.candidates:
                    if candidate.prior_prob <= min_context_prior_prob:
                        continue
                    entity_ids[entity_index] = entity_vocab[candidate.title]
                    entity_position_ids[entity_index][:mention_size] = range(start + 1, end + 1)
                    entity_index += 1
                    if entity_index == max_entity_length:
                        break

            entity_segment_ids = np.zeros(max_entity_length, dtype=np.int)
            entity_attention_mask = np.zeros(max_entity_length, dtype=np.int)
            entity_attention_mask[:entity_index] = 1

            entity_candidate_ids = np.zeros(max_candidate_size, dtype=np.int)
            entity_prior_prob_ids = np.zeros(max_candidate_size, dtype=np.int)

            candidates = target_mention.candidates[:max_candidate_size]
            entity_candidate_ids[:len(candidates)] = [entity_vocab[c.title] for c in candidates]
            if prior_prob_bin_size != 0:
                entity_prior_prob_ids[:len(candidates)] = [
                    min(int(c.prior_prob * prior_prob_bin_size), prior_prob_bin_size - 1)
                    for c in candidates]

            entity_label = entity_vocab[target_mention.title]

            feature = dict(word_ids=word_data['word_ids'],
                           word_segment_ids=word_data['word_segment_ids'],
                           word_attention_mask=word_data['word_attention_mask'],
                           entity_ids=entity_ids,
                           entity_position_ids=entity_position_ids,
                           entity_segment_ids=entity_segment_ids,
                           entity_attention_mask=entity_attention_mask,
                           entity_candidate_ids=entity_candidate_ids,
                           entity_prior_prob_ids=entity_prior_prob_ids,
                           entity_label=entity_label)

            ret.append((document, target_mention, feature))

    return ret


def run(data_dir, dump_db, model_file, output_dir, max_seq_length, max_entity_length,
        max_candidate_size, min_context_prior_prob, prior_prob_bin_size, batch_size,
        eval_batch_size, learning_rate, iteration, warmup_proportion, lr_decay, seed,
        gradient_accumulation_steps, fix_entity_emb, test_set):
    device = torch.device('cuda')
    n_gpu = torch.cuda.device_count()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    logger.info('Loading model and configurations...')

    state_dict = torch.load(model_file, map_location='cpu')
    data_file = model_file.replace('.bin', '.pkl').replace('model', 'data')
    model_data = joblib.load(data_file)

    config = LukeConfigForEntityDisambiguation(prior_prob_bin_size=prior_prob_bin_size,
                                               **model_data['config'])
    corpus = WikiCorpus(model_data['args']['corpus_data_file'])
    orig_entity_vocab = EntityVocab(model_data['args']['entity_vocab_file'])
    single_token_per_mention = model_data['args'].get('single_token_per_mention', False)
    # max_mention_length = model_data['args'].get('max_mention_length', 100)
    max_mention_length = 20

    logger.info('Loading dataset...')

    if os.path.exists(DATASET_CACHE_FILE):
        logger.info('Using cache: %s', DATASET_CACHE_FILE)
        with open(DATASET_CACHE_FILE, mode='rb') as f:
            (dataset, entity_titles) = pickle.load(f)
    else:
        dataset = EntityDisambiguationDataset(data_dir)
        # build entity vocabulary and resolve Wikipedia redirects
        entity_titles = set([MASK_TOKEN])
        for documents in dataset.get_all_datasets():
            for document in documents:
                for mention in document.mentions:
                    mention.title = dump_db.resolve_redirect(mention.title)
                    entity_titles.add(mention.title)
                    for candidate in mention.candidates:
                        candidate.title = dump_db.resolve_redirect(candidate.title)
                        entity_titles.add(candidate.title)

        with open(DATASET_CACHE_FILE, mode='wb') as f:
            pickle.dump((dataset, entity_titles), f)

    # build a vocabulary, embeddings and biases of entities contained in the dataset
    orig_entity_emb = state_dict['entity_embeddings.entity_embeddings.weight']
    orig_entity_bias = state_dict['entity_predictions.bias']
    entity_emb = orig_entity_emb.new_zeros((len(entity_titles) + 1, config.entity_emb_size))
    entity_bias = orig_entity_bias.new_zeros(len(entity_titles) + 1)
    entity_vocab = {PAD_TOKEN: 0}
    for (n, title) in enumerate(entity_titles, 1):
        entity_vocab[title] = n
        if title in orig_entity_vocab:
            orig_index = orig_entity_vocab[title]
            entity_emb[n] = orig_entity_emb[orig_index]
            entity_bias[n] = orig_entity_bias[orig_index]

    config.entity_vocab_size = len(entity_vocab)
    state_dict['entity_embeddings.entity_embeddings.weight'] = entity_emb
    state_dict['entity_predictions.decoder.weight'] = entity_emb
    state_dict['entity_predictions.bias'] = entity_bias

    logger.info('Instantiating LukeForEntityDisambiguation model')
    logger.info('Model configuration: %s', config)

    model = LukeForEntityDisambiguation(config)
    model.load_state_dict(state_dict, strict=False)

    logger.info('Fix entity embeddings during training: %s', fix_entity_emb)
    if fix_entity_emb:
        model.entity_embeddings.entity_embeddings.weight.requires_grad = False

    train_batch_size = int(batch_size / gradient_accumulation_steps)
    model_arg_names = inspect.getfullargspec(LukeForEntityDisambiguation.forward)[0][1:]

    logger.info('Creating TensorDataset for training')

    train_data = generate_features(dataset.train, corpus.tokenizer, entity_vocab, max_seq_length,
                                   max_entity_length, max_candidate_size, min_context_prior_prob,
                                   prior_prob_bin_size, single_token_per_mention, max_mention_length)
    train_tensors = TensorDataset(*[torch.tensor([f[k] for (_, _, f) in train_data], dtype=torch.long)
                                    for k in model_arg_names])
    train_sampler = RandomSampler(train_tensors)
    train_dataloader = DataLoader(train_tensors, sampler=train_sampler, batch_size=train_batch_size)

    model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    num_train_steps = int(len(train_tensors) / batch_size * iteration)

    parameters = {'params': [], 'weight_decay': 0.01}
    no_decay_parameters = {'params': [], 'weight_decay': 0.0}

    params_set = set()
    for module in model.modules():
        if isinstance(module, LayerNorm):
            no_decay_parameters['params'].extend(list(module.parameters(recurse=False)))
        else:
            for (name, param) in module.named_parameters(recurse=False):
                if param in params_set:
                    continue
                params_set.add(param)

                if 'bias' in name:
                    no_decay_parameters['params'].append(param)
                else:
                    parameters['params'].append(param)

    opt_device = torch.device('cuda:0')
    optimizer = BertAdam([parameters, no_decay_parameters], lr=learning_rate, device=opt_device,
                         warmup=warmup_proportion, lr_decay=lr_decay, t_total=num_train_steps)
    global_step = 0

    if output_dir:
        writer = open(os.path.join(output_dir, 'eval_results.jl'), 'a')

    def evaluate(model, target, global_step, n_iter):
        model.eval()

        with open(DATASET_CACHE_FILE, mode='rb') as f:
            (dataset, entity_titles) = pickle.load(f)
        documents = getattr(dataset, target)
        eval_data = generate_features(documents, corpus.tokenizer, entity_vocab, max_seq_length,
                                      max_entity_length, max_candidate_size, min_context_prior_prob,
                                      prior_prob_bin_size, single_token_per_mention, max_mention_length)
        eval_tensors = TensorDataset(*[torch.tensor([f[k] for (_, _, f) in eval_data], dtype=torch.long)
                                    for k in model_arg_names])
        eval_sampler = SequentialSampler(eval_tensors)
        eval_dataloader = DataLoader(eval_tensors, sampler=eval_sampler, batch_size=eval_batch_size)

        (precision, recall, f1) = compute_precision_recall_f1(model, eval_data, eval_dataloader, device)

        result = dict(
            global_step=global_step,
            model_file=model_file,
            batch_size=batch_size,
            learning_rate=learning_rate,
            total_iteration=iteration,
            warmup_proportion=warmup_proportion,
            seed=seed,
            lr_decay=lr_decay,
            max_seq_length=max_seq_length,
            iteration=n_iter,
        )
        result['eval_precision'] = precision
        result['eval_recall'] = recall
        result['eval_f1'] = f1

        logger.info("***** Eval results: %s *****", target)
        for key in sorted(result.keys()):
            if key.startswith('eval_') or key in ('loss', 'global_step', 'iteration'):
                logger.info("  %s = %s", key, str(result[key]))

        if output_dir:
            writer.write("%s\n" % json.dumps(result, sort_keys=True))
            writer.flush()

    for n_iter in trange(int(iteration), desc='Epoch'):
        for target in test_set:
            evaluate(model, target, global_step, n_iter)

        model.train()

        nb_tr_examples = 0
        nb_tr_steps = 0
        for (step, batch) in enumerate(tqdm(train_dataloader, desc='Iteration')):
            batch = tuple(t.to(device) for t in batch)
            loss = model(*batch)
            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps
            loss.backward()
            nb_tr_examples += batch[0].size(0)
            nb_tr_steps += 1
            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                model.zero_grad()
                global_step += 1

    for target in test_set:
        evaluate(model, target, global_step, n_iter)

    if output_dir:
        writer.close()


def compute_precision_recall_f1(model, eval_data, eval_dataloader, device):
    eval_logits = []
    eval_labels = []
    for batch in tqdm(eval_dataloader):
        args = [t.to(device) for t in batch[:-1]]
        with torch.no_grad():
            logits = model(*args)

        eval_logits.append(logits.detach().cpu().numpy())
        eval_labels.append(batch[-1].numpy())

    eval_labels = np.concatenate(eval_labels)
    outputs = np.argmax(np.vstack(eval_logits), axis=1)

    num_correct = 0
    num_mentions = 0
    num_mentions_with_candidates = 0
    for (predicted, correct, (document, mention, _)) in zip(outputs, eval_labels, eval_data):
        if predicted == correct:
            num_correct += 1

        num_mentions += 1
        if mention.candidates:
            num_mentions_with_candidates += 1

    precision = num_correct / num_mentions_with_candidates
    recall = num_correct / num_mentions
    f1 = 2.0 * precision * recall / (precision + recall)

    return (precision, recall, f1)
