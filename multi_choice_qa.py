# -*- coding: utf-8 -*-

import csv
import importlib
import json
import os
import logging
import random
from tqdm import tqdm, trange

import joblib
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from pytorch_pretrained_bert.modeling import BertForPreTraining

from optimization import BertAdam
from utils import clean_text

logger = logging.getLogger(__name__)


class Example(object):
    def __init__(self, example_id, context_sentence, start_ending, endings, label=None):
        self.example_id = example_id
        self.context_sentence = context_sentence
        self.start_ending = start_ending
        self.endings = endings
        self.label = label

    def __str__(self):
        return self.__repr__()


class InputFeatures(object):
    def __init__(self, word_ids, word_segment_ids, word_attention_mask, entity_ids,
                 entity_position_ids, entity_segment_ids, entity_attention_mask,
                 entity_link_prob_ids, entity_prior_prob_ids, answer_mask, label_id):
        self.word_ids = word_ids
        self.word_segment_ids = word_segment_ids
        self.word_attention_mask = word_attention_mask
        self.entity_ids = entity_ids
        self.entity_position_ids = entity_position_ids
        self.entity_segment_ids = entity_segment_ids
        self.entity_attention_mask = entity_attention_mask
        self.entity_link_prob_ids = entity_link_prob_ids
        self.entity_prior_prob_ids = entity_prior_prob_ids
        self.answer_mask = answer_mask
        self.label_id = label_id


def read_examples(dataset_name, data_dir, fold):
    reader = {'arc-easy': read_arc_examples, 'arc-challenge': read_arc_examples,
              'openbookqa': read_arc_examples, 'swag': read_swag_examples}
    return reader[dataset_name](data_dir, fold)


def read_arc_examples(data_dir, fold):
    input_file = os.path.join(data_dir, fold + '.jsonl')
    examples = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            example_obj = json.loads(line)
            label_to_id = {}
            endings = []
            for (index, choice) in enumerate(example_obj['question']['choices']):
                label_to_id[choice['label']] = index
                endings.append(choice['text'])

            example = Example(example_id=example_obj['id'],
                              context_sentence=example_obj['question']['stem'],
                              start_ending='',
                              endings=endings,
                              label=label_to_id[example_obj['answerKey']])
            examples.append(example)

    return examples


def read_swag_examples(data_dir, fold):
    input_file = os.path.join(data_dir, fold + '.csv')
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = [l for l in csv.reader(f)]

    examples = []
    for line in lines[1:]:
        example = Example(example_id=line[2],
                          context_sentence=line[4],
                          start_ending=line[5],
                          endings=line[7:11],
                          label=int(line[11]))
        examples.append(example)

    return examples


def convert_examples_to_features(examples, tokenizer, entity_linker, entity_vocab, max_seq_length,
                                 max_entity_length, min_prior_prob, link_prob_bin_size,
                                 prior_prob_bin_size, num_choices):
    def detect_mentions(text, tokens):
        token_start_map = np.full(len(text), -1)
        token_end_map = np.full(len(text), -1)
        for (ind, token) in enumerate(tokens):
            token_start_map[token.start] = ind
            token_end_map[token.end - 1] = ind

        detected_mentions = []
        for (mention_span, mentions) in entity_linker.detect_mentions(text):
            for mention in mentions:
                if mention.prior_prob < min_prior_prob:
                    continue

                token_start = token_start_map[mention_span[0]]
                if token_start == -1:
                    continue

                token_end = token_end_map[mention_span[1] - 1]
                if token_end == -1:
                    continue
                token_end += 1

                if mention.title in entity_vocab:
                    for position in range(token_start, token_end):
                        detected_mentions.append((position, entity_vocab[mention.title],
                                                  mention.link_prob, mention.prior_prob))

        return detected_mentions


    features = []

    for (example_index, example) in enumerate(examples):
        context_sentence = clean_text(example.context_sentence)
        context_tokens = tokenizer.tokenize(context_sentence)
        context_mentions = detect_mentions(context_sentence, context_tokens)

        ending_tokens_list = []
        ending_mentions_list = []
        for ending_text in example.endings:
            text = clean_text(example.start_ending + ending_text)
            tokens = tokenizer.tokenize(text)
            mentions = detect_mentions(text, tokens)
            ending_tokens_list.append(tokens)
            ending_mentions_list.append(mentions)

        answer_mask = np.zeros(num_choices, dtype=np.int)
        choices_features = []
        for ending_index in range(num_choices):
            if ending_index < len(example.endings):
                answer_mask[ending_index] = 1
                context_tokens_choice = context_tokens[:]
                context_mentions_choice = context_mentions[:]
                ending_tokens = ending_tokens_list[ending_index]
                ending_mentions = ending_mentions_list[ending_index]

                _truncate_seq_pair(context_tokens_choice, ending_tokens, max_seq_length - 5)
                _truncate_seq_pair(context_mentions_choice, ending_mentions, max_entity_length)

                word_ids = np.zeros(max_seq_length, dtype=np.int)
                entity_ids = np.zeros(max_entity_length, dtype=np.int)
                entity_position_ids = np.zeros(max_entity_length, dtype=np.int)
                entity_link_prob_ids = np.zeros(max_entity_length, dtype=np.int)
                entity_prior_prob_ids = np.zeros(max_entity_length, dtype=np.int)

                word_ids[0] = tokenizer.vocab['[CLS]']
                word_ids[1] = tokenizer.vocab['[unused99]']
                word_ids[2] = tokenizer.vocab['[unused99]']
                for (n, token) in enumerate(context_tokens_choice, 3):
                    word_ids[n] = token.id
                word_ids[n + 1] = tokenizer.vocab['[SEP]']

                for (n, (pos, entity_id, link_prob, prior_prob)) in enumerate(context_mentions_choice):
                    entity_ids[n] = entity_id
                    entity_position_ids[n] = pos + 3
                    entity_link_prob_ids[n] = int(link_prob * (link_prob_bin_size - 1))
                    entity_prior_prob_ids[n] = int(prior_prob * (prior_prob_bin_size - 1))

                for (n, token) in enumerate(ending_tokens, len(context_tokens_choice) + 4):
                    word_ids[n] = token.id
                word_ids[n + 1] = tokenizer.vocab['[SEP]']

                for (n, (pos, entity_id, link_prob, prior_prob)) in enumerate(ending_mentions,
                    len(context_mentions_choice)):
                    entity_ids[n] = entity_id
                    entity_position_ids[n] = pos + len(context_tokens_choice) + 4
                    entity_link_prob_ids[n] = int(link_prob * (link_prob_bin_size - 1))
                    entity_prior_prob_ids[n] = int(prior_prob * (prior_prob_bin_size - 1))

                word_attention_mask = np.ones(max_seq_length, dtype=np.int)
                word_attention_mask[len(context_tokens_choice) + len(ending_tokens) + 5:] = 0
                word_segment_ids = np.zeros(max_seq_length, dtype=np.int)
                word_segment_ids[len(context_tokens_choice) + 4:
                                len(context_tokens_choice) + len(ending_tokens) + 5] = 1

                entity_attention_mask = np.ones(max_entity_length, dtype=np.int)
                entity_attention_mask[len(context_mentions_choice) + len(ending_mentions):] = 0
                entity_segment_ids = np.zeros(max_entity_length, dtype=np.int)
                entity_segment_ids[len(context_mentions_choice):
                                len(context_mentions_choice) + len(ending_mentions)] = 1

            else:
                word_ids = word_attention_mask = word_segment_ids = np.zeros(max_seq_length, dtype=np.int)
                entity_ids = entity_attention_mask = entity_segment_ids = entity_position_ids =\
                    entity_link_prob_ids = entity_prior_prob_ids = np.zeros(max_entity_length, dtype=np.int)

            choices_features.append(dict(
                word_ids=word_ids,
                word_attention_mask=word_attention_mask,
                word_segment_ids=word_segment_ids,
                entity_ids=entity_ids,
                entity_attention_mask=entity_attention_mask,
                entity_position_ids=entity_position_ids,
                entity_segment_ids=entity_segment_ids,
                entity_link_prob_ids=entity_link_prob_ids,
                entity_prior_prob_ids=entity_prior_prob_ids
            ))

        stacked_features = {
            k: np.vstack([choices_features[n][k] for n in range(len(choices_features))])
            for k in choices_features[0].keys()
        }

        label_id = example.label
        if example_index == 0:
            pass
            # logger.info("*** Example ***")
            # logger.info("swag_id: {}".format(example.swag_id))
            # for choice_idx, (tokens, input_ids, input_mask, segment_ids) in enumerate(choices_features):
            #     logger.info("choice: {}".format(choice_idx))
            #     logger.info("tokens: {}".format(' '.join(tokens)))
            #     logger.info("input_ids: {}".format(' '.join(map(str, input_ids))))
            #     logger.info("input_mask: {}".format(' '.join(map(str, input_mask))))
            #     logger.info("segment_ids: {}".format(' '.join(map(str, segment_ids))))
            # if is_training:
            #     logger.info("label: {}".format(label))

        features.append(InputFeatures(**stacked_features, answer_mask=answer_mask,
                                      label_id=label_id))

    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def run(tokenizer, entity_linker, data_dir, task_name, model_file, output_dir, max_seq_length,
        max_entity_length, min_prior_prob, batch_size, eval_batch_size, learning_rate,
        iteration, warmup_proportion, lr_decay, seed, gradient_accumulation_steps, fix_entity_emb):
    device = torch.device("cuda")
    n_gpu = torch.cuda.device_count()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

    data_dir = os.path.join(data_dir, task_name)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if task_name.startswith('arc'):
        num_choices = 5
    else:
        num_choices = 4

    data_file = model_file.replace('.bin', '.pkl').replace('model', 'data')
    model_data = joblib.load(data_file)

    train_batch_size = int(batch_size / gradient_accumulation_steps)

    model_type = model_data['model_type']
    model_module = importlib.import_module('luke.' + model_type)
    LukeConfig = getattr(model_module, 'LukeConfig')
    LukeForMultipleChoice = getattr(model_module, 'LukeForMultipleChoice')
    LayerNorm = getattr(model_module, 'LayerNorm')

    bert_model = BertForPreTraining.from_pretrained(model_data['bert_model_name'])

    config = LukeConfig(entity_vocab_size=model_data['entity_vocab'].size,
                        entity_emb_size=model_data['entity_emb_size'],
                        link_prob_bin_size=model_data['link_prob_bin_size'],
                        prior_prob_bin_size=model_data['prior_prob_bin_size'],
                        **bert_model.config.to_dict())
    del bert_model

    train_examples = None
    num_train_steps = None
    train_examples = read_examples(task_name, data_dir, 'train')
    num_train_steps = int(len(train_examples) / train_batch_size * iteration)

    # Prepare model
    state_dict = torch.load(model_file, map_location='cpu')
    model = LukeForMultipleChoice(config, num_choices=num_choices)
    model.load_state_dict(state_dict, strict=False)

    if fix_entity_emb:
        try:
            model.entity_embeddings.entity_embeddings.weight.requires_grad = False
        except AttributeError:
            pass

    model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    parameters = {'params': [], 'weight_decay': 0.01}
    no_decay_parameters = {'params': [], 'weight_decay': 0.0}

    for module in model.modules():
        if isinstance(module, LayerNorm):
            no_decay_parameters['params'].extend(
                list(module.parameters(recurse=False)))
        else:
            for (name, param) in module.named_parameters(recurse=False):
                if 'bias' in name:
                    no_decay_parameters['params'].append(param)
                else:
                    parameters['params'].append(param)

    opt_device = torch.device('cuda:0')
    optimizer = BertAdam([parameters, no_decay_parameters], lr=learning_rate, device=opt_device,
                         warmup=warmup_proportion, lr_decay=lr_decay, t_total=num_train_steps)

    global_step = 0

    train_features = convert_examples_to_features(
        train_examples, tokenizer, entity_linker, model_data['entity_vocab'], max_seq_length,
        max_entity_length, min_prior_prob, model_data['link_prob_bin_size'],
        model_data['prior_prob_bin_size'], num_choices
    )

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_examples))
    logger.info("  Batch size = %d", train_batch_size)
    logger.info("  Learning rate = %f", learning_rate)
    logger.info("  Minimum Prior Probability = %f", min_prior_prob)
    logger.info("  Iteration = %d", iteration)
    train_data = TensorDataset(
        torch.tensor([f.word_ids for f in train_features], dtype=torch.long),
        torch.tensor([f.word_segment_ids for f in train_features], dtype=torch.long),
        torch.tensor([f.word_attention_mask for f in train_features], dtype=torch.long),
        torch.tensor([f.entity_ids for f in train_features], dtype=torch.long),
        torch.tensor([f.entity_position_ids for f in train_features], dtype=torch.long),
        torch.tensor([f.entity_segment_ids for f in train_features], dtype=torch.long),
        torch.tensor([f.entity_attention_mask for f in train_features], dtype=torch.long),
        torch.tensor([f.entity_link_prob_ids for f in train_features], dtype=torch.long),
        torch.tensor([f.entity_prior_prob_ids for f in train_features], dtype=torch.long),
        torch.tensor([f.answer_mask for f in train_features], dtype=torch.long),
        torch.tensor([f.label_id for f in train_features], dtype=torch.long)
    )

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(
        train_data, sampler=train_sampler, batch_size=train_batch_size)

    eval_examples = read_examples(task_name, data_dir, 'dev')
    eval_features = convert_examples_to_features(
        eval_examples, tokenizer, entity_linker, model_data['entity_vocab'], max_seq_length,
        max_entity_length, min_prior_prob, model_data['link_prob_bin_size'],
        model_data['prior_prob_bin_size'], num_choices
    )

    eval_data = TensorDataset(
        torch.tensor([f.word_ids for f in eval_features], dtype=torch.long),
        torch.tensor([f.word_segment_ids for f in eval_features], dtype=torch.long),
        torch.tensor([f.word_attention_mask for f in eval_features], dtype=torch.long),
        torch.tensor([f.entity_ids for f in eval_features], dtype=torch.long),
        torch.tensor([f.entity_position_ids for f in eval_features], dtype=torch.long),
        torch.tensor([f.entity_segment_ids for f in eval_features], dtype=torch.long),
        torch.tensor([f.entity_attention_mask for f in eval_features], dtype=torch.long),
        torch.tensor([f.entity_link_prob_ids for f in eval_features], dtype=torch.long),
        torch.tensor([f.entity_prior_prob_ids for f in eval_features], dtype=torch.long),
        torch.tensor([f.answer_mask for f in eval_features], dtype=torch.long),
        torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
    )
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=eval_batch_size)

    if output_dir:
        writer = open(os.path.join(output_dir, "eval_results.jl"), 'a')

    for n_iter in trange(int(iteration), desc="Epoch"):
        model.train()

        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for (step, batch) in enumerate(tqdm(train_dataloader, desc="Iteration")):
            batch = tuple(t.to(device) for t in batch)
            (word_ids, word_segment_ids, word_attention_mask, entity_ids, entity_position_ids,
             entity_segment_ids, entity_attention_mask, entity_link_prob_ids, entity_prior_prob_ids,
             answer_mask, label_ids) = batch
            loss = model(word_ids=word_ids,
                         word_segment_ids=word_segment_ids,
                         word_attention_mask=word_attention_mask,
                         entity_ids=entity_ids,
                         entity_position_ids=entity_position_ids,
                         entity_segment_ids=entity_segment_ids,
                         entity_attention_mask=entity_attention_mask,
                         entity_link_prob_ids=entity_link_prob_ids,
                         entity_prior_prob_ids=entity_prior_prob_ids,
                         answer_mask=answer_mask,
                         labels=label_ids)
            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps
            loss.backward()
            tr_loss += loss.item()
            nb_tr_examples += batch[0].size(0)
            nb_tr_steps += 1
            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                model.zero_grad()
                global_step += 1

        model.eval()

        eval_loss = 0
        eval_logits = []
        eval_labels = []
        nb_eval_steps, nb_eval_examples = 0, 0
        for batch in eval_dataloader:
            batch = tuple(t.to(device) for t in batch)
            (word_ids, word_segment_ids, word_attention_mask, entity_ids, entity_position_ids,
             entity_segment_ids, entity_attention_mask, entity_link_prob_ids, entity_prior_prob_ids,
             answer_mask, label_ids) = batch
            with torch.no_grad():
                tmp_eval_loss = model(word_ids=word_ids,
                                      word_segment_ids=word_segment_ids,
                                      word_attention_mask=word_attention_mask,
                                      entity_ids=entity_ids,
                                      entity_position_ids=entity_position_ids,
                                      entity_segment_ids=entity_segment_ids,
                                      entity_attention_mask=entity_attention_mask,
                                      entity_link_prob_ids=entity_link_prob_ids,
                                      entity_prior_prob_ids=entity_prior_prob_ids,
                                      answer_mask=answer_mask,
                                      labels=label_ids)
                logits = model(word_ids=word_ids,
                               word_segment_ids=word_segment_ids,
                               word_attention_mask=word_attention_mask,
                               entity_ids=entity_ids,
                               entity_position_ids=entity_position_ids,
                               entity_segment_ids=entity_segment_ids,
                               entity_attention_mask=entity_attention_mask,
                               entity_link_prob_ids=entity_link_prob_ids,
                               entity_prior_prob_ids=entity_prior_prob_ids,
                               answer_mask=answer_mask)

            eval_logits.append(logits.detach().cpu().numpy())
            eval_labels.append(label_ids.cpu().numpy())

            eval_loss += tmp_eval_loss.mean().item()

            nb_eval_examples += batch[0].size(0)
            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps
        outputs = np.argmax(np.vstack(eval_logits), axis=1)
        eval_accuracy = np.sum(outputs == np.concatenate(eval_labels)) / nb_eval_examples

        result = dict(
            task_name=task_name,
            eval_loss=eval_loss,
            eval_accuracy=eval_accuracy,
            global_step=global_step,
            loss=tr_loss/nb_tr_steps,
            model_file=model_file,
            model_type=model_type,
            batch_size=batch_size,
            learning_rate=learning_rate,
            iteration=n_iter,
            total_iteration=iteration,
            min_prior_prob=min_prior_prob,
            warmup_proportion=warmup_proportion,
            seed=seed,
            lr_decay=lr_decay,
            max_seq_length=max_seq_length,
            max_entity_length=max_entity_length,
            fix_entity_emb=fix_entity_emb,
        )

        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            if key in ('eval_loss', 'eval_accuracy', 'loss', 'global_step', 'iteration'):
                logger.info("  %s = %s", key, str(result[key]))
        if output_dir:
            writer.write("%s\n" % json.dumps(result, sort_keys=True))
            writer.flush()

    if output_dir:
        writer.close()
