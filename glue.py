# -*- coding: utf-8 -*-

import csv
import importlib
import json
import os
import logging
import random
import joblib
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from pytorch_pretrained_bert.modeling import BertForPreTraining
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm, trange

from optimization import BertAdam, warmup_linear
from utils import clean_text

logger = logging.getLogger(__name__)


class InputExample(object):
    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    def __init__(self, word_ids, word_segment_ids, word_attention_mask, entity_ids,
                 entity_position_ids, entity_segment_ids, entity_attention_mask,
                 entity_link_prob_ids, entity_prior_prob_ids, label):
        self.word_ids = word_ids
        self.word_segment_ids = word_segment_ids
        self.word_attention_mask = word_attention_mask
        self.entity_ids = entity_ids
        self.entity_position_ids = entity_position_ids
        self.entity_segment_ids = entity_segment_ids
        self.entity_attention_mask = entity_attention_mask
        self.entity_link_prob_ids = entity_link_prob_ids
        self.entity_prior_prob_ids = entity_prior_prob_ids
        self.label = label


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class MrpcProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    @property
    def task_type(self):
        return 'classification'

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(
            os.path.join(data_dir, "train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            text_b = line[4]
            label = line[0]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class QqpProcessor(DataProcessor):
    """Processor for the Quora Question Pairs data set."""

    @property
    def task_type(self):
        return 'classification'

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(
            os.path.join(data_dir, "train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            if len(line) <= 5:
                print(line)
                continue
            text_a = line[3]
            text_b = line[4]
            label = line[5]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    @property
    def task_type(self):
        return 'classification'

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")),
            "dev_matched")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[8]
            text_b = line[9]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class QnliProcessor(DataProcessor):
    """Processor for the QultiNLI data set"""

    @property
    def task_type(self):
        return 'classification'

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")),
            "dev_matched")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[3]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class RTEProcessor(DataProcessor):
    """Processor for the RTE data set (GLUE version)."""

    @property
    def task_type(self):
        return 'classification'

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev_matched")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[3]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class ColaProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    @property
    def task_type(self):
        return 'classification'

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class SciTailProcessor(DataProcessor):
    @property
    def task_type(self):
        return 'classification'

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")),
            "dev_matched")

    def get_labels(self):
        """See base class."""
        return ["entails", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%d" % (set_type, i)
            text_a = line[0]
            text_b = line[1]
            label = line[2]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class STSProcessor(DataProcessor):
    """Processor for the STS-B data set.
    Obtained from: https://github.com/Colanim/BERT_STS-B
    """

    @property
    def task_type(self):
        return 'regression'

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    # # ADDED
    # def get_test_examples(self, data_dir):
    #     """See base class."""
    #     return self._create_examples(
    #         self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[-3]
            text_b = line[-2]
            label = float(line[-1])
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        return examples


def convert_examples_to_features(
    examples, task_type, label_list, tokenizer, entity_linker, entity_vocab, max_seq_length,
    max_entity_length, min_prior_prob, link_prob_bin_size, prior_prob_bin_size):
    if task_type == 'classification':
        label_map = {}
        for (i, label) in enumerate(label_list):
            label_map[label] = i

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_ab = [[], []]
        mentions_ab = [[], []]
        for (n, text) in enumerate((example.text_a, example.text_b)):
            if not text:
                continue

            text = clean_text(text, strip_accents=True)

            tokens = tokenizer.tokenize(text)
            tokens_ab[n] = tokens

            token_start_map = np.full(len(text), -1)
            token_end_map = np.full(len(text), -1)
            for (ind, token) in enumerate(tokens):
                token_start_map[token.start] = ind
                token_end_map[token.end - 1] = ind

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
                            mentions_ab[n].append((position, entity_vocab[mention.title],
                                                   mention.link_prob, mention.prior_prob))

        (tokens_a, tokens_b) = tokens_ab
        (mentions_a, mentions_b) = mentions_ab

        if tokens_b:
            # Account for [CLS], [ENT], [ENT], [SEP], [SEP] with "- 5"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 5)
            _truncate_seq_pair(mentions_a, mentions_b, max_entity_length)
        else:
            # Account for [CLS], [ENT], [ENT], and [SEP] with "- 4"
            if len(tokens_a) > max_seq_length - 4:
                tokens_a = tokens_a[0:(max_seq_length - 4)]
            if len(mentions_a) > max_entity_length:
                mentions_a = mentions_a[0:max_entity_length]

        word_ids = np.zeros(max_seq_length, dtype=np.int)
        entity_ids = np.zeros(max_entity_length, dtype=np.int)
        entity_position_ids = np.zeros(max_entity_length, dtype=np.int)
        entity_link_prob_ids = np.zeros(max_entity_length, dtype=np.int)
        entity_prior_prob_ids = np.zeros(max_entity_length, dtype=np.int)

        word_ids[0] = tokenizer.vocab['[CLS]']
        word_ids[1] = tokenizer.vocab['[unused99]']
        word_ids[2] = tokenizer.vocab['[unused99]']
        for (n, token) in enumerate(tokens_a, 3):
            word_ids[n] = token.id
        word_ids[n + 1] = tokenizer.vocab['[SEP]']

        for (n, (pos, entity_id, link_prob, prior_prob)) in enumerate(mentions_a):
            entity_ids[n] = entity_id
            entity_position_ids[n] = pos + 3
            entity_link_prob_ids[n] = int(link_prob * (link_prob_bin_size - 1))
            entity_prior_prob_ids[n] = int(prior_prob * (prior_prob_bin_size - 1))

        if tokens_b:
            for (n, token) in enumerate(tokens_b, len(tokens_a) + 4):
                word_ids[n] = token.id
            word_ids[n + 1] = tokenizer.vocab['[SEP]']

            for (n, (pos, entity_id, link_prob, prior_prob)) in enumerate(mentions_b, len(mentions_a)):
                entity_ids[n] = entity_id
                entity_position_ids[n] = pos + len(tokens_a) + 4
                entity_link_prob_ids[n] = int(link_prob * (link_prob_bin_size - 1))
                entity_prior_prob_ids[n] = int(prior_prob * (prior_prob_bin_size - 1))

        word_attention_mask = np.ones(max_seq_length, dtype=np.int)
        word_attention_mask[len(tokens_a) + len(tokens_b) + 5:] = 0
        word_segment_ids = np.zeros(max_seq_length, dtype=np.int)
        word_segment_ids[len(tokens_a) + 4:len(tokens_a) + len(tokens_b) + 5] = 1

        entity_attention_mask = np.ones(max_entity_length, dtype=np.int)
        entity_attention_mask[len(mentions_a) + len(mentions_b):] = 0
        entity_segment_ids = np.zeros(max_entity_length, dtype=np.int)
        entity_segment_ids[len(mentions_a):len(mentions_a) + len(mentions_b)] = 1

        if task_type == 'classification':
            label = label_map[example.label]
        else:
            label = example.label

        feat = InputFeatures(word_ids=word_ids,
                             word_attention_mask=word_attention_mask,
                             word_segment_ids=word_segment_ids,
                             entity_ids=entity_ids,
                             entity_attention_mask=entity_attention_mask,
                             entity_position_ids=entity_position_ids,
                             entity_segment_ids=entity_segment_ids,
                             entity_link_prob_ids=entity_link_prob_ids,
                             entity_prior_prob_ids=entity_prior_prob_ids,
                             label=label)

        if ex_index < 2:
            logger.info("*** Example ***")
            logger.info("guid: %s", (example.guid))
            words = ['[CLS]', '[ENT]', '[ENT]'] + [t.text for t in tokens_a] + ['[SEP]'] +\
                [t.text for t in tokens_b] + ['[SEP]']
            logger.info("words: %s", " ".join(
                [str(n) + ':' + str(x) for (n, x) in enumerate(words)]))
            logger.info("word_ids: %s", " ".join(
                [str(n) + ':' + str(x) for (n, x) in enumerate(word_ids)]))
            logger.info("word_attention_mask: %s", " ".join(
                [str(n) + ':' + str(x) for (n, x) in enumerate(word_attention_mask)]))
            logger.info("word_segment_ids: %s", " ".join(
                [str(n) + ':' + str(x) for (n, x) in enumerate(word_segment_ids)]))
            logger.info("entity_ids: %s", " ".join(
                [str(n) + ':' + str(x) for (n, x) in enumerate(entity_ids)]))
            logger.info("entity_attention_mask: %s", " ".join(
                [str(n) + ':' + str(x) for (n, x) in enumerate(entity_attention_mask)]))
            logger.info("entity_position_ids: %s", " ".join(
                [str(n) + ':' + str(x) for (n, x) in enumerate(entity_position_ids)]))
            logger.info("entity_segment_ids: %s", " ".join(
                [str(n) + ':' + str(x) for (n, x) in enumerate(entity_segment_ids)]))
            logger.info("entity_link_prob_ids: %s", " ".join(
                [str(n) + ':' + str(x) for (n, x) in enumerate(entity_link_prob_ids)]))
            logger.info("entity_prior_prob_ids: %s", " ".join(
                [str(n) + ':' + str(x) for (n, x) in enumerate(entity_prior_prob_ids)]))
            logger.info("label: %s (id = %d)", example.label, label)

        features.append(feat)

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


def run(tokenizer, entity_linker, data_dir, task_name, model_file, output_dir, max_seq_length,
        max_entity_length, min_prior_prob, batch_size, eval_batch_size, learning_rate,
        iteration, warmup_proportion, lr_decay, seed, gradient_accumulation_steps, fix_entity_emb):
    processors = {
        "cola": ColaProcessor,
        "mnli": MnliProcessor,
        "qnli": QnliProcessor,
        "mrpc": MrpcProcessor,
        "qqp": QqpProcessor,
        "scitail": SciTailProcessor,
        "rte": RTEProcessor,
        "sts-b": STSProcessor,
    }

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

    data_file = model_file.replace('.bin', '.pkl').replace('model', 'data')
    model_data = joblib.load(data_file)

    train_batch_size = int(batch_size / gradient_accumulation_steps)

    model_type = model_data['model_type']
    model_module = importlib.import_module('luke.' + model_type)
    LukeConfig = getattr(model_module, 'LukeConfig')
    LayerNorm = getattr(model_module, 'LayerNorm')

    bert_model = BertForPreTraining.from_pretrained(model_data['bert_model_name'])

    config = LukeConfig(entity_vocab_size=model_data['entity_vocab'].size,
                        entity_emb_size=model_data['entity_emb_size'],
                        link_prob_bin_size=model_data['link_prob_bin_size'],
                        prior_prob_bin_size=model_data['prior_prob_bin_size'],
                        **bert_model.config.to_dict())
    del bert_model

    processor = processors[task_name]()
    if processor.task_type == 'classification':
        label_list = processor.get_labels()
        TaskClass = getattr(model_module, 'LukeForSequenceClassification')
        model = TaskClass(config, num_labels=len(label_list))
        label_dtype = torch.long
    else:  # regression
        label_list = None  # scoring task
        TaskClass = getattr(model_module, 'LukeForSequenceRegression')
        model = TaskClass(config)
        label_dtype = torch.float

    train_examples = None
    num_train_steps = None
    train_examples = processor.get_train_examples(data_dir)
    num_train_steps = int(len(train_examples) / train_batch_size * iteration)

    # Prepare model
    state_dict = torch.load(model_file, map_location='cpu')
    model.load_state_dict(state_dict, strict=False)

    if fix_entity_emb:
        try:
            model.entity_embeddings.entity_embeddings.weight.requires_grad = False
        except AttributeError:
            pass

    # if fp16:
    #     model.half()
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
    # opt_device = torch.device('cuda:' + str(n_gpu - 1))
    optimizer = BertAdam([parameters, no_decay_parameters], lr=learning_rate, device=opt_device,
                        warmup=warmup_proportion, lr_decay=lr_decay, t_total=num_train_steps)
    # if fp16:
    #     # from apex.optimizers import FP16_Optimizer
    #     # from apex.optimizers import FusedAdam
    #     from apex.fp16_utils import FP16_Optimizer
    #     # optimizer = FusedAdam([parameters, no_decay_parameters], lr=learning_rate,
    #     #                       bias_correction=False, max_grad_norm=1.0)
    #     optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)

    global_step = 0

    train_features = convert_examples_to_features(
        train_examples, processor.task_type, label_list, tokenizer, entity_linker,
        model_data['entity_vocab'], max_seq_length, max_entity_length, min_prior_prob,
        model_data['link_prob_bin_size'], model_data['prior_prob_bin_size'])

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_examples))
    logger.info("  Batch size = %d", batch_size)
    logger.info("  Learning rate = %f", learning_rate)
    logger.info("  Minimum Prior Probability = %f", min_prior_prob)
    logger.info("  Iteration = %d", iteration)
    train_data = TensorDataset(
        torch.tensor([f.word_ids for f in train_features], dtype=torch.long),
        torch.tensor([f.word_segment_ids for f in train_features],
                     dtype=torch.long),
        torch.tensor([f.word_attention_mask for f in train_features], dtype=torch.long),
        torch.tensor([f.entity_ids for f in train_features], dtype=torch.long),
        torch.tensor([f.entity_position_ids for f in train_features], dtype=torch.long),
        torch.tensor([f.entity_segment_ids for f in train_features], dtype=torch.long),
        torch.tensor([f.entity_attention_mask for f in train_features], dtype=torch.long),
        torch.tensor([f.entity_link_prob_ids for f in train_features], dtype=torch.long),
        torch.tensor([f.entity_prior_prob_ids for f in train_features], dtype=torch.long),
        torch.tensor([f.label for f in train_features], dtype=label_dtype),
    )

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(
        train_data, sampler=train_sampler, batch_size=train_batch_size)

    eval_examples = processor.get_dev_examples(data_dir)
    eval_features = convert_examples_to_features(
        eval_examples, processor.task_type, label_list, tokenizer, entity_linker,
        model_data['entity_vocab'], max_seq_length, max_entity_length, min_prior_prob,
        model_data['link_prob_bin_size'], model_data['prior_prob_bin_size'])

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
        torch.tensor([f.label for f in eval_features], dtype=label_dtype),
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
             labels) = batch
            loss = model(word_ids=word_ids,
                         word_segment_ids=word_segment_ids,
                         word_attention_mask=word_attention_mask,
                         entity_ids=entity_ids,
                         entity_position_ids=entity_position_ids,
                         entity_segment_ids=entity_segment_ids,
                         entity_attention_mask=entity_attention_mask,
                         entity_link_prob_ids=entity_link_prob_ids,
                         entity_prior_prob_ids=entity_prior_prob_ids,
                         labels=labels)
            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps
            loss.backward()
            # if fp16:
            #     optimizer.backward(loss)
            # else:
                # loss.backward()
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
             labels) = batch
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
                                      labels=labels)
                logits = model(word_ids=word_ids,
                               word_segment_ids=word_segment_ids,
                               word_attention_mask=word_attention_mask,
                               entity_ids=entity_ids,
                               entity_position_ids=entity_position_ids,
                               entity_segment_ids=entity_segment_ids,
                               entity_attention_mask=entity_attention_mask,
                               entity_link_prob_ids=entity_link_prob_ids,
                               entity_prior_prob_ids=entity_prior_prob_ids)

            eval_logits.append(logits.detach().cpu().numpy())
            eval_labels.append(labels.cpu().numpy())

            eval_loss += tmp_eval_loss.mean().item()

            nb_eval_examples += batch[0].size(0)
            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps

        result = dict(
            task_name=task_name,
            eval_loss=eval_loss,
            global_step=global_step,
            loss=tr_loss / nb_tr_steps,
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

        if processor.task_type == 'classification':
            outputs = np.argmax(np.vstack(eval_logits), axis=1)
            result['eval_accuracy'] = np.sum(outputs == np.concatenate(eval_labels)) / nb_eval_examples
        elif processor.task_type == 'regression':
            outputs = np.vstack(eval_logits).flatten()
            result['eval_pearson'] = float(pearsonr(outputs, np.concatenate(eval_labels))[0])
            result['eval_spearman'] = float(spearmanr(outputs, np.concatenate(eval_labels))[0])

        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            if key.startswith('eval_') or key in ('loss', 'global_step', 'iteration'):
                logger.info("  %s = %s", key, str(result[key]))
        if output_dir:
            writer.write("%s\n" % json.dumps(result, sort_keys=True))
            writer.flush()

    if output_dir:
        writer.close()
