import csv
import os
import logging
import numpy as np

from luke.utils import clean_text

logger = logging.getLogger(__name__)


class InputExample(object):
    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    def __init__(self, word_ids, word_segment_ids, word_attention_mask, entity_ids,
                 entity_position_ids, entity_segment_ids, entity_attention_mask, label):
        self.word_ids = word_ids
        self.word_segment_ids = word_segment_ids
        self.word_attention_mask = word_attention_mask
        self.entity_ids = entity_ids
        self.entity_position_ids = entity_position_ids
        self.entity_segment_ids = entity_segment_ids
        self.entity_attention_mask = entity_attention_mask
        self.label = label


class DataProcessor(object):
    def get_train_examples(self, data_dir):
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        raise NotImplementedError()

    def get_labels(self):
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class MrpcProcessor(DataProcessor):
    @property
    def task_type(self):
        return 'classification'

    def get_train_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            text_b = line[4]
            label = line[0]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class QqpProcessor(DataProcessor):
    @property
    def task_type(self):
        return 'classification'

    def get_train_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            try:
                text_a = line[3]
                text_b = line[4]
                label = line[5]
            except IndexError:
                continue
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        return examples


class MnliProcessor(DataProcessor):
    @property
    def task_type(self):
        return 'classification'

    def get_train_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")),
                                     "dev_matched")

    def get_labels(self):
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[8]
            text_b = line[9]
            label = line[-1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MnliMismatchedProcessor(MnliProcessor):
    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_mismatched.tsv")), "dev_matched")


class Sst2Processor(DataProcessor):
    @property
    def task_type(self):
        return 'classification'

    def get_train_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class QnliProcessor(DataProcessor):
    @property
    def task_type(self):
        return 'classification'

    def get_train_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")),
                                     "dev_matched")

    def get_labels(self):
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[3]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class RTEProcessor(DataProcessor):
    @property
    def task_type(self):
        return 'classification'

    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev_matched")

    def get_labels(self):
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[3]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        return examples


class ColaProcessor(DataProcessor):
    @property
    def task_type(self):
        return 'classification'

    def get_train_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            label = line[1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))

        return examples


class SciTailProcessor(DataProcessor):
    @property
    def task_type(self):
        return 'classification'

    def get_train_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")),
                                     "dev_matched")

    def get_labels(self):
        return ["entails", "neutral"]

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%d" % (set_type, i)
            text_a = line[0]
            text_b = line[1]
            label = line[2]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        return examples


class StsProcessor(DataProcessor):
    @property
    def task_type(self):
        return 'regression'

    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    # # ADDED
    # def get_test_examples(self, data_dir):
    #     """See base class."""
    #     return self._create_examples(
    #         self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def _create_examples(self, lines, set_type):
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


class WnliProcessor(DataProcessor):
    @property
    def task_type(self):
        return 'regression'

    def get_train_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


def convert_examples_to_features(examples, task_type, label_list, tokenizer, entity_linker,
                                 entity_vocab, max_seq_length, max_entity_length,
                                 max_mention_length, use_entities):
    if task_type == 'classification':
        label_map = {}
        for (i, label) in enumerate(label_list):
            label_map[label] = i

    def process_text(text):
        text = clean_text(text, strip_accents=True)

        tokens = tokenizer.tokenize(text)

        token_start_map = np.full(len(text), -1)
        token_end_map = np.full(len(text), -1)
        for (ind, token) in enumerate(tokens):
            token_start_map[token.start] = ind
            token_end_map[token.end - 1] = ind

        entities = []
        if use_entities:
            for (mention_span, mentions) in entity_linker.detect_mentions(text):
                for mention in mentions:
                    token_start = token_start_map[mention_span[0]]
                    if token_start == -1:
                        continue

                    token_end = token_end_map[mention_span[1] - 1]
                    if token_end == -1:
                        continue
                    token_end += 1

                    if mention.title in entity_vocab:
                        entities.append(((token_start, token_end), entity_vocab[mention.title]))

        return (tokens, entities)

    cls_id = tokenizer.vocab['[CLS]']
    sep_id = tokenizer.vocab['[SEP]']
    features = []
    for example in examples:
        tokens_a, entities_a = process_text(example.text_a)
        tokens_b = entities_b = []
        if example.text_b:
            tokens_b, entities_b = process_text(example.text_b)
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
            _truncate_seq_pair(entities_a, entities_b, max_entity_length)
        else:
            # Account for [CLS] and [SEP] with "- 4"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[0:(max_seq_length - 2)]
            if len(entities_a) > max_entity_length:
                entities_a = entities_a[0:max_entity_length]

        word_ids = [cls_id] + [t.id for t in tokens_a] + [sep_id]
        if example.text_b:
            word_ids += [t.id for t in tokens_b] + [sep_id]

        output_word_ids = np.zeros(max_seq_length, dtype=np.int)
        output_word_ids[:len(word_ids)] = word_ids

        word_attention_mask = np.ones(max_seq_length, dtype=np.int)
        word_attention_mask[len(word_ids):] = 0
        word_segment_ids = np.zeros(max_seq_length, dtype=np.int)
        word_segment_ids[len(tokens_a) + 2:len(word_ids)] = 1

        entity_ids = np.zeros(max_entity_length, dtype=np.int)
        entity_position_ids = np.full((max_entity_length, max_mention_length), -1, dtype=np.int)
        for (n, ((start, end), entity_id)) in enumerate(entities_a):
            entity_ids[n] = entity_id
            entity_position_ids[n][:end - start] = range(start + 1, end + 1)[:max_mention_length]

        if example.text_b:
            for (n, ((start, end), entity_id)) in enumerate(entities_b, len(entities_a)):
                entity_ids[n] = entity_id
                ofs = len(tokens_a) + 2
                entity_position_ids[n][:end - start] = range(start + ofs, end + ofs)[:max_mention_length]

        entity_attention_mask = np.ones(max_entity_length, dtype=np.int)
        entity_attention_mask[len(entities_a) + len(entities_b):] = 0

        entity_segment_ids = np.zeros(max_entity_length, dtype=np.int)
        entity_segment_ids[len(entities_a):len(entities_a) + len(entities_b)] = 1

        if task_type == 'classification':
            label = label_map[example.label]
        else:
            label = example.label

        feat = InputFeatures(word_ids=output_word_ids,
                             word_attention_mask=word_attention_mask,
                             word_segment_ids=word_segment_ids,
                             entity_ids=entity_ids,
                             entity_attention_mask=entity_attention_mask,
                             entity_position_ids=entity_position_ids,
                             entity_segment_ids=entity_segment_ids,
                             label=label)

        features.append(feat)

    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()
