import itertools
import math
import os
import unicodedata
from transformers import RobertaTokenizer


class InputExample(object):
    def __init__(self, guid, words, labels, sentence_boundaries):
        self.guid = guid
        self.words = words
        self.labels = labels
        self.sentence_boundaries = sentence_boundaries


class InputFeatures(object):
    def __init__(
        self,
        example_index,
        word_ids,
        word_segment_ids,
        word_attention_mask,
        entity_start_positions,
        entity_end_positions,
        entity_ids,
        entity_position_ids,
        entity_segment_ids,
        entity_attention_mask,
        original_entity_spans,
        labels,
    ):
        self.example_index = example_index
        self.word_ids = word_ids
        self.word_segment_ids = word_segment_ids
        self.word_attention_mask = word_attention_mask
        self.entity_start_positions = entity_start_positions
        self.entity_end_positions = entity_end_positions
        self.entity_ids = entity_ids
        self.entity_position_ids = entity_position_ids
        self.entity_segment_ids = entity_segment_ids
        self.entity_attention_mask = entity_attention_mask
        self.original_entity_spans = original_entity_spans
        self.labels = labels


class CoNLLProcessor(object):
    def get_train_examples(self, data_dir):
        return list(self._create_examples(self._read_data(os.path.join(data_dir, "eng.train")), "train"))

    def get_dev_examples(self, data_dir):
        return list(self._create_examples(self._read_data(os.path.join(data_dir, "eng.testa")), "dev"))

    def get_test_examples(self, data_dir):
        return list(self._create_examples(self._read_data(os.path.join(data_dir, "eng.testb")), "test"))

    def get_labels(self):
        return ["NIL", "MISC", "PER", "ORG", "LOC"]

    def _read_data(self, input_file):
        data = []
        words = []
        labels = []
        sentence_boundaries = []
        with open(input_file) as f:
            for line in f:
                line = line.rstrip()
                if line.startswith("-DOCSTART"):
                    if words:
                        data.append((words, labels, sentence_boundaries))
                        assert sentence_boundaries[0] == 0
                        assert sentence_boundaries[-1] == len(words)
                        words = []
                        labels = []
                        sentence_boundaries = []
                    continue

                if not line:
                    if not sentence_boundaries or len(words) != sentence_boundaries[-1]:
                        sentence_boundaries.append(len(words))
                else:
                    parts = line.split(" ")
                    words.append(parts[0])
                    labels.append(parts[-1])

        if words:
            data.append((words, labels, sentence_boundaries))

        return data

    def _create_examples(self, data, fold):
        return [InputExample(f"{fold}-{i}", *args) for i, args in enumerate(data)]


def convert_examples_to_features(
    examples, label_list, tokenizer, max_seq_length, max_entity_length, max_mention_length
):
    max_num_subwords = max_seq_length - 2
    label_map = {label: i for i, label in enumerate(label_list)}
    features = []

    def tokenize_word(text):
        if (
            isinstance(tokenizer, RobertaTokenizer)
            and (text[0] != "'")
            and (len(text) != 1 or not is_punctuation(text))
        ):
            return tokenizer.tokenize(text, add_prefix_space=True)
        return tokenizer.tokenize(text)

    for example_index, example in enumerate(examples):
        tokens = [tokenize_word(w) for w in example.words]
        subwords = [w for li in tokens for w in li]

        subword2token = list(itertools.chain(*[[i] * len(li) for i, li in enumerate(tokens)]))
        token2subword = [0] + list(itertools.accumulate(len(li) for li in tokens))
        subword_start_positions = frozenset(token2subword)
        subword_sentence_boundaries = [sum(len(li) for li in tokens[:p]) for p in example.sentence_boundaries]

        entity_labels = {}
        start = None
        cur_type = None
        for n, label in enumerate(example.labels):
            if label == "O" or n in example.sentence_boundaries:
                if start is not None:
                    entity_labels[(token2subword[start], token2subword[n])] = label_map[cur_type]
                    start = None
                    cur_type = None

            if label.startswith("B"):
                if start is not None:
                    entity_labels[(token2subword[start], token2subword[n])] = label_map[cur_type]
                start = n
                cur_type = label[2:]

            elif label.startswith("I"):
                if start is None:
                    start = n
                    cur_type = label[2:]
                elif cur_type != label[2:]:
                    entity_labels[(token2subword[start], token2subword[n])] = label_map[cur_type]
                    start = n
                    cur_type = label[2:]

        if start is not None:
            entity_labels[(token2subword[start], len(subwords))] = label_map[cur_type]

        for n in range(len(subword_sentence_boundaries) - 1):
            doc_sent_start, doc_sent_end = subword_sentence_boundaries[n : n + 2]

            left_length = doc_sent_start
            right_length = len(subwords) - doc_sent_end
            sentence_length = doc_sent_end - doc_sent_start
            half_context_length = int((max_num_subwords - sentence_length) / 2)

            if left_length < right_length:
                left_context_length = min(left_length, half_context_length)
                right_context_length = min(right_length, max_num_subwords - left_context_length - sentence_length)
            else:
                right_context_length = min(right_length, half_context_length)
                left_context_length = min(left_length, max_num_subwords - right_context_length - sentence_length)

            doc_offset = doc_sent_start - left_context_length
            target_tokens = subwords[doc_offset : doc_sent_end + right_context_length]

            word_ids = tokenizer.convert_tokens_to_ids([tokenizer.cls_token] + target_tokens + [tokenizer.sep_token])
            word_attention_mask = [1] * (len(target_tokens) + 2)
            word_segment_ids = [0] * (len(target_tokens) + 2)

            entity_start_positions = []
            entity_end_positions = []
            entity_ids = []
            entity_attention_mask = []
            entity_segment_ids = []
            entity_position_ids = []
            original_entity_spans = []
            labels = []

            for entity_start in range(left_context_length, left_context_length + sentence_length):
                doc_entity_start = entity_start + doc_offset
                if doc_entity_start not in subword_start_positions:
                    continue
                for entity_end in range(entity_start + 1, left_context_length + sentence_length + 1):
                    doc_entity_end = entity_end + doc_offset
                    if doc_entity_end not in subword_start_positions:
                        continue

                    if entity_end - entity_start > max_mention_length:
                        continue

                    entity_start_positions.append(entity_start + 1)
                    entity_end_positions.append(entity_end)
                    entity_ids.append(1)
                    entity_attention_mask.append(1)
                    entity_segment_ids.append(0)

                    position_ids = list(range(entity_start + 1, entity_end + 1))
                    position_ids += [-1] * (max_mention_length - entity_end + entity_start)
                    entity_position_ids.append(position_ids)

                    original_entity_spans.append(
                        (subword2token[doc_entity_start], subword2token[doc_entity_end - 1] + 1)
                    )

                    labels.append(entity_labels.get((doc_entity_start, doc_entity_end), 0))
                    entity_labels.pop((doc_entity_start, doc_entity_end), None)

            if len(entity_ids) == 1:
                entity_start_positions.append(0)
                entity_end_positions.append(0)
                entity_ids.append(0)
                entity_attention_mask.append(0)
                entity_segment_ids.append(0)
                entity_position_ids.append(([-1] * max_mention_length))
                original_entity_spans.append(None)
                labels.append(-1)

            split_size = math.ceil(len(entity_ids) / max_entity_length)
            for i in range(split_size):
                entity_size = math.ceil(len(entity_ids) / split_size)
                start = i * entity_size
                end = start + entity_size
                features.append(
                    InputFeatures(
                        example_index=example_index,
                        word_ids=word_ids,
                        word_attention_mask=word_attention_mask,
                        word_segment_ids=word_segment_ids,
                        entity_start_positions=entity_start_positions[start:end],
                        entity_end_positions=entity_end_positions[start:end],
                        entity_ids=entity_ids[start:end],
                        entity_position_ids=entity_position_ids[start:end],
                        entity_segment_ids=entity_segment_ids[start:end],
                        entity_attention_mask=entity_attention_mask[start:end],
                        original_entity_spans=original_entity_spans[start:end],
                        labels=labels[start:end],
                    )
                )

        assert not entity_labels

    return features


def is_punctuation(char):
    # obtained from:
    # https://github.com/huggingface/transformers/blob/5f25a5f367497278bf19c9994569db43f96d5278/transformers/tokenization_bert.py#L489
    cp = ord(char)
    if (cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False
