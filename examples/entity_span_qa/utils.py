import json
import logging
import multiprocessing
import os
from argparse import Namespace
from contextlib import closing
from multiprocessing.pool import Pool

from tqdm import tqdm
from transformers.tokenization_roberta import RobertaTokenizer

logger = logging.getLogger(__name__)

PLACEHOLDER_TOKEN = "[PLACEHOLDER]"
HIGHLIGHT_TOKEN = "[HIGHLIGHT]"
ENTITY_MARKER_TOKEN = "[ENTITY]"


class InputExample(object):
    def __init__(self, qas_id, question_text, context_text, answers, entities):
        self.qas_id = qas_id
        self.question_text = question_text
        self.context_text = context_text
        self.answers = answers
        self.entities = entities


class RecordProcessor(object):
    train_file = "train.json"
    dev_file = "dev.json"

    def get_train_examples(self, data_dir):
        with open(os.path.join(data_dir, self.train_file)) as reader:
            input_data = json.load(reader)["data"]
        return self._create_examples(input_data)

    def get_dev_examples(self, data_dir):
        with open(os.path.join(data_dir, self.dev_file)) as reader:
            input_data = json.load(reader)["data"]
        return self._create_examples(input_data)

    def _create_examples(self, input_data):
        examples = []
        for entry in input_data:
            context_text = entry["passage"]["text"]

            entities = entry["passage"]["entities"]
            for entity in entities:
                entity["end"] += 1
                entity["text"] = context_text[entity["start"] : entity["end"]]

            for qa in entry["qas"]:
                qas_id = qa["id"]
                question_text = qa["query"]
                answers = qa.get("answers", [])
                for answer in answers:
                    answer["end"] += 1

                example = InputExample(
                    qas_id=qas_id,
                    question_text=question_text,
                    context_text=context_text,
                    answers=answers,
                    entities=entities,
                )
                examples.append(example)

        return examples


class InputFeatures(object):
    def __init__(
        self,
        unique_id,
        entities,
        example_index,
        doc_span_index,
        word_ids,
        word_segment_ids,
        word_attention_mask,
        placeholder_position_ids,
        entity_position_ids,
        labels,
    ):
        self.unique_id = unique_id
        self.entities = entities
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.word_ids = word_ids
        self.word_segment_ids = word_segment_ids
        self.word_attention_mask = word_attention_mask
        self.placeholder_position_ids = placeholder_position_ids
        self.entity_position_ids = entity_position_ids
        self.labels = labels


def convert_examples_to_features(
    examples,
    tokenizer,
    max_seq_length,
    max_mention_length,
    doc_stride,
    max_query_length,
    segment_b_id,
    add_extra_sep_token,
    pool_size=multiprocessing.cpu_count(),
    chunk_size=30,
):
    worker_params = Namespace(
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        max_mention_length=max_mention_length,
        doc_stride=doc_stride,
        max_query_length=max_query_length,
        add_extra_sep_token=add_extra_sep_token,
        segment_b_id=segment_b_id,
    )
    features = []
    unique_id = 1000000000
    with closing(Pool(pool_size, initializer=_initialize_worker, initargs=(worker_params,))) as pool:
        with tqdm(total=len(examples)) as pbar:
            for ret in pool.imap(_process_example, enumerate(examples), chunksize=chunk_size):
                for feature in ret:
                    feature.unique_id = unique_id
                    features.append(feature)
                    unique_id += 1
                pbar.update()
    return features


params = None


def _initialize_worker(_params):
    global params
    params = _params


def _process_example(args):
    example_index, example = args

    example.question_text = example.question_text.replace("\n", " ")
    example.context_text = example.context_text.replace("\n", " ")

    tokenizer = params.tokenizer

    def tokenize(text, add_prefix_space=False):
        if isinstance(tokenizer, RobertaTokenizer):
            return tokenizer.tokenize(text, add_prefix_space=add_prefix_space)
        else:
            return tokenizer.tokenize(text)

    text_a, text_b = example.question_text.split("@placeholder")
    query_tokens = tokenize(text_a, add_prefix_space=True)

    placeholder_start = len(query_tokens) + 1
    query_tokens.append(PLACEHOLDER_TOKEN)
    placeholder_end = len(query_tokens) + 1

    placeholder_position_ids = list(range(placeholder_start, placeholder_end))
    placeholder_position_ids += [-1] * (params.max_mention_length - placeholder_end + placeholder_start)
    placeholder_position_ids = [placeholder_position_ids]

    if text_b:
        if text_b[0] == " ":
            query_tokens += tokenize(text_b, add_prefix_space=True)
        else:
            query_tokens += tokenize(text_b)

    if len(query_tokens) > params.max_query_length:
        query_tokens = query_tokens[0 : params.max_query_length]

    doc_entities = sorted(example.entities, key=lambda o: o["start"])
    answer_spans = frozenset((a["start"], a["end"]) for a in example.answers)
    entity_labels = [(e["start"], e["end"]) in answer_spans for e in doc_entities]

    def preprocess_and_tokenize(context_text, start, end=None):
        text = context_text[start:end]

        tokens = []
        text_parts = text.split("@highlight")

        if start == 0 or context_text[start - 1] == " " or (text_parts[0] and text_parts[0][0] == " "):
            tokens += tokenize(text_parts[0], add_prefix_space=True)
        else:
            tokens += tokenize(text)

        for text in text_parts[1:]:
            tokens.append(HIGHLIGHT_TOKEN)

            if text[0] == " ":
                tokens += tokenize(text, add_prefix_space=True)
            else:
                tokens += tokenize(text)

        return tokens

    doc_tokens = []
    entities_with_spans = []
    cur = 0
    for entity in doc_entities:
        assert cur <= entity["start"]
        doc_tokens.extend(preprocess_and_tokenize(example.context_text, cur, entity["start"]))
        entity_start = len(doc_tokens)

        doc_tokens.append(ENTITY_MARKER_TOKEN)
        doc_tokens.extend(preprocess_and_tokenize(example.context_text, entity["start"], entity["end"]))
        doc_tokens.append(ENTITY_MARKER_TOKEN)

        entities_with_spans.append((entity_start, len(doc_tokens), entity))
        cur = entity["end"]
    doc_tokens.extend(preprocess_and_tokenize(example.context_text, cur))

    max_tokens_for_doc = params.max_seq_length - len(query_tokens) - 3
    if params.add_extra_sep_token:
        max_tokens_for_doc -= 1

    doc_spans = []
    start_offset = 0
    while start_offset < len(doc_tokens):
        length = len(doc_tokens) - start_offset
        if length > max_tokens_for_doc:
            length = max_tokens_for_doc
        doc_spans.append((start_offset, start_offset + length))
        if start_offset + length == len(doc_tokens):
            break
        start_offset += min(length, params.doc_stride)

    features = []

    for doc_span_index, (doc_start, doc_end) in enumerate(doc_spans):
        word_ids = tokenizer.convert_tokens_to_ids([tokenizer.cls_token] + query_tokens + [tokenizer.sep_token])
        word_attention_mask = [1] * (len(query_tokens) + 2)
        word_segment_ids = [0] * (len(query_tokens) + 2)

        if params.add_extra_sep_token:
            answer_tokens = [tokenizer.sep_token] + doc_tokens[doc_start:doc_end] + [tokenizer.sep_token]
            answer_offset = len(query_tokens) + 3
        else:
            answer_tokens = doc_tokens[doc_start:doc_end] + [tokenizer.sep_token]
            answer_offset = len(query_tokens) + 2

        word_ids += tokenizer.convert_tokens_to_ids(answer_tokens)
        word_attention_mask += [1] * (len(answer_tokens))
        word_segment_ids += [params.segment_b_id] * (len(answer_tokens))

        entities = []
        labels = []
        entity_position_ids = []

        for (entity_start, entity_end, entity), label in zip(entities_with_spans, entity_labels):
            if not (entity_start >= doc_start and entity_end <= doc_end):
                continue

            entities.append(entity)
            labels.append(label)

            start = entity_start - doc_start + answer_offset
            end = entity_end - doc_start + answer_offset
            position_ids = list(range(start, end))[: params.max_mention_length]
            position_ids += [-1] * (params.max_mention_length - end + start)
            entity_position_ids.append(position_ids)

        if not entity_position_ids:
            continue

        features.append(
            InputFeatures(
                unique_id=None,
                entities=entities,
                example_index=example_index,
                doc_span_index=doc_span_index,
                word_ids=word_ids,
                word_segment_ids=word_segment_ids,
                word_attention_mask=word_attention_mask,
                placeholder_position_ids=placeholder_position_ids,
                entity_position_ids=entity_position_ids,
                labels=labels,
            )
        )

    return features
