from argparse import Namespace
from collections import namedtuple
from contextlib import closing
from functools import partial
import multiprocessing
from multiprocessing.pool import Pool
from transformers.tokenization_roberta import RobertaTokenizer
from tqdm import tqdm

from ..utils.text_encoder import LukeTextEncoder
from .utils_squad import _improve_answer_span, _check_is_max_context


class InputFeatures(object):
    def __init__(self, unique_id, example_index, doc_span_index, tokens, mentions, token_to_orig_map,
                 token_is_max_context, word_ids, word_segment_ids, word_attention_mask, entity_candidate_ids,
                 entity_position_ids, entity_segment_ids, entity_attention_mask, start_position=None, end_position=None,
                 is_impossible=None):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.mentions = mentions
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.word_ids = word_ids
        self.word_segment_ids = word_segment_ids
        self.word_attention_mask = word_attention_mask
        self.entity_candidate_ids = entity_candidate_ids
        self.entity_position_ids = entity_position_ids
        self.entity_segment_ids = entity_segment_ids
        self.entity_attention_mask = entity_attention_mask
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible


def convert_examples_to_features(
        examples, tokenizer, entity_vocab, mention_db, max_seq_length, max_mention_length, max_candidate_length,
        doc_stride, max_query_length, add_extra_sep_token, is_training, pool_size=multiprocessing.cpu_count(),
        chunk_size=30):
    if isinstance(tokenizer, RobertaTokenizer):
        tokenizer.tokenize = partial(tokenizer.tokenize, add_prefix_space=True)

    text_encoder = LukeTextEncoder(tokenizer, entity_vocab, mention_db, max_mention_length, max_candidate_length,
                                   add_extra_sep_token)

    worker_params = Namespace(
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        doc_stride=doc_stride,
        max_query_length=max_query_length,
        add_extra_sep_token=add_extra_sep_token,
        text_encoder=text_encoder,
        is_training=is_training,
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

    tokenizer = params.tokenizer
    query_tokens = tokenizer.tokenize(example.question_text)

    if len(query_tokens) > params.max_query_length:
        query_tokens = query_tokens[0:params.max_query_length]

    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
    for i, token in enumerate(example.doc_tokens):
        orig_to_tok_index.append(len(all_doc_tokens))
        sub_tokens = tokenizer.tokenize(token)
        for sub_token in sub_tokens:
            tok_to_orig_index.append(i)
            all_doc_tokens.append(sub_token)

    tok_start_position = None
    tok_end_position = None
    if params.is_training:
        if example.is_impossible:
            tok_start_position = -1
            tok_end_position = -1
        else:
            tok_start_position = orig_to_tok_index[example.start_position]
            if example.end_position < len(example.doc_tokens) - 1:
                tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1
            tok_start_position, tok_end_position = _improve_answer_span(
                all_doc_tokens, tok_start_position, tok_end_position, tokenizer, example.orig_answer_text)

    max_tokens_for_doc = params.max_seq_length - len(query_tokens) - 3
    if params.add_extra_sep_token:
        max_tokens_for_doc -= 1

    _DocSpan = namedtuple("DocSpan", ["start", "length"])
    doc_spans = []
    start_offset = 0
    while start_offset < len(all_doc_tokens):
        length = len(all_doc_tokens) - start_offset
        if length > max_tokens_for_doc:
            length = max_tokens_for_doc
        doc_spans.append(_DocSpan(start=start_offset, length=length))
        if start_offset + length == len(all_doc_tokens):
            break
        start_offset += min(length, params.doc_stride)

    features = []

    for doc_span_index, doc_span in enumerate(doc_spans):
        token_to_orig_map = {}
        token_is_max_context = {}

        answer_tokens = []
        answer_offset = len(query_tokens) + 2
        if params.add_extra_sep_token:
            answer_offset += 1

        for i in range(doc_span.length):
            split_token_index = doc_span.start + i
            token_to_orig_map[answer_offset + i] = tok_to_orig_index[split_token_index]

            is_max_context = _check_is_max_context(doc_spans, doc_span_index, split_token_index)
            token_is_max_context[answer_offset + i] = is_max_context
            answer_tokens.append(all_doc_tokens[split_token_index])

        span_is_impossible = example.is_impossible
        start_position = None
        end_position = None
        if params.is_training:
            if span_is_impossible:
                start_position = 0
                end_position = 0
            else:
                # For training, if our document chunk does not contain an annotation
                # we throw it out, since there is nothing to predict.
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1
                out_of_span = False
                if not (tok_start_position >= doc_start and tok_end_position <= doc_end):
                    out_of_span = True
                if out_of_span:
                    start_position = 0
                    end_position = 0
                    span_is_impossible = True
                else:
                    doc_offset = len(query_tokens) + 2
                    if params.add_extra_sep_token:
                        doc_offset += 1
                    start_position = tok_start_position - doc_start + doc_offset
                    end_position = tok_end_position - doc_start + doc_offset

        features.append(InputFeatures(
            unique_id=None,
            example_index=example_index,
            doc_span_index=doc_span_index,
            token_to_orig_map=token_to_orig_map,
            token_is_max_context=token_is_max_context,
            start_position=start_position,
            end_position=end_position,
            is_impossible=span_is_impossible,
            **params.text_encoder.encode_text_pair(query_tokens, answer_tokens)
        ))

    return features
