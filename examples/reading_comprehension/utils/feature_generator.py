import multiprocessing
from argparse import Namespace
from contextlib import closing
from multiprocessing.pool import Pool
from tqdm import tqdm

from ...utils.text_encoder import TextEncoder


class InputFeatures(object):
    def __init__(self, unique_id, example_index, doc_span_index, tokens, mentions, token_to_orig_map,
                 token_is_max_context, word_ids, word_segment_ids, word_attention_mask, entity_candidate_ids,
                 entity_position_ids, entity_segment_ids, entity_attention_mask, start_positions, end_positions):
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
        self.start_positions = start_positions
        self.end_positions = end_positions


def convert_examples_to_features(
        examples, tokenizer, entity_vocab, mention_db, max_seq_length, max_mention_length, max_candidate_length,
        doc_stride, max_query_length, segment_b_id, add_extra_sep_token, is_training,
        pool_size=multiprocessing.cpu_count(), chunk_size=30):
    text_encoder = TextEncoder(tokenizer, entity_vocab, mention_db, max_mention_length, max_candidate_length,
                               add_extra_sep_token, segment_b_id)

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

    tok_start_positions = []
    tok_end_positions = []
    if params.is_training and not example.is_impossible:
        for start, end, answer_text in zip(example.start_positions, example.end_positions, example.answer_texts):
            tok_start = orig_to_tok_index[start]
            if end < len(example.doc_tokens) - 1:
                tok_end = orig_to_tok_index[end + 1] - 1
            else:
                tok_end = len(all_doc_tokens) - 1
            tok_start, tok_end = _improve_answer_span(all_doc_tokens, tok_start, tok_end, tokenizer, answer_text)
            tok_start_positions.append(tok_start)
            tok_end_positions.append(tok_end)

    max_tokens_for_doc = params.max_seq_length - len(query_tokens) - 3
    if params.add_extra_sep_token:
        max_tokens_for_doc -= 1

    doc_spans = []
    start_offset = 0
    while start_offset < len(all_doc_tokens):
        length = len(all_doc_tokens) - start_offset
        if length > max_tokens_for_doc:
            length = max_tokens_for_doc
        doc_spans.append(dict(start=start_offset, length=length))
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

        for i in range(doc_span['length']):
            split_token_index = doc_span['start'] + i
            token_to_orig_map[answer_offset + i] = tok_to_orig_index[split_token_index]

            is_max_context = _check_is_max_context(doc_spans, doc_span_index, split_token_index)
            token_is_max_context[answer_offset + i] = is_max_context
            answer_tokens.append(all_doc_tokens[split_token_index])

        start_positions = []
        end_positions = []

        if params.is_training:
            if example.is_impossible:
                start_positions = [0]
                end_positions = [0]
            else:
                doc_start = doc_span['start']
                doc_end = doc_span['start'] + doc_span['length'] - 1
                for tok_start, tok_end in zip(tok_start_positions, tok_end_positions):
                    if not (tok_start >= doc_start and tok_end <= doc_end):
                        continue
                    doc_offset = len(query_tokens) + 2
                    if params.add_extra_sep_token:
                        doc_offset += 1
                    start_positions.append(tok_start - doc_start + doc_offset)
                    end_positions.append(tok_end - doc_start + doc_offset)

                if not start_positions:
                    # continue
                    start_positions = [0]
                    end_positions = [0]

        features.append(InputFeatures(
            unique_id=None,
            example_index=example_index,
            doc_span_index=doc_span_index,
            token_to_orig_map=token_to_orig_map,
            token_is_max_context=token_is_max_context,
            start_positions=start_positions,
            end_positions=end_positions,
            **params.text_encoder.encode_text_pair(query_tokens, answer_tokens)
        ))

    return features


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer, orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer.
       Original version was obtained from here:
       https://github.com/huggingface/transformers/blob/23c6998bf46e43092fc59543ea7795074a720f08/src/transformers/data/processors/squad.py#L25
    """
    tok_answer_text = tokenizer.convert_tokens_to_string(tokenizer.tokenize(orig_answer_text)).strip()

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = tokenizer.convert_tokens_to_string(doc_tokens[new_start:(new_end + 1)]).strip()
            if text_span == tok_answer_text:
                return new_start, new_end

    return input_start, input_end


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token.
       Original version was obtained from here:
       https://github.com/huggingface/transformers/blob/23c6998bf46e43092fc59543ea7795074a720f08/src/transformers/data/processors/squad.py#L38
    """
    best_score = None
    best_span_index = None
    for span_index, doc_span in enumerate(doc_spans):
        end = doc_span['start'] + doc_span['length'] - 1
        if position < doc_span['start']:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span['start']
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span['length']
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index
