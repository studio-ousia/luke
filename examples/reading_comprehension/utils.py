import json
import logging
import math
import collections
from argparse import Namespace
from collections import namedtuple
from contextlib import closing
import multiprocessing
from multiprocessing.pool import Pool
from tqdm import tqdm

from transformers.tokenization_bert import BasicTokenizer

from ..utils.text_encoder import LukeTextEncoder

logger = logging.getLogger(__name__)


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
                all_doc_tokens, tok_start_position, tok_end_position, tokenizer, example.answer_text)

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


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer, orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer.
       Original version was obtained from here:
       https://github.com/huggingface/transformers/blob/23c6998bf46e43092fc59543ea7795074a720f08/src/transformers/data/processors/squad.py#L25
    """
    tok_answer_text = tokenizer.convert_tokens_to_string(tokenizer.tokenize(orig_answer_text)).strip()
    # tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = tokenizer.convert_tokens_to_string(doc_tokens[new_start:(new_end + 1)]).strip()
            # text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token.
       Original version was obtained from here:
       https://github.com/huggingface/transformers/blob/23c6998bf46e43092fc59543ea7795074a720f08/src/transformers/data/processors/squad.py#L38
    """
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


def write_predictions(all_examples, all_features, all_results, n_best_size, max_answer_length, do_lower_case,
                      output_prediction_file, output_nbest_file, output_null_log_odds_file, verbose_logging,
                      version_2_with_negative, null_score_diff_threshold, tokenizer):
    logger.info("Writing predictions to: %s" % (output_prediction_file))
    logger.info("Writing nbest to: %s" % (output_nbest_file))

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(
        "PrelimPrediction", ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()

    for example_index, example in enumerate(all_examples):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        score_null = 1000000  # large and positive
        min_null_feature_index = 0  # the paragraph slice with min null score
        null_start_logit = 0  # the start logit at the slice with min null score
        null_end_logit = 0  # the end logit at the slice with min null score
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]
            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)
            # if we could have irrelevant answers, get the min score of irrelevant
            if version_2_with_negative:
                feature_null_score = result.start_logits[0] + result.end_logits[0]
                if feature_null_score < score_null:
                    score_null = feature_null_score
                    min_null_feature_index = feature_index
                    null_start_logit = result.start_logits[0]
                    null_end_logit = result.end_logits[0]
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index]))
        if version_2_with_negative:
            prelim_predictions.append(
                _PrelimPrediction(
                    feature_index=min_null_feature_index,
                    start_index=0,
                    end_index=0,
                    start_logit=null_start_logit,
                    end_logit=null_end_logit))
        prelim_predictions = sorted(prelim_predictions, key=lambda x: (x.start_logit + x.end_logit), reverse=True)

        _NbestPrediction = collections.namedtuple("NbestPrediction", ["text", "start_logit", "end_logit"])

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]
            if pred.start_index > 0:  # this is a non-null prediction
                tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]

                tok_text = tokenizer.convert_tokens_to_string(tok_tokens)

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = " ".join(orig_tokens)

                final_text = get_final_text(tok_text, orig_text, do_lower_case, verbose_logging)
                if final_text in seen_predictions:
                    continue

                seen_predictions[final_text] = True
            else:
                final_text = ""
                seen_predictions[final_text] = True

            nbest.append(_NbestPrediction(text=final_text, start_logit=pred.start_logit, end_logit=pred.end_logit))
        # if we didn't include the empty option in the n-best, include it
        if version_2_with_negative:
            if "" not in seen_predictions:
                nbest.append(_NbestPrediction(text="", start_logit=null_start_logit, end_logit=null_end_logit))

            # In very rare edge cases we could only have single null prediction.
            # So we just create a nonce prediction in this case to avoid failure.
            if len(nbest) == 1:
                nbest.insert(0, _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(_NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

        assert len(nbest) >= 1

        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)
            if not best_non_null_entry:
                if entry.text:
                    best_non_null_entry = entry

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            nbest_json.append(output)

        assert len(nbest_json) >= 1

        if not version_2_with_negative:
            all_predictions[example.qas_id] = nbest_json[0]["text"]
        else:
            # predict "" iff the null score - the score of best non-null > threshold
            score_diff = score_null - best_non_null_entry.start_logit -  best_non_null_entry.end_logit
            scores_diff_json[example.qas_id] = score_diff
            if score_diff > null_score_diff_threshold:
                all_predictions[example.qas_id] = ""
            else:
                all_predictions[example.qas_id] = best_non_null_entry.text
        all_nbest_json[example.qas_id] = nbest_json

    with open(output_prediction_file, "w") as writer:
        writer.write(json.dumps(all_predictions, indent=4) + "\n")

    with open(output_nbest_file, "w") as writer:
        writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

    if version_2_with_negative:
        with open(output_null_log_odds_file, "w") as writer:
            writer.write(json.dumps(scores_diff_json, indent=4) + "\n")

    return all_predictions


def get_final_text(pred_text, orig_text, do_lower_case, verbose_logging=False):
    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for i, c in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return ns_text, ns_to_s_map

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.
    tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        if verbose_logging:
            logger.info("Unable to find text: '%s' in '%s'", pred_text, orig_text)
        return orig_text
    end_position = start_position + len(pred_text) - 1

    orig_ns_text, orig_ns_to_s_map = _strip_spaces(orig_text)
    tok_ns_text, tok_ns_to_s_map = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if verbose_logging:
            logger.info("Length not equal after stripping spaces: '%s' vs '%s'", orig_ns_text, tok_ns_text)
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for i, tok_index in tok_ns_to_s_map.items():
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        if verbose_logging:
            logger.info("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if verbose_logging:
            logger.info("Couldn't map end position")
        return orig_text

    return orig_text[orig_start_position:(orig_end_position + 1)]


def _get_best_indexes(logits, n_best_size):
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def _compute_softmax(scores):
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs
