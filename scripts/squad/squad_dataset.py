import json
import collections
from luke.utils import clean_text
import numpy as np


class SquadExample(object):
    def __init__(self,
                 qas_id,
                 question_text,
                 doc_tokens,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None,
                 is_impossible=None):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 example_index,
                 doc_span_index,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 word_ids,
                 word_segment_ids,
                 word_attention_mask,
                 entity_ids,
                 entity_position_ids,
                 entity_segment_ids,
                 entity_attention_mask,
                 start_position=None,
                 end_position=None,
                 is_impossible=None):
        # Meta data
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        # For LUKE
        self.word_ids = word_ids
        self.word_segment_ids = word_segment_ids
        self.word_attention_mask = word_attention_mask
        self.entity_ids = entity_ids
        self.entity_position_ids = entity_position_ids
        self.entity_segment_ids = entity_segment_ids
        self.entity_attention_mask = entity_attention_mask
        # GT
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible


def read_squad_examples(input_file, is_training, version_2_with_negative):
    """Read a SQuAD json file into a list of SquadExample."""
    with open(input_file, "r", encoding='utf-8') as reader:
        input_data = json.load(reader)["data"]

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    examples = []
    for entry in input_data:
        for paragraph in entry["paragraphs"]:
            paragraph_text = paragraph["context"]
            doc_tokens = []
            char_to_word_offset = []
            prev_is_whitespace = True
            for c in paragraph_text:
                if is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        doc_tokens.append(c)
                    else:
                        doc_tokens[-1] += c
                    prev_is_whitespace = False
                char_to_word_offset.append(len(doc_tokens) - 1)

            for qa in paragraph["qas"]:
                qas_id = qa["id"]
                question_text = qa["question"]
                start_position = None
                end_position = None
                orig_answer_text = None
                is_impossible = False
                if is_training:
                    if version_2_with_negative:
                        is_impossible = qa["is_impossible"]
                    if (len(qa["answers"]) != 1) and (not is_impossible):
                        raise ValueError(
                            "For training, each question should have exactly 1 answer.")
                    if not is_impossible:
                        answer = qa["answers"][0]
                        orig_answer_text = answer["text"]
                        answer_offset = answer["answer_start"]
                        answer_length = len(orig_answer_text)
                        start_position = char_to_word_offset[answer_offset]
                        end_position = char_to_word_offset[answer_offset +
                                                           answer_length - 1]
                    else:
                        start_position = -1
                        end_position = -1
                        orig_answer_text = ""

                example = SquadExample(
                    qas_id=qas_id,
                    question_text=question_text,
                    doc_tokens=doc_tokens,
                    orig_answer_text=orig_answer_text,
                    start_position=start_position,
                    end_position=end_position,
                    is_impossible=is_impossible)
                examples.append(example)
    return examples


def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 doc_stride, max_query_length, is_training,
                                 entity_linker, entity_vocab, max_entity_length,
                                 max_mention_length, use_entities):
    """Loads a data file into a list of `InputBatch`s."""

    unique_id = 1000000000

    features = []

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
                        entities.append(
                            ((token_start, token_end), entity_vocab[mention.title]))

        return (tokens, entities)

    cls_id = tokenizer.vocab['[CLS]']
    sep_id = tokenizer.vocab['[SEP]']

    for (example_index, example) in enumerate(examples):
        query_tokens, query_entities = process_text(example.question_text)

        # truncate seq_tokens and entities if the length exceeds the max_query_length.
        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]
            query_entities = query_entities[0:max_query_length]

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        all_doc_entities = []

        for (i, token) in enumerate(example.doc_tokens):
            # TODO: investigtate whether the entity linker works on single token level.
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens, entities = process_text(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

            for entity in entities:
                all_doc_entities.append(entity)

        tok_start_position = None
        tok_end_position = None
        if is_training and example.is_impossible:
            tok_start_position = -1
            tok_end_position = -1
        if is_training and not example.is_impossible:
            tok_start_position = orig_to_tok_index[example.start_position]
            if example.end_position < len(example.doc_tokens) - 1:
                tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1
            (tok_start_position, tok_end_position) = _improve_answer_span(
                all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
                example.orig_answer_text)

        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3
        _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
            "DocSpan", ["start", "length"])
        doc_spans = []
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, doc_stride)

        for (doc_span_index, doc_span) in enumerate(doc_spans):
            tokens = []
            token_to_orig_map = {}
            token_is_max_context = {}
            word_ids = []
            word_segment_ids = []
            word_attention_mask = []
            entity_ids = []
            entity_position_ids = []
            entity_segment_ids = []
            entity_attention_mask = []

            tokens.append("[CLS]")
            word_ids.append(cls_id)
            word_segment_ids.append(0)

            # Store word information from questions.
            for token in query_tokens:
                tokens.append(token)
                word_ids.append(token.id)
                word_segment_ids.append(0)
            tokens.append("[SEP]")
            word_ids.append(sep_id)
            word_segment_ids.append(0)

            # Store entity information from questions
            entity_ids = np.zeros(max_entity_length, dtype=np.int)
            entity_position_ids = np.full(
                (max_entity_length, max_mention_length), -1, dtype=np.int)
            for (n, ((start, end), entity_id)) in enumerate(query_entities):
                entity_ids[n] = entity_id
                entity_position_ids[n][:end -
                                       start] = range(start + 1, end + 1)[:max_mention_length]

            num_entities_inside_window = 0
            # adding entities. index starts from len(query_entities)
            for (n, ((start, end), entity_id)) in enumerate(entities, len(query_entities)):
                # only add entity existing inside the window.
                if start > doc_span.start and end < max_seq_length:
                    entity_ids[n] = entity_id
                    ofs = len(query_tokens) + 2
                    entity_position_ids[n][:end - start] = range(
                        start + ofs, end + ofs)[:max_mention_length]
                    # keep track with the number of entities which are included in the sliding window.
                    num_entities_inside_window += 1

            entity_attention_mask = np.ones(max_entity_length, dtype=np.int)
            entity_attention_mask[len(
                query_entities) + num_entities_inside_window:] = 0

            entity_segment_ids = np.zeros(max_entity_length, dtype=np.int)
            entity_segment_ids[len(query_entities):len(
                query_entities) + num_entities_inside_window] = 1

            # Store token information from paragraph, dividing paragraphs.
            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[len(
                    tokens)] = tok_to_orig_index[split_token_index]

                is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                       split_token_index)
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])
                word_ids.append(all_doc_tokens[split_token_index].id)
                word_segment_ids.append(1)
            tokens.append("[SEP]")
            word_ids.append(sep_id)
            word_segment_ids.append(1)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            word_attention_mask = [1] * len(word_ids)

            # Zero-pad up to the sequence length.
            while len(word_ids) < max_seq_length:
                word_ids.append(0)
                word_attention_mask.append(0)
                word_segment_ids.append(0)

            assert len(word_ids) == max_seq_length
            assert len(word_attention_mask) == max_seq_length
            assert len(word_segment_ids) == max_seq_length
            assert len(entity_ids) == max_entity_length
            assert len(entity_segment_ids) == max_entity_length
            assert len(entity_attention_mask) == max_entity_length
            assert len(entity_position_ids) == max_entity_length

            start_position = None
            end_position = None

            if is_training and not example.is_impossible:
                # For training, if our document chunk does not contain an annotation
                # we throw it out, since there is nothing to predict.
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1
                out_of_span = False
                if not (tok_start_position >= doc_start and
                        tok_end_position <= doc_end):
                    out_of_span = True
                if out_of_span:
                    start_position = 0
                    end_position = 0
                else:
                    doc_offset = len(query_tokens) + 2
                    start_position = tok_start_position - doc_start + doc_offset
                    end_position = tok_end_position - doc_start + doc_offset
            if is_training and example.is_impossible:
                start_position = 0
                end_position = 0

            features.append(
                InputFeatures(
                    unique_id=unique_id,
                    example_index=example_index,
                    doc_span_index=doc_span_index,
                    tokens=tokens,
                    token_to_orig_map=token_to_orig_map,
                    token_is_max_context=token_is_max_context,
                    word_ids=word_ids,
                    word_segment_ids=word_segment_ids,
                    word_attention_mask=word_attention_mask,
                    entity_ids=entity_ids,
                    entity_position_ids=entity_position_ids,
                    entity_segment_ids=entity_segment_ids,
                    entity_attention_mask=entity_attention_mask,
                    start_position=start_position,
                    end_position=end_position,
                    is_impossible=example.is_impossible))
            unique_id += 1

    return features


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
    tok_answer_text = " ".join(
        [ans.text for ans in tokenizer.tokenize(orig_answer_text)])

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(
                [token.txt for token in doc_tokens[new_start:(new_end + 1)]])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
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
        score = min(num_left_context, num_right_context) + \
            0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index
