import unicodedata
from itertools import chain, repeat

from transformers import RobertaTokenizer

from luke.utils.entity_vocab import UNK_TOKEN


class TextEncoder(object):
    def __init__(self, tokenizer, entity_vocab, mention_db, max_mention_length, max_candidate_length,
                 add_extra_sep_token=False, segment_b_id=1):
        self._tokenizer = tokenizer
        self._entity_vocab = entity_vocab
        self._mention_db = mention_db
        self._max_mention_length = max_mention_length
        self._max_candidate_length = max_candidate_length
        self._add_extra_sep_token = add_extra_sep_token
        self._segment_b_id = segment_b_id

    def encode_text(self, text_or_tokens):
        if isinstance(text_or_tokens, str):
            tokens = self._tokenizer.tokenize(text_or_tokens)
        else:
            tokens = text_or_tokens

        mentions = self.detect_mentions(tokens)

        tokens = [self._tokenizer.cls_token] + tokens + [self._tokenizer.sep_token]
        word_ids = self._tokenizer.convert_tokens_to_ids(tokens)
        word_attention_mask = [1] * len(tokens)
        word_segment_ids = [0] * len(tokens)

        if not mentions:
            entity_segment_ids = [0]
            entity_attention_mask = [0]
            entity_candidate_ids = [[0 for y in range(self._max_candidate_length)]]
            entity_position_ids = [[-1 for y in range(self._max_mention_length)]]
        else:
            entity_segment_ids = [0] * len(mentions)
            entity_attention_mask = [1] * len(mentions)
            entity_candidate_ids = [[0 for y in range(self._max_candidate_length)] for x in range(len(mentions))]
            entity_position_ids = [[-1 for y in range(self._max_mention_length)] for x in range(len(mentions))]

            for i, (candidates, start, end) in enumerate(mentions):
                cand_ids = [self._entity_vocab[UNK_TOKEN]] +\
                    [self._entity_vocab[c] for c in candidates][:self._max_candidate_length - 1]
                entity_candidate_ids[i][:len(cand_ids)] = cand_ids
                entity_position_ids[i][:end - start] = range(start + 1, end + 1)  # 1 for CLS token

        return dict(
            tokens=tokens,
            mentions=mentions,
            word_ids=word_ids,
            word_segment_ids=word_segment_ids,
            word_attention_mask=word_attention_mask,
            entity_candidate_ids=entity_candidate_ids,
            entity_position_ids=entity_position_ids,
            entity_segment_ids=entity_segment_ids,
            entity_attention_mask=entity_attention_mask,
        )

    def encode_text_pair(self, text_or_tokens_a, text_or_tokens_b):
        if isinstance(text_or_tokens_a, str):
            tokens_a = self._tokenizer.tokenize(text_or_tokens_a)
        else:
            tokens_a = text_or_tokens_a

        if isinstance(text_or_tokens_b, str):
            tokens_b = self._tokenizer.tokenize(text_or_tokens_b)
        else:
            tokens_b = text_or_tokens_b

        if self._add_extra_sep_token:
            mid_sep_tokens = [self._tokenizer.sep_token] * 2
        else:
            mid_sep_tokens = [self._tokenizer.sep_token]

        all_tokens = [self._tokenizer.cls_token] + tokens_a + mid_sep_tokens + tokens_b + [self._tokenizer.sep_token]

        word_ids = self._tokenizer.convert_tokens_to_ids(all_tokens)
        word_segment_ids = [0] * (len(tokens_a) + len(mid_sep_tokens) + 1) + [self._segment_b_id] * (len(tokens_b) + 1)
        word_attention_mask = [1] * len(all_tokens)

        mentions_a = self.detect_mentions(tokens_a)
        mentions_b = self.detect_mentions(tokens_b)
        all_mentions = mentions_a + mentions_b

        if not all_mentions:
            entity_segment_ids = [0]
            entity_attention_mask = [0]
            entity_candidate_ids = [[0 for y in range(self._max_candidate_length)]]
            entity_position_ids = [[-1 for y in range(self._max_mention_length)]]
        else:
            entity_segment_ids = [0] * len(mentions_a) + [self._segment_b_id] * len(mentions_b)
            entity_attention_mask = [1] * len(all_mentions)
            entity_candidate_ids = [[0 for y in range(self._max_candidate_length)] for x in range(len(all_mentions))]
            entity_position_ids = [[-1 for y in range(self._max_mention_length)] for x in range(len(all_mentions))]

            offset_a = 1
            offset_b = len(tokens_a) + 2  # 2 for CLS and SEP tokens
            if self._add_extra_sep_token:
                offset_b += 1

            for i, (offset, (candidates, start, end)) in enumerate(chain(zip(repeat(offset_a), mentions_a),
                                                                         zip(repeat(offset_b), mentions_b))):
                cand_ids = [self._entity_vocab[UNK_TOKEN]] +\
                    [self._entity_vocab[c] for c in candidates][:self._max_candidate_length - 1]
                entity_candidate_ids[i][:len(cand_ids)] = cand_ids
                entity_position_ids[i][:end - start] = range(start + offset, end + offset)

        return dict(
            tokens=all_tokens,
            mentions=all_mentions,
            word_ids=word_ids,
            word_segment_ids=word_segment_ids,
            word_attention_mask=word_attention_mask,
            entity_candidate_ids=entity_candidate_ids,
            entity_position_ids=entity_position_ids,
            entity_segment_ids=entity_segment_ids,
            entity_attention_mask=entity_attention_mask,
        )

    def detect_mentions(self, tokens):
        mentions = []
        cur = 0
        for start, token in enumerate(tokens):
            if start < cur:
                continue
            if self._is_subword(token):
                continue

            for end in range(min(start + self._max_mention_length, len(tokens)), start, -1):
                if end < len(tokens) and self._is_subword(tokens[end]):
                    continue
                mention_text = self._tokenizer.convert_tokens_to_string(tokens[start:end]).strip()
                candidates = sorted((c for c in self._mention_db.query(mention_text) if c.title in self._entity_vocab),
                                    key=lambda c: -c.prior_prob)
                if candidates:
                    mentions.append(([c.title for c in candidates], start, end))
                    cur = end
                    break

        return mentions

    def _is_subword(self, token):
        if isinstance(self._tokenizer, RobertaTokenizer) and\
            not self._tokenizer.convert_tokens_to_string(token).startswith(' ') and\
            not self._is_punctuation(token[0]):
            return True
        elif token.startswith('##'):
            return True

        return False

    @staticmethod
    def _is_punctuation(char):
        # obtained from:
        # https://github.com/huggingface/transformers/blob/5f25a5f367497278bf19c9994569db43f96d5278/transformers/tokenization_bert.py#L489
        cp = ord(char)
        if (cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or(cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126):
            return True
        cat = unicodedata.category(char)
        if cat.startswith("P"):
            return True
        return False
