# -*- coding: utf-8 -*-

import re
import unicodedata

from .vocab import UNK_TOKEN


class Token(object):
    __slots__ = ('text', 'start', 'end', 'id')

    def __init__(self, text, start, end, id_=None):
        self.text = text
        self.start = start
        self.end = end
        self.id = id_

    @property
    def span(self):
        return (self.start, self.end)

    def __str__(self):
        return self.text

    def __repr__(self):
        return '<%s>' % self.text


class BasicTokenizer(object):
    def __init__(self, lowercase=False):
        self._lowercase = lowercase
        self._rule = re.compile(r'\S+')

    def tokenize(self, text):
        if self._lowercase:
            text = text.lower()

        tokens = []
        for match_obj in self._rule.finditer(text):
            (start, end) = match_obj.span()
            word_text = match_obj.group(0)

            # iterate over characters and split the word if the character is a
            # punctuation
            cur = start
            for (i, char) in enumerate(word_text, cur):
                if self._is_punctuation(char):
                    if cur != i:
                        tokens.append(Token(text[cur:i], cur, i))
                    tokens.append(Token(text[i:i+1], i, i + 1))
                    cur = i + 1
            if cur != end:
                tokens.append(Token(text[cur:end], cur, end))

        return tokens

    @staticmethod
    def _is_punctuation(char):
        # obtained from https://github.com/google-research/bert/blob/4a47cc2da23dcb4ab1bf5c08085910b6fd94a4cf/tokenization.py#L335

        cp = ord(char)
        # We treat all non-letter/number ASCII as punctuation.
        # Characters such as "^", "$", and "`" are not in the Unicode
        # Punctuation class but we treat them as punctuation anyways, for
        # consistency.
        if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
                (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
            return True
        cat = unicodedata.category(char)
        if cat.startswith('P'):
            return True
        return False


class WordPieceTokenizer(object):
    def __init__(self, vocab, lowercase=False, max_chars_per_word=100):
        self.vocab = vocab
        self.lowercase = lowercase
        self._max_chars_per_word = max_chars_per_word

        self.basic_tokenizer = BasicTokenizer(lowercase)

    def tokenize(self, text_or_tokens, use_subword=True):
        output_tokens = []

        if isinstance(text_or_tokens, str):
            input_tokens = self.basic_tokenizer.tokenize(text_or_tokens)
        else:
            input_tokens = text_or_tokens

        if not use_subword:
            return input_tokens

        for orig_token in input_tokens:
            word_text = orig_token.text

            if len(word_text) > self._max_chars_per_word:
                token = Token(UNK_TOKEN, *orig_token.span, self.vocab[UNK_TOKEN])
                output_tokens.append(token)
                continue

            cur = 0
            unk_flag = False
            tokens = []
            while cur < len(word_text):
                if cur == 0:
                    prefixes = self.vocab.word_prefix_search(word_text)
                else:
                    prefixes = self.vocab.subword_prefix_search(word_text[cur:])

                if not prefixes:
                    unk_flag = True
                    break

                subword = max(prefixes, key=len)
                subword_len = len(subword)
                if cur != 0:
                    subword_len -= 2

                start = cur + orig_token.start
                token = Token(subword, start, start + subword_len, self.vocab[subword])
                tokens.append(token)
                cur += subword_len

            if unk_flag:
                token = Token(UNK_TOKEN, *orig_token.span, self.vocab[UNK_TOKEN])
                output_tokens.append(token)
            else:
                output_tokens.extend(tokens)

        return output_tokens

BertTokenizer = WordPieceTokenizer