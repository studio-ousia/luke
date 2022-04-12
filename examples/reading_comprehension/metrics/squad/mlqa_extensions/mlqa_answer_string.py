import re
import string
from dataclasses import dataclass
from typing import List

from examples.reading_comprehension.metrics.squad.answer_string import AnswerString
from examples.reading_comprehension.metrics.squad.mlqa_extensions.mlqa_languages import (
    PUNCT,
    MLQALanguage,
    whitespace_tokenize,
)


@dataclass
class MLQAAnswerString(AnswerString):
    """ Perform the string normalization in the MLQA evaluation code."""

    string: str
    language: MLQALanguage

    def get_normalized_answer(self):
        """Lower text and remove punctuation, articles and extra whitespace."""

        def remove_articles(text, language: MLQALanguage):
            if language.articles_regex_pattern is None:
                return text
            return re.sub(language.articles_regex_pattern, " ", text)

        def white_space_fix(text, language: MLQALanguage):
            tokens = language.tokenize(text)
            return " ".join([t for t in tokens if t.strip() != ""])

        def remove_punc(text):
            return "".join(ch for ch in text if ch not in PUNCT)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(self.string)), self.language), self.language)
