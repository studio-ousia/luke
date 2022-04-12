import re
import string
from dataclasses import dataclass
from typing import List

from examples.reading_comprehension.metrics.squad.mlqa_extensions.mlqa_languages import (
    PUNCT,
    MLQALanguage,
    whitespace_tokenize,
)


@dataclass
class AnswerString:
    """ Perform the string normalization in the official SQuAD evaluation code."""

    string: str

    def get_normalized_answer(self) -> str:
        """Lower text and remove punctuation, articles and extra whitespace."""

        def remove_articles(text: str):
            return re.sub(r"\b(a|an|the)\b", " ", text)

        def white_space_fix(text: str):
            return " ".join(text.split())

        def remove_punc(text: str):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text: str):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(self.string))))

    def get_tokens(self) -> List[str]:
        return whitespace_tokenize(self.get_normalized_answer())
