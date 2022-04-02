import re
import string
import sys
import unicodedata
from typing import List, Optional

PUNCT = {chr(i) for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith("P")}.union(
    string.punctuation
)


def whitespace_tokenize(text: str) -> List[str]:
    return text.split()


def mixed_segmentation(text: str) -> List[str]:
    segs_out = []
    temp_str = ""
    for char in text:
        if re.search(r"[\u4e00-\u9fa5]", char) or char in PUNCT:
            if temp_str != "":
                ss = whitespace_tokenize(temp_str)
                segs_out.extend(ss)
                temp_str = ""
            segs_out.append(char)
        else:
            temp_str += char

    if temp_str != "":
        ss = whitespace_tokenize(temp_str)
        segs_out.extend(ss)

    return segs_out


class MLQALanguage:
    def __init__(self, articles_regex_pattern: Optional[re.Pattern] = None):
        self.articles_regex_pattern = articles_regex_pattern

    def tokenize(self, text: str):
        return whitespace_tokenize(text)

    @classmethod
    def from_code(cls, code: str):
        code_to_language = {
            "en": English,
            "es": Spanish,
            "hi": Hindi,
            "vi": Vietnamese,
            "de": German,
            "ar": Arabic,
            "zh": Chinese,
        }

        if code not in code_to_language:
            return MLQALanguage()

        return code_to_language[code]()


# The followings are languages defined in MLQA
class English(MLQALanguage):
    def __init__(self):
        super().__init__(re.compile(r"\b(a|an|the)\b"))

    def tokenize(self, text: str):
        return whitespace_tokenize(text)


class Spanish(MLQALanguage):
    def __init__(self):
        super().__init__(re.compile(r"\b(un|una|unos|unas|el|la|los|las)\b"))

    def tokenize(self, text: str):
        return whitespace_tokenize(text)


class Hindi(MLQALanguage):
    def tokenize(self, text: str):
        return whitespace_tokenize(text)


class Vietnamese(MLQALanguage):
    def __init__(self):
        super().__init__(re.compile(r"\b(của|là|cái|chiếc|những)\b"))

    def tokenize(self, text: str):
        return whitespace_tokenize(text)


class German(MLQALanguage):
    def __init__(self):
        super().__init__(re.compile(r"\b(ein|eine|einen|einem|eines|einer|der|die|das|den|dem|des)\b"))

    def tokenize(self, text: str):
        return whitespace_tokenize(text)


class Arabic(MLQALanguage):
    def __init__(self):
        super().__init__(re.compile(r"\sال^|ال"))

    def tokenize(self, text: str):
        return whitespace_tokenize(text)


class Chinese(MLQALanguage):
    def tokenize(self, text: str):
        return mixed_segmentation(text)
