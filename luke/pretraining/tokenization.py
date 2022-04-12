import logging
import re
from typing import List

from transformers import PreTrainedTokenizer, RobertaTokenizer, XLMRobertaTokenizer

logger = logging.getLogger(__name__)

XLM_ROBERTA_UNK_CHAR = "龘"


def tokenize(text: str, tokenizer: PreTrainedTokenizer, add_prefix_space: bool):
    text = re.sub(r"\s+", " ", text).rstrip()
    add_prefix_space = text.startswith(" ") or add_prefix_space
    if not text:
        return []
    try:
        if isinstance(tokenizer, RobertaTokenizer):
            return tokenizer.tokenize(text, add_prefix_space=add_prefix_space)
        elif isinstance(tokenizer, XLMRobertaTokenizer):
            if add_prefix_space:
                return tokenizer.tokenize(text)
            else:
                # Append UNK_CHAR, and remove this below to avoid unnecessary whitespace.
                return tokenizer.tokenize(XLM_ROBERTA_UNK_CHAR + text)[2:]
        else:
            return tokenizer.tokenize(text)
    except TypeError:
        logger.info("Error occured during tokenization. Skip.")
        return []


def tokenize_segments(
    segments: List[str], tokenizer: PreTrainedTokenizer, add_prefix_space: bool = True
) -> List[List[str]]:
    tokenized_segments = []
    for i, text in enumerate(segments):
        if i == 0:
            tokenized_segments.append(tokenize(text, tokenizer, add_prefix_space=add_prefix_space))
        else:
            prev_text = segments[i - 1]
            tokenized_segments.append(
                tokenize(text, tokenizer, add_prefix_space=prev_text.endswith(" ") or len(prev_text) == 0)
            )

    return tokenized_segments
