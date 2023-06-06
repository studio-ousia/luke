import itertools
import math
import warnings
from typing import Dict, List, Tuple

import numpy as np
from allennlp.data import DatasetReader, Instance, Token, TokenIndexer, Tokenizer
from allennlp.data.fields import LabelField, ListField, MetadataField, TensorField, TextField
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from seqeval.scheme import IOB1, IOB2, Entities, Entity
from transformers.models.luke.tokenization_luke import LukeTokenizer
from transformers.models.mluke.tokenization_mluke import MLukeTokenizer


NON_ENTITY = "O"


def parse_conll_ner_data(input_file: str, encoding: str = "utf-8"):

    words: List[str] = []
    labels: List[str] = []
    sentence_boundaries: List[int] = [0]

    try:
        with open(input_file, "r", encoding=encoding) as f:
            for line in f:
                line = line.rstrip()
                if line.startswith("-DOCSTART"):
                    if words:
                        assert sentence_boundaries[0] == 0
                        assert sentence_boundaries[-1] == len(words)
                        yield words, labels, sentence_boundaries
                        words = []
                        labels = []
                        sentence_boundaries = [0]
                    continue

                if not line:
                    if len(words) != sentence_boundaries[-1]:
                        sentence_boundaries.append(len(words))
                else:
                    parts = line.split(" ")
                    words.append(parts[0])
                    labels.append(parts[-1])

        if words:
            yield words, labels, sentence_boundaries
    except UnicodeDecodeError as e:
        raise Exception("The specified encoding seems wrong. Try either ISO-8859-1 or utf-8.") from e


def check_add_prefix_space(tokenizer: Tokenizer):
    """
    Because tokenization is performed on words,
    some tokenizers need add_prefix_space option to preserve word boundaries.
    """
    if isinstance(tokenizer, PretrainedTransformerTokenizer):
        transformer_tokenizer = tokenizer.tokenizer
        if hasattr(transformer_tokenizer, "add_prefix_space"):
            assert transformer_tokenizer.add_prefix_space


@DatasetReader.register("conll_span")
class ConllSpanReader(DatasetReader):
    def __init__(
        self,
        token_indexers: Dict[str, TokenIndexer],
        tokenizer: Tokenizer = None,
        max_sequence_length: int = 512,
        max_entity_length: int = 128,
        max_mention_length: int = 16,
        iob_scheme: str = "iob2",
        encoding: str = "utf-8",
        use_entity_feature: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if iob_scheme == "iob1":
            self.iob_scheme = IOB1
        elif iob_scheme == "iob2":
            self.iob_scheme = IOB2
        else:
            raise ValueError(f"Invalid iob_scheme: {iob_scheme}")

        check_add_prefix_space(tokenizer)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers

        self.max_sequence_length = max_sequence_length
        self.max_num_subwords = max_sequence_length - 2  # take the number of Special tokens into account
        self.max_entity_length = max_entity_length
        self.max_mention_length = max_mention_length

        self.encoding = encoding
        self.use_entity_feature = use_entity_feature

        if isinstance(self.tokenizer.tokenizer, (LukeTokenizer, MLukeTokenizer)):
            self.entity_id = self.tokenizer.tokenizer.entity_vocab["[MASK]"]
            self.entity_pad_id = self.tokenizer.tokenizer.entity_vocab["[PAD]"]
        else:
            self.entity_id = 1
            self.entity_pad_id = 0

    def data_to_instance(self, words: List[str], labels: List[str], sentence_boundaries: List[int], doc_index: str):
        if self.tokenizer is None:
            tokens = [[Token(w)] for w in words]
        else:
            tokens = [self.tokenizer.tokenize(w) for w in words]
        subwords = [sw for token in tokens for sw in token]

        subword2token = list(itertools.chain(*[[i] * len(token) for i, token in enumerate(tokens)]))
        token2subword = [0] + list(itertools.accumulate(len(token) for token in tokens))
        subword_start_positions = frozenset(token2subword)
        subword_sentence_boundaries = [sum(len(token) for token in tokens[:p]) for p in sentence_boundaries]

        # extract entities from IOB tags
        # we need to pass sentence by sentence
        entities: List[Entity] = []
        for s, e in zip(sentence_boundaries[:-1], sentence_boundaries[1:]):
            for ent in Entities([labels[s:e]], scheme=self.iob_scheme).entities[0]:
                ent.start += s
                ent.end += s
                entities.append(ent)

        span_to_entity_label: Dict[Tuple[int, int], str] = dict()
        for ent in entities:
            subword_start = token2subword[ent.start]
            subword_end = token2subword[ent.end]
            span_to_entity_label[(subword_start, subword_end)] = ent.tag

        # split data according to sentence boundaries
        for n in range(len(subword_sentence_boundaries) - 1):
            # process (sub) words
            doc_sent_start, doc_sent_end = subword_sentence_boundaries[n : n + 2]
            assert doc_sent_end - doc_sent_start < self.max_num_subwords

            left_length = doc_sent_start
            right_length = len(subwords) - doc_sent_end
            sentence_length = doc_sent_end - doc_sent_start
            half_context_length = int((self.max_num_subwords - sentence_length) / 2)

            if left_length < right_length:
                left_context_length = min(left_length, half_context_length)
                right_context_length = min(right_length, self.max_num_subwords - left_context_length - sentence_length)
            else:
                right_context_length = min(right_length, half_context_length)
                left_context_length = min(left_length, self.max_num_subwords - right_context_length - sentence_length)

            doc_offset = doc_sent_start - left_context_length
            word_ids = subwords[doc_offset : doc_sent_end + right_context_length]

            if isinstance(self.tokenizer, PretrainedTransformerTokenizer):
                word_ids = self.tokenizer.add_special_tokens(word_ids)

            # process entities
            entity_start_positions = []
            entity_end_positions = []
            entity_ids = []
            entity_position_ids = []
            original_entity_spans = []
            labels = []

            for entity_start in range(left_context_length, left_context_length + sentence_length):
                doc_entity_start = entity_start + doc_offset
                if doc_entity_start not in subword_start_positions:
                    continue
                for entity_end in range(entity_start + 1, left_context_length + sentence_length + 1):
                    doc_entity_end = entity_end + doc_offset
                    if doc_entity_end not in subword_start_positions:
                        continue

                    if entity_end - entity_start > self.max_mention_length:
                        continue

                    entity_start_positions.append(entity_start + 1)
                    entity_end_positions.append(entity_end)
                    entity_ids.append(self.entity_id)

                    position_ids = list(range(entity_start + 1, entity_end + 1))
                    position_ids += [-1] * (self.max_mention_length - entity_end + entity_start)
                    entity_position_ids.append(position_ids)

                    original_entity_spans.append(
                        (subword2token[doc_entity_start], subword2token[doc_entity_end - 1] + 1)
                    )
                    labels.append(span_to_entity_label.pop((doc_entity_start, doc_entity_end), NON_ENTITY))

            # split instances
            split_size = math.ceil(len(entity_ids) / self.max_entity_length)
            for i in range(split_size):
                entity_size = math.ceil(len(entity_ids) / split_size)
                start = i * entity_size
                end = start + entity_size
                fields = {
                    "word_ids": TextField(word_ids, token_indexers=self.token_indexers),
                    "entity_start_positions": TensorField(np.array(entity_start_positions[start:end])),
                    "entity_end_positions": TensorField(np.array(entity_end_positions[start:end])),
                    "original_entity_spans": TensorField(np.array(original_entity_spans[start:end]), padding_value=-1),
                    "labels": ListField([LabelField(l) for l in labels[start:end]]),
                    "doc_id": MetadataField(doc_index),
                    "input_words": MetadataField(words),
                }

                if self.use_entity_feature:
                    fields.update(
                        {
                            "entity_ids": TensorField(np.array(entity_ids[start:end]), padding_value=self.entity_pad_id),
                            "entity_position_ids": TensorField(np.array(entity_position_ids[start:end])),
                            "entity_attention_mask": TensorField(np.ones(len(entity_ids[start:end]), dtype=np.int64), padding_value=0),
                        }
                    )

                yield Instance(fields)

        for (s, e), entity in span_to_entity_label.items():
            mention_length = e - s + 1
            if mention_length > self.max_mention_length:
                warnings.warn(
                    f"An entity is discarded because it exceeds max_mention_length: "
                    f"{mention_length} > {self.max_mention_length}"
                )
            else:
                raise RuntimeError(entity)

    def _read(self, file_path: str):
        for i, (words, labels, sentence_boundaries) in enumerate(
            parse_conll_ner_data(file_path, encoding=self.encoding)
        ):
            yield from self.data_to_instance(words, labels, sentence_boundaries, f"doc{i}")
