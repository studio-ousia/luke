import json
from typing import Dict, List

import numpy as np
from allennlp.data import DatasetReader, Instance, Token, TokenIndexer, Tokenizer
from allennlp.data.fields import MetadataField, MultiLabelField, SpanField, TensorField, TextField
from transformers.models.luke.tokenization_luke import LukeTokenizer

from examples.utils.util import ENT, list_rindex

ENTITY_LABELS = {"entity", "event", "group", "location", "object", "organization", "person", "place", "time"}

NORMALIZATION_TABLE = (
    ("-LRB-", "("),
    ("-LCB-", "("),
    ("-LSB-", "("),
    ("-RRB-", ")"),
    ("-RCB-", ")"),
    ("-RSB-", ")"),
)


def parse_open_entity_dataset(path: str):
    with open(path, "r") as f:
        for line in f:
            example = json.loads(line.strip())
            labels = [l for l in example["y_str"] if l in ENTITY_LABELS]
            left_context_text = " ".join(example["left_context_token"])
            right_context_text = " ".join(example["right_context_token"])
            sentence = left_context_text + ENT + " " + example["mention_span"] + ENT + " " + right_context_text
            for before, after in NORMALIZATION_TABLE:
                sentence = sentence.replace(before, after)
            yield {"sentence": sentence, "labels": labels}


@DatasetReader.register("entity_typing")
class EntityTypingReader(DatasetReader):
    def __init__(
        self, tokenizer: Tokenizer, token_indexers: Dict[str, TokenIndexer], use_entity_feature: bool = False, **kwargs,
    ):
        super().__init__(**kwargs)

        self.tokenizer = tokenizer
        self.token_indexers = token_indexers
        self.use_entity_feature = use_entity_feature

        if isinstance(self.tokenizer.tokenizer, LukeTokenizer):
            self.entity_id = self.tokenizer.tokenizer.entity_vocab["[MASK]"]
        else:
            self.entity_id = 1

    def text_to_instance(self, sentence: str, labels: List[str] = None):
        texts = [t.text for t in self.tokenizer.tokenize(sentence)]

        ent_start_position = texts.index(ENT)
        ent_end_position = list_rindex(texts, ENT)

        tokens = [Token(t) for t in texts]
        text_field = TextField(tokens, token_indexers=self.token_indexers)
        fields = {
            "word_ids": text_field,
            "entity_span": SpanField(ent_start_position, ent_end_position, text_field),
            "input_sentence": MetadataField(sentence),
        }

        if labels is not None:
            fields["labels"] = MultiLabelField(labels)

        if self.use_entity_feature:
            fields["entity_ids"] = TensorField(np.array([self.entity_id]))

        return Instance(fields)

    def _read(self, file_path: str):
        for data in parse_open_entity_dataset(file_path):
            yield self.text_to_instance(data["sentence"], data["labels"])
