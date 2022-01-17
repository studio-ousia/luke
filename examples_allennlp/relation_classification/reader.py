from typing import Dict
import json
from pathlib import Path
import numpy as np
from allennlp.data import DatasetReader, TokenIndexer, Tokenizer, Instance, Token
from allennlp.data.fields import SpanField, TextField, LabelField, TensorField, MetadataField

from transformers.models.luke.tokenization_luke import LukeTokenizer
from examples_allennlp.utils.util import ENT, ENT2, list_rindex


def parse_tacred_file(path: str):
    if Path(path).suffix != ".json":
        raise ValueError(f"{path} does not seem to be a json file. We currently only supports the json format file.")
    for example in json.load(open(path, "r")):
        tokens = example["token"]
        spans = [
            ((example["subj_start"], ENT), (example["subj_end"] + 1, ENT)),
            ((example["obj_start"], ENT2), (example["obj_end"] + 1, ENT2)),
        ]

        # carefully insert special tokens in a specific order
        spans.sort()
        for i, span in enumerate(spans):
            (start_idx, start_token), (end_idx, end_token) = span
            tokens.insert(end_idx + i * 2, end_token)
            tokens.insert(start_idx + i * 2, start_token)

        sentence = " ".join(tokens)
        # we do not need some spaces
        sentence = sentence.replace(f" {ENT} ", f"{ENT} ")
        sentence = sentence.replace(f" {ENT2} ", f"{ENT2} ")

        yield {"example_id": example["id"], "sentence": sentence, "label": example["relation"]}


@DatasetReader.register("relation_classification")
class RelationClassificationReader(DatasetReader):
    def __init__(
        self, tokenizer: Tokenizer, token_indexers: Dict[str, TokenIndexer], use_entity_feature: bool = False, **kwargs,
    ):
        super().__init__(**kwargs)

        self.parser = parse_tacred_file
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers
        self.use_entity_feature = use_entity_feature

        if isinstance(self.tokenizer.tokenizer, LukeTokenizer):
            self.head_entity_id = self.tokenizer.tokenizer.entity_vocab["[MASK]"]
            self.tail_entity_id = self.tokenizer.tokenizer.entity_vocab["[MASK2]"]
        else:
            self.head_entity_id = 1
            self.tail_entity_id = 2

    def text_to_instance(self, sentence: str, label: str = None):
        texts = [t.text for t in self.tokenizer.tokenize(sentence)]
        e1_start_position = texts.index(ENT)
        e1_end_position = list_rindex(texts, ENT)

        e2_start_position = texts.index(ENT2)
        e2_end_position = list_rindex(texts, ENT2)

        tokens = [Token(t) for t in texts]
        text_field = TextField(tokens, token_indexers=self.token_indexers)

        fields = {
            "word_ids": text_field,
            "entity1_span": SpanField(e1_start_position, e1_end_position, text_field),
            "entity2_span": SpanField(e2_start_position, e2_end_position, text_field),
            "input_sentence": MetadataField(sentence),
        }

        if label is not None:
            fields["label"] = LabelField(label)

        if self.use_entity_feature:
            fields["entity_ids"] = TensorField(np.array([self.head_entity_id, self.tail_entity_id]))

        return Instance(fields)

    def _read(self, file_path: str):
        for data in self.parser(file_path):
            yield self.text_to_instance(data["sentence"], data["label"])
