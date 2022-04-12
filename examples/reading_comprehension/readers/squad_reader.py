import json
import re
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, NamedTuple, Tuple

import numpy as np
from allennlp.data import DatasetReader, Instance, Token
from allennlp.data.fields import MetadataField, TensorField
from allennlp_models.rc.dataset_readers import TransformerSquadReader

from examples.utils.wiki_entity_linker import WikiEntityLinker


class SQuADBasedDataset(Enum):
    SQuAD = auto()
    XQuAD = auto()
    MLQA = auto()


class SQuADFile(NamedTuple):
    file_path: str
    dataset: SQuADBasedDataset
    question_language: str
    context_language: str
    title_language: str

    @classmethod
    def from_path(cls, file_path: str):

        file_name = Path(file_path).name

        xquad_matched = re.match(r"xquad\.(..)\.json", file_name)
        mlqa_matched = re.match(f"(dev|test)-context-(..)-question-(..)\.json", file_name)
        if file_name in {"train-v1.1.json", "dev-v1.1.json", "train-v2.0.json", "dev-v1.1.json"}:
            question_language = context_language = title_language = "en"
            dataset = SQuADBasedDataset.SQuAD
        elif xquad_matched:
            # the filename should look like "xquad.language.json"
            question_language = context_language = xquad_matched.groups()[0]
            dataset = SQuADBasedDataset.XQuAD
            title_language = "en"
        elif mlqa_matched:
            _, context_language, question_language = mlqa_matched.groups()
            title_language = context_language
            dataset = SQuADBasedDataset.MLQA
        else:
            raise ValueError(f"Unexpected input filename: {file_name}")

        return cls(file_path, dataset, question_language, context_language, title_language)


@DatasetReader.register("transformers_squad")
class SquadReader(TransformerSquadReader):
    def __init__(
        self,
        transformer_model_name: str,
        wiki_entity_linker: WikiEntityLinker = None,
        max_num_entity_features: int = 128,
        **kwargs,
    ):
        super().__init__(transformer_model_name, **kwargs)
        if wiki_entity_linker is not None:
            wiki_entity_linker.set_tokenizer(self._tokenizer.tokenizer)

        self.wiki_entity_linker = wiki_entity_linker
        self.max_num_entity_features = max_num_entity_features

    def _get_idx_to_title_mapping(self, file_path: str) -> Dict[str, str]:
        data = json.load(open(file_path, "r"))["data"]
        idx_to_title = {}
        for article in data:
            title = article["title"]
            for paragraph in article["paragraphs"]:
                for qa in paragraph["qas"]:
                    idx = qa["id"]
                    idx_to_title[idx] = title
        return idx_to_title

    def _read(self, file_path: str):

        idx_to_title_mapping = self._get_idx_to_title_mapping(file_path)

        squad_file = SQuADFile.from_path(file_path)

        for instance in super()._read(file_path):
            index = instance["metadata"].metadata["id"]
            title = idx_to_title_mapping[index].replace("_", " ")
            input_tokens = [t.text for t in instance["question_with_context"]]
            instance = Instance(
                {
                    "question_with_context": instance["question_with_context"],
                    "answer_span": instance["answer_span"],
                    "context_span": instance["context_span"],
                    "metadata": MetadataField(
                        {
                            "input_tokens": input_tokens,
                            "example_id": instance["metadata"].metadata["id"],
                            "answers": instance["metadata"].metadata["answers"],
                            "context_tokens": instance["metadata"].metadata["context_tokens"],
                            "context": instance["metadata"].metadata["context"],
                            "title": title,
                        }
                    ),
                }
            )
            if self.wiki_entity_linker is not None:

                entity_features = self.get_entity_features(
                    instance["question_with_context"],
                    title=title,
                    question_language=squad_file.question_language,
                    context_language=squad_file.context_language,
                    title_language=squad_file.title_language,
                    context_span=(instance["context_span"].span_start, instance["context_span"].span_end),
                )
                for key, value in entity_features.items():
                    instance.add_field(key, value)

            yield instance

    def get_entity_features(
        self,
        tokens: List[Token],
        title: str,
        question_language: str,
        context_language: str,
        title_language: str,
        context_span: Tuple[int, int],
    ):

        mentions = self.wiki_entity_linker.link_entities(
            tokens, token_language=question_language, title=title, title_language=title_language
        )
        context_start, context_end = context_span
        if context_language != question_language:
            question_mentions = [m for m in mentions if m.end < context_start]
            context_mentions = self.wiki_entity_linker.link_entities(
                tokens, token_language=context_language, title=title, title_language=context_language
            )
            context_mentions = [m for m in context_mentions if m.start > context_start]
            mentions = question_mentions + context_mentions

        mentions = mentions[: self.max_num_entity_features]

        entity_features = self.wiki_entity_linker.mentions_to_entity_features(tokens, mentions)

        entity_feature_fields = {}
        for name, feature in entity_features.items():
            entity_feature_fields[name] = TensorField(np.array(feature), padding_value=0)
        return entity_feature_fields
