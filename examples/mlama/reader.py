import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
from allennlp.data import DatasetReader, Instance, Token
from allennlp.data.fields import ListField, MetadataField, TensorField, TextField
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.tokenizers import PretrainedTransformerTokenizer

from luke.pretraining.tokenization import tokenize_segments
from luke.utils.entity_vocab import EntityVocab

logger = logging.getLogger(__name__)


def parse_mlama_data(mlama_path: str, language: str):
    mlama_path = Path(mlama_path)
    template_path = mlama_path / "mlama1.1" / language / "templates.jsonl"
    relation_templates = [json.loads(l) for l in open(template_path, "r")]

    candidates = json.load(open(mlama_path / "TREx_multilingual_objects" / f"{language}.json", "r"))
    candidates.update(json.load(open(mlama_path / "GoogleRE_objects" / f"{language}.json", "r")))

    for data in relation_templates:
        relation = data["relation"]

        data_path = mlama_path / "mlama1.1" / language / f"{relation}.jsonl"
        if not data_path.exists():
            logger.info(f"{data_path} skipped.")
            continue

        examples = [json.loads(l) for l in open(data_path, "r")]
        for example in examples:
            assert example["sub_label"] in candidates[relation]["subjects"]
            assert example["obj_label"] in candidates[relation]["objects"]
            yield {
                "template": data["template"],
                "subject": example["sub_label"],
                "object": example["obj_label"],
                "candidates": candidates[relation]["objects"],
                "language": language,
            }


@DatasetReader.register("mlama")
class MultilingualLAMAReader(DatasetReader):
    def __init__(
        self,
        mlama_path: str,
        transformers_model_name: str,
        use_subject_entity_mask: bool = False,
        use_subject_entity: bool = False,
        use_object_entity: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mlama_path = mlama_path
        self.transformers_model_name = transformers_model_name

        self.tokenizer = PretrainedTransformerTokenizer(
            model_name=transformers_model_name, add_special_tokens=False, tokenizer_kwargs={"use_fast": False}
        )
        self.transformers_tokenizer = self.tokenizer.tokenizer
        self.token_indexers = {"tokens": PretrainedTransformerIndexer(model_name=transformers_model_name)}

        self.use_subject_entity_mask = use_subject_entity_mask
        self.use_subject_entity = use_subject_entity
        self.use_object_entity = use_object_entity

        self.entity_vocab = None
        if use_subject_entity_mask or use_subject_entity or use_object_entity:
            if "luke" not in transformers_model_name:
                raise ValueError("The model must be LUKE if you want to use entity..")
            self.entity_vocab = EntityVocab(transformers_model_name)

    def text_to_instances(self, template: str, subject: str, object: str, candidates: List[str], language: str):
        segments = re.search(f"(.*)(\[X\])(.*)(\[Y\])(.*)", template).groups()
        assert segments[1] == "[X]"
        assert segments[3] == "[Y]"
        assert len(segments) == 5
        segments = list(segments)

        segments[1] = subject
        segments[3] = self.transformers_tokenizer.mask_token

        tokenized_segments = tokenize_segments(segments, self.transformers_tokenizer)

        num_masks_to_tokens: Dict[int, List[Dict]] = defaultdict(list)
        for candidate_object in candidates:
            # we need to tokenize in context to properly handle white spaces
            _, candidate_tokens = tokenize_segments([segments[2], candidate_object], self.transformers_tokenizer)

            if self.entity_vocab is not None:
                object_entity_id = self.entity_vocab.get_id(candidate_object, language)
            else:
                object_entity_id = None

            num_masks_to_tokens[len(candidate_tokens)].append(
                {
                    "object": candidate_object,
                    "ids": self.transformers_tokenizer.convert_tokens_to_ids(candidate_tokens),
                    "entity_id": object_entity_id,
                }
            )

        _, correct_object_tokens = tokenize_segments([segments[2], object], self.transformers_tokenizer)
        correct_object_token_ids = self.transformers_tokenizer.convert_tokens_to_ids(correct_object_tokens)

        for num_masks in num_masks_to_tokens.keys():
            tokenized_segments[3] = [self.transformers_tokenizer.mask_token] * num_masks
            if self.entity_vocab is not None:
                entity_ids = []
                entity_attention_mask = []
                entity_token_type_ids = []
                entity_position_ids = []

                if self.use_subject_entity_mask or self.use_subject_entity:
                    if self.use_subject_entity_mask:
                        subject_entity_id = self.entity_vocab.get_id("[MASK]", language)
                        entity_attention_mask.append(1)
                    elif self.use_subject_entity:
                        subject_entity_id = self.entity_vocab.get_id(subject, language)
                        if subject_entity_id is not None:
                            entity_attention_mask.append(1)
                        else:
                            subject_entity_id = self.entity_vocab.get_id("[PAD]", language)
                            entity_attention_mask.append(0)

                    entity_ids.append(subject_entity_id)
                    entity_token_type_ids.append(0)
                    start_position = len(tokenized_segments[0])
                    end_position = start_position + len(tokenized_segments[1])
                    entity_position_ids.append([i for i in range(start_position, end_position)])

                if self.use_object_entity:
                    object_entity_id = self.entity_vocab.get_id("[MASK]", language)

                    entity_ids.append(object_entity_id)
                    entity_attention_mask.append(1)
                    entity_token_type_ids.append(0)
                    start_position = (
                        len(tokenized_segments[0]) + len(tokenized_segments[1]) + len(tokenized_segments[2])
                    )
                    end_position = start_position + len(tokenized_segments[3])
                    entity_position_ids.append([i for i in range(start_position, end_position)])

                entity_features = {
                    "entity_ids": TensorField(np.array(entity_ids), dtype=np.int64),
                    "entity_attention_mask": TensorField(np.array(entity_attention_mask), dtype=np.int64),
                    "entity_token_type_ids": TensorField(np.array(entity_token_type_ids), dtype=np.int64),
                    "entity_position_ids": ListField(
                        # + 1 for CLS token
                        [
                            TensorField(
                                np.array(pos_ids) + 1,
                                padding_value=-1,
                                dtype=np.int64,
                            )
                            for pos_ids in entity_position_ids
                        ]
                    ),
                }
            else:
                entity_features = {}

            input_tokens = [Token(t) for tokens in tokenized_segments for t in tokens]
            input_tokens = (
                [Token(self.transformers_tokenizer.cls_token)]
                + input_tokens
                + [Token(self.transformers_tokenizer.sep_token)]
            )

            masked_span = [i for i, t in enumerate(input_tokens) if t.text == self.transformers_tokenizer.mask_token]

            yield Instance(
                {
                    "input_tokens": TextField(input_tokens, token_indexers=self.token_indexers),
                    "masked_span": MetadataField(masked_span),
                    "correct_object": MetadataField({"object": object, "ids": correct_object_token_ids}),
                    "candidate_objects": MetadataField(num_masks_to_tokens[num_masks]),
                    "question": MetadataField(template.replace("[X]", subject)),
                    "template": MetadataField(template),
                    **entity_features,
                }
            )

    def _read(self, language: str):
        for example in parse_mlama_data(self.mlama_path, language):
            # hard-coding to make it compatible with the LUKE entity vocabulary where the language is None
            if "luke" in self.transformers_model_name and "mluke" not in self.transformers_model_name:
                example["language"] = None
            yield from self.text_to_instances(**example)
