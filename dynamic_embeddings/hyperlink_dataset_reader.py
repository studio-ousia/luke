from typing import Dict, Any, Iterable
import h5py

import numpy as np
import torch

from allennlp.data import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.data.fields import TensorField

from .util import h5py_safe_name_to_original

from .hyperlink_dataset import HyperlinkDataset


@DatasetReader.register("hyperlink")
class HyperlinkDatasetReader(DatasetReader):
    def __init__(
        self, luke_model_name: str, tokenizer_kwargs: Dict[str, Any] = None, max_sequence_length: int = 512, **kwargs
    ):
        super().__init__(**kwargs)
        self.luke_model_name = luke_model_name

        self.tokenizer = PretrainedTransformerTokenizer(
            luke_model_name,
            add_special_tokens=False,
            tokenizer_kwargs=tokenizer_kwargs,
        )
        self.transformers_tokenizer = self.tokenizer.tokenizer
        self.entity_vocab = self.transformers_tokenizer.entity_vocab

        self.token_indexers = {
            "tokens": PretrainedTransformerIndexer(luke_model_name, tokenizer_kwargs=tokenizer_kwargs)
        }
        self.max_sequence_length = max_sequence_length

    def sample_data(self, sentence_lengths: np.ndarray):

        indices = list(range(len(sentence_lengths)))
        np.random.shuffle(indices)

        sampled_indices = []
        current_sequence_length = 0
        for i in indices:
            next_sentence_length = current_sequence_length + sentence_lengths[i] + 1
            if next_sentence_length > self.max_sequence_length:
                break
            current_sequence_length = next_sentence_length
            sampled_indices.append(i)
        return sampled_indices

    def text_to_instance(self, entity_name: str, word_ids: np.ndarray, entity_position_ids: np.ndarray) -> Instance:
        sentence_lengths = (word_ids != -1).sum(axis=-1)

        entity_position_offsets = np.cumsum((1 + sentence_lengths[:-1]))  # +1 for the SEP token
        entity_position_offsets = np.insert(entity_position_offsets, 0, 0)
        entity_position_offsets += 1  # +1 for the CLS token
        new_word_ids = []
        new_entity_position_ids = []
        for ws, es, position_offset in zip(word_ids, entity_position_ids, entity_position_offsets):
            if new_word_ids:
                new_word_ids.append(np.array([self.transformers_tokenizer.sep_token_id]))
            else:
                new_word_ids.append(np.array([self.transformers_tokenizer.cls_token_id]))
            new_word_ids.append(ws[ws > -1])

            new_position_ids = es
            new_position_ids[new_position_ids > -1] += position_offset
            new_entity_position_ids.append(new_position_ids)

        return Instance(
            {
                "word_ids": TensorField(
                    torch.LongTensor(np.concatenate(new_word_ids)),
                    padding_value=self.transformers_tokenizer.pad_token_id,
                ),
                "entity_position_ids": TensorField(torch.LongTensor(np.stack(new_entity_position_ids))),
                "entity_mask_tokens": TensorField(
                    torch.LongTensor([self.entity_vocab["[MASK]"]]), padding_value=self.entity_vocab["[PAD]"]
                ),
                "entity_id": TensorField(torch.LongTensor([self.entity_vocab[entity_name]])),
            }
        )

    def _read(self, file_path) -> Iterable[Instance]:

        with HyperlinkDataset(file_path, "r") as f:
            for entity_name, word_ids, entity_position_ids in f.generate_entity_data():
                sampled_indices = self.sample_data(sentence_lengths=(word_ids != -1).sum(axis=-1))
                yield self.text_to_instance(
                    entity_name,
                    word_ids=word_ids[sampled_indices],
                    entity_position_ids=entity_position_ids[sampled_indices],
                )
