from typing import Dict, Any, Iterable
import h5py

import numpy as np
import torch

from allennlp.data import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.data.fields import TensorField


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

    def text_to_instance(self, entity_name: str, word_ids: np.ndarray, entity_position_ids: np.ndarray) -> Instance:
        batch_size = word_ids.shape[0]
        sentence_lengths = (word_ids != -1).sum(axis=-1)

        indices = list(range(batch_size))
        np.random.shuffle(indices)

        sampled_indices = []
        current_sequence_length = 0
        for i in indices:
            next_sentence_length = current_sequence_length + sentence_lengths[i] + 1
            if next_sentence_length > self.max_sequence_length:
                break
            current_sequence_length = next_sentence_length
            sampled_indices.append(i)

        entity_position_offsets = np.cumsum((1 + sentence_lengths[sampled_indices][:-1]))
        entity_position_offsets = np.insert(entity_position_offsets, 0, 0)
        new_word_ids = []
        new_entity_position_ids = []
        for i, position_offset in zip(sampled_indices, entity_position_offsets):
            if new_word_ids:
                new_word_ids.append(np.array([self.transformers_tokenizer.sep_token_id]))
            else:
                new_word_ids.append(np.array([self.transformers_tokenizer.cls_token_id]))
            new_word_ids.append(word_ids[i][word_ids[i] > -1])

            new_position_ids = entity_position_ids[i]
            new_position_ids[new_position_ids > -1] += position_offset
            new_entity_position_ids.append(new_position_ids)

        return Instance(
            {
                "word_ids": TensorField(
                    np.concatenate(new_word_ids),
                    padding_value=self.transformers_tokenizer.pad_token_id,
                    dtype=torch.long,
                ),
                "entity_position_ids": TensorField(np.stack(new_entity_position_ids), dtype=torch.long),
                "entity_mask_tokens": TensorField(
                    torch.tensor([self.entity_vocab["[MASK]"]]),
                    padding_value=self.entity_vocab["[PAD]"],
                    dtype=torch.long,
                ),
                "entity_id": TensorField(torch.tensor([self.entity_vocab[entity_name]]), dtype=torch.long),
            }
        )

    def _read(self, file_path) -> Iterable[Instance]:
        hf = h5py.File(file_path, "r")

        for entity_name in hf.keys():
            yield self.text_to_instance(
                entity_name,
                np.array(hf[f"/{entity_name}/word_ids"]),
                np.array(hf[f"/{entity_name}/entity_position_ids"]),
            )
