import functools
import logging
import multiprocessing
import queue
import random
import unicodedata
from collections import Counter
from typing import Dict, List, NamedTuple

import numpy as np
from transformers.models.roberta import RobertaTokenizer

from luke.pretraining.dataset import WikipediaPretrainingDataset
from luke.utils.entity_vocab import MASK_TOKEN

logger = logging.getLogger(__name__)


class LukePretrainingBatchGenerator:
    """
    Launch a new process in order to avoid data processing being a bottleneck during training.
    """

    def __init__(
        self,
        datasets: List[WikipediaPretrainingDataset],
        batch_size: int,
        masked_lm_prob: float,
        masked_entity_prob: float,
        whole_word_masking: bool,
        unmasked_word_prob: float,
        random_word_prob: float,
        unmasked_entity_prob: float,
        random_entity_prob: float,
        mask_words_in_entity_span: bool,
        starting_step: int,
        word_only: bool = False,
        cls_entity_prediction: bool = False,
        **dataset_kwargs
    ):
        self._worker_func = functools.partial(
            LukePretrainingBatchWorker,
            datasets=datasets,
            batch_size=batch_size,
            masked_lm_prob=masked_lm_prob,
            masked_entity_prob=masked_entity_prob,
            whole_word_masking=whole_word_masking,
            unmasked_word_prob=unmasked_word_prob,
            random_word_prob=random_word_prob,
            unmasked_entity_prob=unmasked_entity_prob,
            random_entity_prob=random_entity_prob,
            mask_words_in_entity_span=mask_words_in_entity_span,
            starting_step=starting_step,
            word_only=word_only,
            cls_entity_prediction=cls_entity_prediction,
            **dataset_kwargs
        )

    def generate_batches(self, queue_size: int = 10000):
        output_queue = multiprocessing.Queue(queue_size)
        worker = self._worker_func(output_queue)
        worker.daemon = True
        worker.start()

        try:
            while True:
                try:
                    yield output_queue.get(True, 1)
                except queue.Empty:
                    logger.debug("Queue is empty")
                    if not worker.is_alive():
                        raise RuntimeError("Worker exited unexpectedly")
        finally:
            worker.terminate()
            output_queue.close()


class LukePretrainingBatchWorker(multiprocessing.Process):
    def __init__(
        self,
        output_queue: multiprocessing.Queue,
        datasets: List[WikipediaPretrainingDataset],
        batch_size: int,
        masked_lm_prob: float,
        masked_entity_prob: float,
        whole_word_masking: bool,
        unmasked_word_prob: float,
        random_word_prob: float,
        unmasked_entity_prob: float,
        random_entity_prob: float,
        mask_words_in_entity_span: bool,
        starting_step: int,
        word_only: bool,
        cls_entity_prediction: bool,
        **dataset_kwargs
    ):
        super().__init__()

        self._output_queue = output_queue
        self._datasets = datasets
        self._batch_size = batch_size
        self._masked_lm_prob = masked_lm_prob
        self._masked_entity_prob = masked_entity_prob
        self._whole_word_masking = whole_word_masking
        self._unmasked_word_prob = unmasked_word_prob
        self._random_word_prob = random_word_prob
        self._unmasked_entity_prob = unmasked_entity_prob
        self._random_entity_prob = random_entity_prob
        self._mask_words_in_entity_span = mask_words_in_entity_span
        self._starting_step = starting_step
        self._word_only = word_only
        self._cls_entity_prediction = cls_entity_prediction
        self._dataset_kwargs = dataset_kwargs

        if "shuffle_buffer_size" not in self._dataset_kwargs:
            self._dataset_kwargs["shuffle_buffer_size"] = batch_size * 1000

    def run(self):
        representative_dataset = self._datasets[0]

        self._tokenizer = representative_dataset.tokenizer
        self._entity_vocab = representative_dataset.entity_vocab
        self._max_seq_length = representative_dataset.max_seq_length
        self._max_entity_length = representative_dataset.max_entity_length
        self._max_mention_length = representative_dataset.max_mention_length
        self._cls_id = self._tokenizer.convert_tokens_to_ids(self._tokenizer.cls_token)
        self._sep_id = self._tokenizer.convert_tokens_to_ids(self._tokenizer.sep_token)
        self._mask_id = self._tokenizer.convert_tokens_to_ids(self._tokenizer.mask_token)
        self._pad_id = self._tokenizer.convert_tokens_to_ids(self._tokenizer.pad_token)
        self._entity_mask_id = representative_dataset.entity_vocab.get_id(MASK_TOKEN, representative_dataset.language)
        dataset_sampler = DatasetSampler(
            datasets=self._datasets,
            dataset_kwargs=self._dataset_kwargs,
            starting_step=self._starting_step,
        )

        buf = []

        class BufferItem(NamedTuple):
            word_features: Dict[str, np.ndarray]
            entity_features: Dict[str, np.ndarray]
            page_id: int

        max_word_len = 1
        max_entity_len = 1
        for item in dataset_sampler:
            if not self._word_only:
                entity_feat, masked_entity_positions = self._create_entity_features(
                    item["entity_ids"], item["entity_position_ids"]
                )
                max_entity_len = max(max_entity_len, item["entity_ids"].size)
            else:
                entity_feat = None
                masked_entity_positions = []

            word_feat = self._create_word_features(item["word_ids"], masked_entity_positions)
            max_word_len = max(max_word_len, item["word_ids"].size + 2)  # 2 for [CLS] and [SEP]

            buf.append(BufferItem(word_feat, entity_feat, item["page_id"]))

            if len(buf) == self._batch_size:
                batch = {}
                word_keys = buf[0].word_features.keys()
                batch.update({k: np.stack([o.word_features[k][:max_word_len] for o in buf]) for k in word_keys})

                if self._cls_entity_prediction:
                    batch.update({"page_id": np.array([o.page_id for o in buf], dtype=np.int)})

                if not self._word_only:
                    entity_keys = buf[0].entity_features.keys()
                    batch.update(
                        {k: np.stack([o.entity_features[k][:max_entity_len] for o in buf]) for k in entity_keys}
                    )
                self._output_queue.put(batch, True)

                buf = []
                max_word_len = 1
                max_entity_len = 1

    def _create_word_features(self, word_ids: np.ndarray, masked_entity_positions: List[List[int]]):
        output_word_ids = np.full(self._max_seq_length, self._pad_id, dtype=np.int)
        output_word_ids[: word_ids.size + 2] = np.concatenate([[self._cls_id], word_ids, [self._sep_id]])
        word_attention_mask = np.zeros(self._max_seq_length, dtype=np.int)
        word_attention_mask[: word_ids.size + 2] = 1

        ret = dict(
            word_ids=output_word_ids,
            word_attention_mask=word_attention_mask,
            word_segment_ids=np.zeros(self._max_seq_length, dtype=np.int),
        )

        if self._masked_lm_prob != 0.0:
            num_masked_words = 0
            masked_lm_labels = np.full(self._max_seq_length, -1, dtype=np.int)

            def perform_masking(indices: List[int]):
                p = random.random()
                for index in indices:
                    masked_lm_labels[index] = output_word_ids[index]
                    if p < (1.0 - self._random_word_prob - self._unmasked_word_prob):
                        output_word_ids[index] = self._mask_id
                    elif p < (1.0 - self._unmasked_word_prob):
                        output_word_ids[index] = random.randint(self._pad_id + 1, self._tokenizer.vocab_size - 1)

            masked_entity_positions_set = frozenset()
            if self._mask_words_in_entity_span:
                for indices in masked_entity_positions:
                    perform_masking(indices)
                    num_masked_words += len(indices)
                masked_entity_positions_set = frozenset([p for li in masked_entity_positions for p in li])

            num_to_predict = max(1, int(round(word_ids.size * self._masked_lm_prob)))
            candidate_word_indices = []

            for i, word in enumerate(self._tokenizer.convert_ids_to_tokens(word_ids), 1):  # 1 for [CLS]
                if self._whole_word_masking and self._is_subword(word) and candidate_word_indices:
                    candidate_word_indices[-1].append(i)
                else:
                    candidate_word_indices.append([i])

            candidate_word_indices = [
                indices
                for indices in candidate_word_indices
                if all(ind not in masked_entity_positions_set for ind in indices)
            ]

            for i in np.random.permutation(len(candidate_word_indices)):
                indices_to_mask = candidate_word_indices[i]
                if len(indices_to_mask) > num_to_predict - num_masked_words:
                    continue

                perform_masking(indices_to_mask)
                num_masked_words += len(indices_to_mask)

                if num_masked_words == num_to_predict:
                    break

            # If whole-word-masking is enabled, it is possible that no word cannot be selected for masking.
            # To deal with this, we randomly select one (sub-)word for masking if num_masked_words is zero.
            if num_masked_words == 0:
                random_index = random.randint(1, word_ids.size - 2)
                masked_lm_labels[random_index] = output_word_ids[random_index]
                output_word_ids[random_index] = self._mask_id

            ret["masked_lm_labels"] = masked_lm_labels

        return ret

    def _create_entity_features(self, entity_ids: np.ndarray, entity_position_ids: np.ndarray):
        output_entity_ids = np.zeros(self._max_entity_length, dtype=np.int)
        output_entity_ids[: entity_ids.size] = entity_ids

        entity_attention_mask = np.zeros(self._max_entity_length, dtype=np.int)
        entity_attention_mask[: entity_ids.size] = 1

        entity_position_ids += entity_position_ids != -1  # +1 for [CLS]
        output_entity_position_ids = np.full((self._max_entity_length, self._max_mention_length), -1, dtype=np.int)
        output_entity_position_ids[: entity_position_ids.shape[0]] = entity_position_ids

        ret = dict(
            entity_ids=output_entity_ids,
            entity_position_ids=output_entity_position_ids,
            entity_attention_mask=entity_attention_mask,
            entity_segment_ids=np.zeros(self._max_entity_length, dtype=np.int),
        )

        masked_positions = []
        if self._masked_entity_prob != 0.0:
            num_to_predict = max(1, int(round(entity_ids.size * self._masked_entity_prob)))
            masked_entity_labels = np.full(self._max_entity_length, -1, dtype=np.int)
            for index in np.random.permutation(range(entity_ids.size))[:num_to_predict]:
                p = random.random()
                masked_entity_labels[index] = entity_ids[index]
                if p < (1.0 - self._random_entity_prob - self._unmasked_entity_prob):
                    output_entity_ids[index] = self._entity_mask_id
                elif p < (1.0 - self._unmasked_entity_prob):
                    output_entity_ids[index] = random.randint(self._entity_mask_id + 1, self._entity_vocab.size - 1)

                masked_positions.append([int(p) for p in entity_position_ids[index] if p != -1])

            ret["masked_entity_labels"] = masked_entity_labels

        return ret, masked_positions

    def _is_subword(self, token: str):
        if (
            isinstance(self._tokenizer, RobertaTokenizer)
            and not self._tokenizer.convert_tokens_to_string(token).startswith(" ")
            and not self._is_punctuation(token[0])
        ):
            return True
        elif token.startswith("##"):
            return True

        return False

    @staticmethod
    def _is_punctuation(char: str):
        # obtained from:
        # https://github.com/huggingface/transformers/blob/5f25a5f367497278bf19c9994569db43f96d5278/transformers/tokenization_bert.py#L489
        cp = ord(char)
        if (cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126):
            return True
        cat = unicodedata.category(char)
        if cat.startswith("P"):
            return True
        return False


class DatasetSampler:
    def __init__(
        self,
        datasets: List[WikipediaPretrainingDataset],
        dataset_kwargs: Dict,
        starting_step: int,
        smoothing_factor: float = 0.7,
        random_seed: int = 0,
    ):
        np.random.seed(random_seed)
        self.datasets = datasets
        self.datasets_dirs = [d.dataset_dir for d in datasets]

        self._dataset_kwargs = dataset_kwargs
        self.num_datasets = len(datasets)
        self.sampling_rate = self.get_sampling_rate([len(d) for d in self.datasets], smoothing_factor=smoothing_factor)
        self.iterators = self._prepare_iterators(starting_step, num_workers=dataset_kwargs["num_workers"])

    def _prepare_iterators(self, starting_step: int, num_workers: int = 1):
        skip_counter = Counter()
        for i in range(starting_step // num_workers):
            d = self._sample_dataset()
            skip_counter[d] += num_workers
        iterators = {
            d.dataset_dir: d.create_iterator(skip=skip_counter[d], **self._dataset_kwargs) for d in self.datasets
        }
        return iterators

    def _sample_dataset(self):
        return np.random.choice(self.datasets_dirs, p=self.sampling_rate)

    def __iter__(self):
        """
        Randomly choose an iterator according to ``sampling_rate``, and yield an element from it.
        """
        stopped_datasets = set()
        while len(stopped_datasets) < self.num_datasets:
            sampled_dataset = self._sample_dataset()
            if sampled_dataset in stopped_datasets:
                continue
            sampled_iterator = self.iterators[sampled_dataset]
            try:
                yield next(sampled_iterator)
            except StopIteration:
                stopped_datasets.add(sampled_dataset)
                continue

    @staticmethod
    def get_sampling_rate(data_size_list: List[int], smoothing_factor: float = 0.7) -> List[float]:
        """
        Exponentially smoothing the weighting of multilingual data.
        When ``smoothing_factor`` is set to 1, the sampling distribution is faithful to the original data size.
        When 0, the distribution will be the uniform distribution.
        """
        data_size_list = [size**smoothing_factor for size in data_size_list]
        size_sum = sum(data_size_list)
        return [size / size_sum for size in data_size_list]
