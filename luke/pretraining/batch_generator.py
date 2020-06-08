from typing import List, Iterator
import functools
import logging
import multiprocessing
import queue
import random
import unicodedata

import numpy as np
from transformers.tokenization_roberta import RobertaTokenizer

from luke.pretraining.dataset import WikipediaPretrainingDataset
from luke.utils.entity_vocab import MASK_TOKEN

logger = logging.getLogger(__name__)


class LukePretrainingBatchGenerator(object):
    """
    Launch a new process in order to avoid data processing being a bottleneck during training.
    """

    def __init__(
        self,
        dataset_dir: str,
        batch_size: int,
        masked_lm_prob: float,
        masked_entity_prob: float,
        whole_word_masking: bool,
        **dataset_kwargs
    ):
        self._worker_func = functools.partial(
            LukePretrainingBatchWorker,
            dataset_dir=dataset_dir,
            batch_size=batch_size,
            masked_lm_prob=masked_lm_prob,
            masked_entity_prob=masked_entity_prob,
            whole_word_masking=whole_word_masking,
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
        dataset_dir: str,
        batch_size: int,
        masked_lm_prob: float,
        masked_entity_prob: float,
        whole_word_masking: bool,
        **dataset_kwargs
    ):
        super(LukePretrainingBatchWorker, self).__init__()

        self._output_queue = output_queue
        self._dataset_dir = dataset_dir
        self._batch_size = batch_size
        self._masked_lm_prob = masked_lm_prob
        self._masked_entity_prob = masked_entity_prob
        self._whole_word_masking = whole_word_masking
        self._dataset_kwargs = dataset_kwargs

        if "shuffle_buffer_size" not in self._dataset_kwargs:
            self._dataset_kwargs["shuffle_buffer_size"] = batch_size * 1000

    def run(self):
        self._pretraining_dataset = WikipediaPretrainingDataset(self._dataset_dir)
        self._tokenizer = self._pretraining_dataset.tokenizer
        self._entity_vocab = self._pretraining_dataset.entity_vocab
        self._max_seq_length = self._pretraining_dataset.max_seq_length
        self._max_entity_length = self._pretraining_dataset.max_entity_length
        self._max_mention_length = self._pretraining_dataset.max_mention_length
        self._cls_id = self._tokenizer.convert_tokens_to_ids(self._tokenizer.cls_token)
        self._sep_id = self._tokenizer.convert_tokens_to_ids(self._tokenizer.sep_token)
        self._mask_id = self._tokenizer.convert_tokens_to_ids(
            self._tokenizer.mask_token
        )
        self._pad_id = self._tokenizer.convert_tokens_to_ids(self._tokenizer.pad_token)
        self._entity_mask_id = self._pretraining_dataset.entity_vocab[MASK_TOKEN]

        buf = []
        max_word_len = 1
        max_entity_len = 1
        for item in self._pretraining_dataset.create_iterator(**self._dataset_kwargs):
            word_feat = self._create_word_features(item["word_ids"])
            entity_feat = self._create_entity_features(
                item["entity_ids"], item["entity_position_ids"]
            )
            max_word_len = max(
                max_word_len, item["word_ids"].size + 2
            )  # 2 for [CLS] and [SEP]
            max_entity_len = max(max_entity_len, item["entity_ids"].size)
            buf.append((word_feat, entity_feat, item["page_id"]))

            if len(buf) == self._batch_size:
                batch = {}
                batch.update(
                    {
                        k: np.stack([o[0][k][:max_word_len] for o in buf])
                        for k in buf[0][0].keys()
                    }
                )
                batch.update(
                    {
                        k: np.stack([o[1][k][:max_entity_len] for o in buf])
                        for k in buf[0][1].keys()
                    }
                )
                self._output_queue.put(batch, True)

                buf = []
                max_word_len = 1
                max_entity_len = 1

    def _create_word_features(self, word_ids: np.ndarray):
        output_word_ids = np.full(self._max_seq_length, self._pad_id, dtype=np.int)
        output_word_ids[: word_ids.size + 2] = np.concatenate(
            [[self._cls_id], word_ids, [self._sep_id]]
        )
        word_attention_mask = np.zeros(self._max_seq_length, dtype=np.int)
        word_attention_mask[: word_ids.size + 2] = 1

        ret = dict(
            word_ids=output_word_ids,
            word_attention_mask=word_attention_mask,
            word_segment_ids=np.zeros(self._max_seq_length, dtype=np.int),
        )

        if self._masked_lm_prob != 0.0:
            num_to_predict = max(1, int(round(word_ids.size * self._masked_lm_prob)))
            candidate_word_indices = []

            for i, word in enumerate(
                self._tokenizer.convert_ids_to_tokens(word_ids), 1
            ):  # 1 for [CLS]
                if (
                    self._whole_word_masking
                    and self._is_subword(word)
                    and candidate_word_indices
                ):
                    candidate_word_indices[-1].append(i)
                else:
                    candidate_word_indices.append([i])

            masked_lm_labels = np.full(self._max_seq_length, -1, dtype=np.int)
            num_masked_words = 0

            for i in np.random.permutation(len(candidate_word_indices)):
                indices_to_mask = candidate_word_indices[i]
                if len(indices_to_mask) > num_to_predict - num_masked_words:
                    continue

                p = random.random()
                for index in indices_to_mask:
                    masked_lm_labels[index] = output_word_ids[index]
                    if p < 0.8:
                        output_word_ids[index] = self._mask_id
                    elif p < 0.9:
                        output_word_ids[index] = random.randint(
                            self._pad_id + 1, self._tokenizer.vocab_size - 1
                        )
                    num_masked_words += 1

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

    def _create_entity_features(
        self, entity_ids: np.ndarray, entity_position_ids: np.ndarray
    ):
        output_entity_ids = np.zeros(self._max_entity_length, dtype=np.int)
        output_entity_ids[: entity_ids.size] = entity_ids

        entity_attention_mask = np.zeros(self._max_entity_length, dtype=np.int)
        entity_attention_mask[: entity_ids.size] = 1

        entity_position_ids += entity_position_ids != -1  # +1 for [CLS]
        output_entity_position_ids = np.full(
            (self._max_entity_length, self._max_mention_length), -1, dtype=np.int
        )
        output_entity_position_ids[: entity_position_ids.shape[0]] = entity_position_ids

        ret = dict(
            entity_ids=output_entity_ids,
            entity_position_ids=output_entity_position_ids,
            entity_attention_mask=entity_attention_mask,
            entity_segment_ids=np.zeros(self._max_entity_length, dtype=np.int),
        )

        if self._masked_entity_prob != 0.0:
            num_to_predict = max(
                1, int(round(entity_ids.size * self._masked_entity_prob))
            )
            masked_entity_labels = np.full(self._max_entity_length, -1, dtype=np.int)
            for index in np.random.permutation(range(entity_ids.size))[:num_to_predict]:
                masked_entity_labels[index] = entity_ids[index]
                output_entity_ids[index] = self._entity_mask_id
            ret["masked_entity_labels"] = masked_entity_labels

        return ret

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
        if (
            (cp >= 33 and cp <= 47)
            or (cp >= 58 and cp <= 64)
            or (cp >= 91 and cp <= 96)
            or (cp >= 123 and cp <= 126)
        ):
            return True
        cat = unicodedata.category(char)
        if cat.startswith("P"):
            return True
        return False


class MultilingualBatchGenerator(LukePretrainingBatchGenerator):
    """
    Launch a new process in order to avoid data processing being a bottleneck during training.
    """

    def __init__(
        self,
        dataset_dir_list: List[str],
        dataset_size_list: List[int],
        sampling_smoothing_factor: float,
        batch_size: int,
        masked_lm_prob: float,
        masked_entity_prob: float,
        whole_word_masking: bool,
        **dataset_kwargs
    ):

        self.batch_generator_list = [
            LukePretrainingBatchGenerator(
                dataset_dir,
                batch_size,
                masked_lm_prob,
                masked_entity_prob,
                whole_word_masking,
                **dataset_kwargs
            )
            for dataset_dir in dataset_dir_list
        ]
        self.sampling_rate = self.get_sampling_rate(
            dataset_size_list, sampling_smoothing_factor
        )

    def generate_batches(self, queue_size: int = 10000):
        batch_iterators = [
            g.generate_batches(queue_size) for g in self.batch_generator_list
        ]
        yield from self.sampling_from_iterators(
            batch_iterators, sampling_rate=self.sampling_rate
        )

    @staticmethod
    def get_sampling_rate(
        data_size_list: List[int], smoothing_factor: float = 0.7
    ) -> List[float]:
        """
        Exponentially smoothing the weighting of multilingual data.
        When ``smoothing_factor`` is set to 1, the sampling distribution is faithful to the original data size.
        When 0, the distribution will be the uniform distribution.
        """
        data_size_list = [size ** smoothing_factor for size in data_size_list]
        size_sum = sum(data_size_list)
        return [size / size_sum for size in data_size_list]

    @staticmethod
    def sampling_from_iterators(iterators: List[Iterator], sampling_rate: List[float]):
        """
        Randomly choose an iterator according to ``sampling_rate``, and yield an element from it.
        """
        while True:
            g = np.random.choice(iterators, p=sampling_rate)
            try:
                yield next(g)
            except StopIteration:
                break
