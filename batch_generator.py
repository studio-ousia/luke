# -*- coding: utf-8 -*-

import logging
import multiprocessing
import queue
import random
from itertools import chain, repeat
import numpy as np

from wiki_corpus import WikiCorpus

logger = logging.getLogger(__name__)


class BatchGenerator(object):
    def __init__(self, *args, **kwargs):
        self._worker_args = args
        self._worker_kwargs = kwargs

    def generate_batches(self, page_indices=None, queue_size=100):
        output_queue = multiprocessing.Queue(queue_size)
        is_finished = multiprocessing.Event()

        worker = BatchWorker(output_queue, is_finished, page_indices, *self._worker_args,
                             **self._worker_kwargs)
        worker.daemon = True
        worker.start()

        try:
            while True:
                try:
                    yield output_queue.get(True, 1)
                except queue.Empty:
                    if is_finished.is_set():
                        break
                    logger.debug('The queue is empty')

                    if not worker.is_alive():
                        raise RuntimeError('Worker exited unexpectedly')

        finally:
            worker.terminate()
            output_queue.close()


class BatchWorker(multiprocessing.Process):
    def __init__(self, output_queue, is_finished, page_indices, corpus_data_file, entity_vocab,
                 batch_size, max_seq_length, max_entity_length, short_seq_prob, masked_lm_prob,
                 max_predictions_per_seq, mmap):
        super(BatchWorker, self).__init__()

        self._output_queue = output_queue
        self._is_finished = is_finished
        self._page_indices = page_indices
        self._corpus_data_file = corpus_data_file
        self._entity_vocab = entity_vocab
        self._batch_size = batch_size
        self._max_seq_length = max_seq_length
        self._max_entity_length = max_entity_length
        self._short_seq_prob = short_seq_prob
        self._masked_lm_prob = masked_lm_prob
        self._max_predictions_per_seq = max_predictions_per_seq

        self._mmap_mode = None
        if mmap:
            self._mmap_mode = 'r'

        self._max_num_tokens = max_seq_length - 3  # [CLS], [SEP], and [SEP]

    def run(self):
        # WikiCorpus needs to be initialized here because the BatchWorker class is pickled when the
        # worker starts, and the WikiCorpus instance is too large to be pickled
        self._corpus = WikiCorpus(self._corpus_data_file, self._mmap_mode)

        self._word_vocab = self._corpus.word_vocab
        self._cls_id = self._word_vocab['[CLS]']
        self._sep_id = self._word_vocab['[SEP]']
        self._mask_id = self._word_vocab['[MASK]']

        buf = []
        total_pages = self._corpus.page_size
        if self._page_indices is not None:
            total_pages = len(self._page_indices)

        page_iterator = self._corpus.iterate_pages(page_indices=self._page_indices)

        for (i, sents) in enumerate(page_iterator):
            target_seq_length = self._max_num_tokens
            if random.random() < self._short_seq_prob:
                target_seq_length = random.randint(2, self._max_num_tokens)

            a_titles = set([m.title for s in sents for m in s.mentions if m.is_gold])
            a_titles.add(sents[0].page_entity.title)  # page title

            sent_stack = list(reversed(sents))
            chunk_sents = []
            chunk_word_length = 0

            while sent_stack:
                sentence = sent_stack.pop()
                chunk_sents.append(sentence)
                chunk_word_length += len(sentence.word_ids)

                if not sent_stack or chunk_word_length >= target_seq_length:
                    a_end = 1
                    if len(chunk_sents) > 1:
                        a_end = random.randint(1, len(chunk_sents) - 1)
                    a_sents = chunk_sents[:a_end]

                    if len(chunk_sents) == 1 or random.random() < 0.5:
                        is_next = False

                        a_length = sum([len(s.word_ids) for s in a_sents])
                        target_b_length = target_seq_length - a_length

                        random_sents = self._corpus.read_random_page()
                        random_start = random.randint(0, len(random_sents) - 1)
                        b_sents = []
                        b_length = 0
                        for sent in random_sents[random_start:]:
                            b_sents.append(sent)
                            b_length += len(sent.word_ids)
                            if b_length >= target_b_length:
                                break

                        b_titles = set([m.title for s in random_sents for m in s.mentions
                                        if m.is_gold])
                        b_titles.add(random_sents[0].page_entity.title)  # page title

                        # put back the unused sentences to the list
                        sent_stack.extend(reversed(chunk_sents[a_end:]))

                    else:  # actual next
                        is_next = True
                        b_sents = chunk_sents[a_end:]
                        b_titles = a_titles

                    item = self._create_item(a_sents, b_sents, a_titles, b_titles, is_next,
                                             target_seq_length)
                    buf.append(item)
                    if len(buf) == self._batch_size:
                        self._output_queue.put(self._create_batch(buf), True)
                        buf = []

                    chunk_sents = []
                    chunk_word_length = 0

            if buf and i == total_pages - 1:
                self._output_queue.put(self._create_batch(buf), True)

        self._is_finished.set()

    def _create_item(self, a_sents, b_sents, a_titles, b_titles, is_next, target_seq_length):
        a_orig_len = sum([len(s.word_ids) for s in a_sents])
        b_orig_len = sum([len(s.word_ids) for s in b_sents])

        half_target_len = round(target_seq_length / 2)
        if a_orig_len < b_orig_len:
            a_len = min(half_target_len, a_orig_len)
            b_len = min(target_seq_length - a_len, b_orig_len)
        else:
            b_len = min(half_target_len, b_orig_len)
            a_len = min(target_seq_length - b_len, a_orig_len)

        a_left_num_trunc = random.randint(0, a_orig_len - a_len)
        b_left_num_trunc = random.randint(0, b_orig_len - b_len)

        a_word_ids = []
        a_mentions = []
        for sent in a_sents:
            for mention in sent.mentions:
                ofs = len(a_word_ids) - a_left_num_trunc
                (start, end) = (mention.start + ofs, mention.end + ofs)
                if start < 0 or end > a_len:
                    continue
                a_mentions.append((start + 1, end + 1, mention.title, mention.is_gold))
            a_word_ids.extend(sent.word_ids)

        a_word_ids = a_word_ids[a_left_num_trunc:a_left_num_trunc + a_len]

        b_word_ids = []
        b_mentions = []
        for sent in b_sents:
            for mention in sent.mentions:
                ofs = len(b_word_ids) - b_left_num_trunc
                (start, end) = (mention.start + ofs, mention.end + ofs)
                if start < 0 or end > b_len:
                    continue
                b_mentions.append((start + a_len + 2, end + a_len + 2, mention.title,
                                   mention.is_gold))  # +2 for [CLS] and [SEP]
            b_word_ids.extend(sent.word_ids)

        b_word_ids = b_word_ids[b_left_num_trunc:b_left_num_trunc + b_len]

        (word_ids, word_segment_ids, masked_lm_labels, word_attention_mask) =\
            self._create_word_data(a_word_ids, b_word_ids)

        (entity_ids, entity_position_ids, entity_segment_ids, entity_attention_mask,
         entity_labels) = self._create_entity_data(a_mentions, b_mentions, a_titles, b_titles)

        return dict(
            word_ids=word_ids,
            word_segment_ids=word_segment_ids,
            word_attention_mask=word_attention_mask,
            entity_ids=entity_ids,
            entity_position_ids=entity_position_ids,
            entity_segment_ids=entity_segment_ids,
            entity_attention_mask=entity_attention_mask,
            masked_lm_labels=masked_lm_labels,
            is_random_next=int(not is_next),
            entity_labels=entity_labels,
        )

    def _create_batch(self, data):
        return {k: np.stack([o[k] for o in data]) for k in data[0].keys()}

    def _create_word_data(self, a_word_ids, b_word_ids):
        word_ids = [self._cls_id] + a_word_ids + [self._sep_id] + b_word_ids + [self._sep_id]

        num_to_predict = min(self._max_predictions_per_seq,
                             max(1, int(round((len(word_ids) - 3) * self._masked_lm_prob))))
        output_word_ids = np.zeros(self._max_seq_length, dtype=np.int)
        output_word_ids[:len(word_ids)] = word_ids

        masked_lm_labels = np.full(self._max_seq_length, -1, dtype=np.int)

        count = 0
        for index in np.random.permutation(len(word_ids)):
            if word_ids[index] in (self._cls_id, self._sep_id):
                continue

            masked_lm_labels[index] = word_ids[index]
            p = random.random()
            if p < 0.8:
                output_word_ids[index] = self._mask_id
            elif p < 0.9:
                output_word_ids[index] = random.randint(0, len(self._word_vocab) - 1)

            count += 1
            if count == num_to_predict:
                break

        word_attention_mask = np.ones(self._max_seq_length, dtype=np.int)
        word_attention_mask[len(word_ids):] = 0

        word_segment_ids = np.zeros(self._max_seq_length, dtype=np.int)
        word_segment_ids[len(a_word_ids) + 2:len(word_ids)] = 1

        return (output_word_ids, word_segment_ids, masked_lm_labels, word_attention_mask)

    def _create_entity_data(self, a_mentions, b_mentions, a_titles, b_titles):
        entity_ids = np.zeros(self._max_entity_length, dtype=np.int)
        entity_position_ids = np.zeros(self._max_entity_length, dtype=np.int)
        entity_segment_ids = np.zeros(self._max_entity_length, dtype=np.int)
        entity_labels = np.full(self._max_entity_length, -1, dtype=np.int)

        ind = 0
        for ((start, end, title, is_gold), segment_id, titles) in chain(
            zip(a_mentions, repeat(0), repeat(a_titles)),
            zip(b_mentions, repeat(1), repeat(b_titles))
        ):
            entity_id = self._entity_vocab.get_id(title)
            mention_len = end - start
            if entity_id is not None and ind + mention_len <= self._max_entity_length:
                entity_ids[ind:ind+mention_len] = entity_id
                entity_position_ids[ind:ind+mention_len] = range(start, end)
                entity_segment_ids[ind:ind+mention_len] = segment_id
                if is_gold:
                    entity_labels[ind:ind+mention_len] = 1
                elif title in titles:
                    entity_labels[ind:ind+mention_len] = -1
                else:
                    entity_labels[ind:ind+mention_len] = 0
                ind += mention_len

        entity_attention_mask = np.ones(self._max_entity_length, dtype=np.int)
        entity_attention_mask[ind:] = 0

        return (entity_ids, entity_position_ids, entity_segment_ids, entity_attention_mask,
                entity_labels)
