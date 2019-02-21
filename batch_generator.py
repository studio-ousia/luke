# -*- coding: utf-8 -*-

import logging
import multiprocessing
import queue
import random
from itertools import chain, repeat
import numpy as np

from wiki_corpus import WikiCorpus, Word

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
                 max_predictions_per_seq, link_prob_bin_size, prior_prob_bin_size, mask_title_words,
                 mmap):
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
        self._link_prob_bin_size = link_prob_bin_size
        self._prior_prob_bin_size = prior_prob_bin_size
        self._mask_title_words = mask_title_words

        self._mmap_mode = None
        if mmap:
            self._mmap_mode = 'r'

        self._max_num_tokens = max_seq_length - 5  # 5 for CLS, ENT, ENT, SEP, SEP

    def run(self):
        # WikiCorpus needs to be initialized here because the BatchWorker class is pickled when the
        # worker starts, and the WikiCorpus instance is too large to be pickled
        self._corpus = WikiCorpus(self._corpus_data_file, self._mmap_mode)

        self._word_vocab = self._corpus.word_vocab
        self._cls_word = Word(self._word_vocab['[CLS]'], False, self._word_vocab)
        self._ent_word = Word(self._word_vocab['[unused99]'], False, self._word_vocab)
        self._sep_word = Word(self._word_vocab['[SEP]'], False, self._word_vocab)
        self._mask_word = Word(self._word_vocab['[MASK]'], False, self._word_vocab)

        buf = []
        total_pages = self._corpus.page_size
        if self._page_indices is not None:
            total_pages = len(self._page_indices)

        page_iterator = self._corpus.iterate_pages(page_indices=self._page_indices)

        for (i, page) in enumerate(page_iterator):
            target_seq_length = self._max_num_tokens
            if random.random() < self._short_seq_prob:
                target_seq_length = random.randint(2, self._max_num_tokens)

            sent_stack = list(reversed(page.sentences))
            chunk_sents = []
            chunk_word_length = 0

            while sent_stack:
                sentence = sent_stack.pop()
                chunk_sents.append(sentence)
                chunk_word_length += len(sentence.words)

                if not sent_stack or chunk_word_length >= target_seq_length:
                    a_end = 1
                    if len(chunk_sents) > 1:
                        a_end = random.randint(1, len(chunk_sents) - 1)
                    a_sents = chunk_sents[:a_end]

                    if len(chunk_sents) == 1 or random.random() < 0.5:
                        is_next = False
                        a_length = sum([len(s.words) for s in a_sents])
                        target_b_length = target_seq_length - a_length

                        random_page = self._corpus.read_random_page()
                        b_entity = random_page.entity

                        random_sents = random_page.sentences
                        random_start = random.randint(0, len(random_sents) - 1)
                        b_sents = []
                        b_length = 0
                        for sent in random_sents[random_start:]:
                            b_sents.append(sent)
                            b_length += len(sent.words)
                            if b_length >= target_b_length:
                                break

                        # put back the unused sentences to the list
                        sent_stack.extend(reversed(chunk_sents[a_end:]))

                    else:  # actual next
                        is_next = True
                        b_entity = page.entity
                        b_sents = chunk_sents[a_end:]

                    item = self._create_item(page.entity, b_entity, a_sents, b_sents, is_next,
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

    def _create_item(self, a_entity, b_entity, a_sents, b_sents, is_next, target_seq_length):
        a_orig_len = sum([len(s.words) for s in a_sents])
        b_orig_len = sum([len(s.words) for s in b_sents])

        half_target_len = round(target_seq_length / 2)
        if a_orig_len < b_orig_len:
            a_len = min(half_target_len, a_orig_len)
            b_len = min(target_seq_length - a_len, b_orig_len)
        else:
            b_len = min(half_target_len, b_orig_len)
            a_len = min(target_seq_length - b_len, a_orig_len)

        a_left_num_trunc = random.randint(0, a_orig_len - a_len)
        b_left_num_trunc = random.randint(0, b_orig_len - b_len)

        a_words = []
        a_mentions = []
        for sent in a_sents:
            for mention in sent.mentions:
                ofs = len(a_words) - a_left_num_trunc
                (start, end) = (mention.start + ofs, mention.end + ofs)
                if start < 0 or end > a_len:
                    continue
                a_mentions.append((start + 3, end + 3, mention.title, mention.link_prob,
                                   mention.prior_prob, mention.label))
                                   # 3 for CLS, ENT and ENT
            a_words.extend(sent.words)

        a_words = a_words[a_left_num_trunc:a_left_num_trunc + a_len]

        b_words = []
        b_mentions = []
        for sent in b_sents:
            for mention in sent.mentions:
                ofs = len(b_words) - b_left_num_trunc
                (start, end) = (mention.start + ofs, mention.end + ofs)
                if start < 0 or end > b_len:
                    continue
                b_mentions.append((start + a_len + 4, end + a_len + 4, mention.title,
                                   mention.link_prob, mention.prior_prob, mention.label))
                                   # 4 for CLS, ENT, ENT, and SEP
            b_words.extend(sent.words)

        b_words = b_words[b_left_num_trunc:b_left_num_trunc + b_len]

        a_entity_id = self._entity_vocab.get_id(a_entity.title, -1)
        b_entity_id = self._entity_vocab.get_id(b_entity.title, -1)

        a_mask = (self._mask_title_words and a_entity_id != -1)
        b_mask = (self._mask_title_words and b_entity_id != -1)

        (word_ids, word_segment_ids, masked_lm_labels, word_attention_mask, masked_positions) =\
            self._create_word_data(a_words, b_words, a_mask, b_mask)

        (entity_ids, entity_position_ids, entity_segment_ids, entity_attention_mask,
         entity_link_prob_ids, entity_prior_prob_ids, entity_labels) = self._create_entity_data(
             a_mentions, b_mentions, a_entity, b_entity, masked_positions)

        return dict(
            word_ids=word_ids,
            word_segment_ids=word_segment_ids,
            word_attention_mask=word_attention_mask,
            entity_ids=entity_ids,
            entity_position_ids=entity_position_ids,
            entity_segment_ids=entity_segment_ids,
            entity_attention_mask=entity_attention_mask,
            entity_link_prob_ids=entity_link_prob_ids,
            entity_prior_prob_ids=entity_prior_prob_ids,
            masked_lm_labels=masked_lm_labels,
            is_random_next=int(not is_next),
            entity_labels=entity_labels,
            a_entity_id=a_entity_id,
            b_entity_id=b_entity_id,
        )

    def _create_batch(self, data):
        return {k: np.stack([o[k] for o in data]) for k in data[0].keys()}

    def _create_word_data(self, a_words, b_words, a_mask, b_mask):
        num_to_predict = min(self._max_predictions_per_seq,
                             max(1, int(round((len(a_words) + len(b_words)) * self._masked_lm_prob))))

        masked_lm_labels = np.full(self._max_seq_length, -1, dtype=np.int)
        masked_positions = []

        input_words = [self._cls_word, self._ent_word, self._ent_word]

        for (n, word) in enumerate(a_words, 3):
            if a_mask and word.in_title:
                input_words.append(self._mask_word)
                masked_lm_labels[n] = word.id
                masked_positions.append(n)
            else:
                input_words.append(word)

        input_words.append(self._sep_word)

        for (n, word) in enumerate(b_words, len(a_words) + 4):
            if b_mask and word.in_title:
                input_words.append(self._mask_word)
                masked_lm_labels[n] = word.id
                masked_positions.append(n)
            else:
                input_words.append(word)

        input_words.append(self._sep_word)

        output_word_ids = np.zeros(self._max_seq_length, dtype=np.int)
        output_word_ids[:len(input_words)] = [w.id for w in input_words]

        for index in np.random.permutation(len(input_words)):
            if input_words[index] in (self._cls_word, self._ent_word, self._sep_word, self._mask_word):
                continue

            masked_lm_labels[index] = input_words[index].id
            p = random.random()
            if p < 0.8:
                output_word_ids[index] = self._mask_word.id
            elif p < 0.9:
                output_word_ids[index] = random.randint(0, len(self._word_vocab) - 1)

            masked_positions.append(index)
            if len(masked_positions) >= num_to_predict:
                break

        word_attention_mask = np.ones(self._max_seq_length, dtype=np.int)
        word_attention_mask[len(input_words):] = 0

        word_segment_ids = np.zeros(self._max_seq_length, dtype=np.int)
        word_segment_ids[2] = 1  # 2nd [ENT]
        word_segment_ids[len(a_words) + 4:len(input_words)] = 1

        return (output_word_ids, word_segment_ids, masked_lm_labels, word_attention_mask,
                masked_positions)

    def _create_entity_data(self, a_mentions, b_mentions, a_entity, b_entity, masked_positions):
        entity_ids = np.zeros(self._max_entity_length, dtype=np.int)
        entity_position_ids = np.zeros(self._max_entity_length, dtype=np.int)
        entity_segment_ids = np.zeros(self._max_entity_length, dtype=np.int)
        entity_link_prob_ids = np.zeros(self._max_entity_length, dtype=np.int)
        entity_prior_prob_ids = np.zeros(self._max_entity_length, dtype=np.int)
        entity_labels = np.full(self._max_entity_length, -1, dtype=np.int)
        masked_positions = frozenset(masked_positions)

        ind = 0
        for ((start, end, title, link_prob, prior_prob, label), page_entity, segment_id) in chain(
            zip(a_mentions, repeat(a_entity), repeat(0)),
            zip(b_mentions, repeat(b_entity), repeat(1))
        ):
            if self._mask_title_words and title == page_entity.title:
                continue

            entity_id = self._entity_vocab.get_id(title)
            if entity_id is None:
                continue

            if any(pos in masked_positions for pos in range(start, end)):
                continue

            for pos in range(start, end):
                if ind >= self._max_entity_length:
                    break

                entity_ids[ind] = entity_id
                entity_position_ids[ind] = pos
                entity_segment_ids[ind] = segment_id
                entity_link_prob_ids[ind] = int(link_prob * (self._link_prob_bin_size - 1))
                entity_prior_prob_ids[ind] = int(prior_prob * (self._prior_prob_bin_size - 1))
                entity_labels[ind] = label

                ind += 1

        entity_attention_mask = np.ones(self._max_entity_length, dtype=np.int)
        entity_attention_mask[ind:] = 0

        return (entity_ids, entity_position_ids, entity_segment_ids, entity_attention_mask,
                entity_link_prob_ids, entity_prior_prob_ids, entity_labels)
