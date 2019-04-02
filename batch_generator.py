# -*- coding: utf-8 -*-

import logging
import functools
import multiprocessing
import queue
import random
from itertools import chain, repeat
import numpy as np

from wiki_corpus import WikiCorpus

logger = logging.getLogger(__name__)


class LukeBatchGenerator(object):
    def __init__(self, corpus_data_file, entity_vocab, batch_size, max_seq_length,
                 max_entity_length, max_mention_length, short_seq_prob, masked_lm_prob,
                 max_predictions_per_seq, masked_entity_prob, max_entity_predictions_per_seq,
                 single_sentence, single_token_per_mention, mmap=True, batch_buffer_size=1000):
        self._worker_cls = functools.partial(BatchWorker,
            target_entity_annotation='link',
            corpus_data_file=corpus_data_file,
            entity_vocab=entity_vocab,
            batch_size=batch_size,
            max_seq_length=max_seq_length,
            max_entity_length=max_entity_length,
            short_seq_prob=short_seq_prob,
            masked_lm_prob=masked_lm_prob,
            max_predictions_per_seq=max_predictions_per_seq,
            masked_entity_prob=masked_entity_prob,
            max_entity_predictions_per_seq=max_entity_predictions_per_seq,
            single_sentence=single_sentence,
            single_token_per_mention=single_token_per_mention,
            max_mention_length=max_mention_length,
            mmap=mmap,
            batch_buffer_size=batch_buffer_size,
            link_prob_bin_size=0,
            prior_prob_bin_size=0)

    def generate_batches(self, page_indices=None, queue_size=100):
        for batch in _generate_batches(self._worker_cls, page_indices, queue_size):
            yield batch


class LukeE2EBatchGenerator(object):
    def __init__(self, corpus_data_file, entity_vocab, batch_size, max_seq_length,
                 max_entity_length, max_mention_length, short_seq_prob, masked_lm_prob,
                 max_predictions_per_seq, single_sentence, single_token_per_mention,
                 link_prob_bin_size, prior_prob_bin_size, mmap=True, batch_buffer_size=1000):
        self._worker_cls = functools.partial(BatchWorker,
            target_entity_annotation='mention',
            corpus_data_file=corpus_data_file,
            entity_vocab=entity_vocab,
            batch_size=batch_size,
            max_seq_length=max_seq_length,
            max_entity_length=max_entity_length,
            short_seq_prob=short_seq_prob,
            masked_lm_prob=masked_lm_prob,
            max_predictions_per_seq=max_predictions_per_seq,
            single_sentence=single_sentence,
            single_token_per_mention=single_token_per_mention,
            max_mention_length=max_mention_length,
            link_prob_bin_size=link_prob_bin_size,
            prior_prob_bin_size=prior_prob_bin_size,
            mmap=mmap,
            batch_buffer_size=batch_buffer_size,
            masked_entity_prob=0.0,
            max_entity_predictions_per_seq=0)

    def generate_batches(self, page_indices=None, queue_size=100):
        for batch in _generate_batches(self._worker_cls, page_indices, queue_size):
            yield batch


def _generate_batches(worker_cls, page_indices, queue_size):
    output_queue = multiprocessing.Queue(queue_size)
    is_finished = multiprocessing.Event()

    worker = worker_cls(output_queue, is_finished, page_indices)
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
    def __init__(self, output_queue, is_finished, page_indices, target_entity_annotation,
                 corpus_data_file, entity_vocab, batch_size, max_seq_length,
                 max_entity_length, short_seq_prob, masked_lm_prob, max_predictions_per_seq,
                 masked_entity_prob, max_entity_predictions_per_seq, single_sentence,
                 single_token_per_mention, max_mention_length, link_prob_bin_size,
                 prior_prob_bin_size, mmap, batch_buffer_size):
        super(BatchWorker, self).__init__()

        self._output_queue = output_queue
        self._is_finished = is_finished
        self._page_indices = page_indices
        self._target_entity_annotation = target_entity_annotation
        self._corpus_data_file = corpus_data_file
        self._entity_vocab = entity_vocab
        self._batch_size = batch_size
        self._max_seq_length = max_seq_length
        self._max_entity_length = max_entity_length
        self._short_seq_prob = short_seq_prob
        self._masked_lm_prob = masked_lm_prob
        self._max_predictions_per_seq = max_predictions_per_seq
        self._masked_entity_prob = masked_entity_prob
        self._max_entity_predictions_per_seq = max_entity_predictions_per_seq
        self._single_sentence = single_sentence
        self._single_token_per_mention = single_token_per_mention
        self._max_mention_length = max_mention_length
        self._link_prob_bin_size = link_prob_bin_size
        self._prior_prob_bin_size = prior_prob_bin_size
        self._batch_buffer_size = batch_buffer_size

        self._mmap_mode = None
        if mmap:
            self._mmap_mode = 'r'

        if single_sentence:
            self._max_num_tokens = max_seq_length - 2  # 2 for CLS and SEP
        else:
            self._max_num_tokens = max_seq_length - 3  # 3 for CLS, SEP, and SEP

    def run(self):
        # WikiCorpus needs to be initialized here because the BatchWorker class is pickled when the
        # worker starts, and the WikiCorpus instance is too large to be pickled
        self._corpus = WikiCorpus(self._corpus_data_file, self._mmap_mode)
        self._word_vocab = self._corpus.word_vocab

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

                if not sent_stack or\
                    chunk_word_length + len(sent_stack[-1].words) >= target_seq_length:
                    if self._single_sentence:
                        a_sents = chunk_sents
                        b_sents = None
                        is_next = None

                    else:
                        a_end = 1
                        if len(chunk_sents) > 1:
                            a_end = random.randint(1, len(chunk_sents) - 1)
                        a_sents = chunk_sents[:a_end]

                        if len(chunk_sents) == 1 or random.random() < 0.5:
                            is_next = False
                            a_length = sum([len(s.words) for s in a_sents])
                            target_b_length = target_seq_length - a_length

                            random_sents = self._corpus.read_random_page().sentences
                            random_start = random.randint(0, len(random_sents) - 1)
                            b_sents = []
                            b_length = 0
                            for (n, sent) in enumerate(random_sents[random_start:]):
                                b_length += len(sent.words)
                                if n != 0 and b_length >= target_b_length:
                                    break
                                b_sents.append(sent)

                            # put back the unused sentences to the list
                            sent_stack.extend(reversed(chunk_sents[a_end:]))

                        else:  # actual next
                            is_next = True
                            b_sents = chunk_sents[a_end:]

                    item = self._create_item(a_sents, b_sents, is_next, target_seq_length)
                    if item is not None:
                        buf.append(item)

                        if len(buf) == self._batch_size * self._batch_buffer_size:
                            for batch in self._create_batches(buf):
                                self._output_queue.put(batch, True)
                            buf = []

                    chunk_sents = []
                    chunk_word_length = 0

            if buf and i == total_pages - 1:
                for batch in self._create_batches(buf):
                    self._output_queue.put(batch, True)

        self._is_finished.set()

    def _create_item(self, a_sents, b_sents, is_next, target_seq_length):
        if b_sents is None:
            a_orig_len = sum([len(s.words) for s in a_sents])
            a_len = min(a_orig_len, target_seq_length)
            a_left_num_trunc = random.randint(0, a_orig_len - a_len)

        else:
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
        a_annotations = []
        for sent in a_sents:
            ofs = len(a_words) - a_left_num_trunc
            a_words.extend(sent.words)

            if self._target_entity_annotation == 'link':
                annotations = sent.links
            else:
                annotations = sent.mentions

            for annotation in annotations:
                annotation.start += ofs
                annotation.end += ofs
                if annotation.start < 0 or annotation.end > a_len:
                    continue
                a_annotations.append(annotation)

        a_words = a_words[a_left_num_trunc:a_left_num_trunc + a_len]

        if b_sents is None:
            b_words = None
            b_annotations = None
        else:
            b_words = []
            b_annotations = []

            for sent in b_sents:
                ofs = len(b_words) - b_left_num_trunc
                b_words.extend(sent.words)

                if self._target_entity_annotation == 'link':
                    annotations = sent.links
                else:
                    annotations = sent.mentions

                for annotation in annotations:
                    annotation.start += ofs
                    annotation.end += ofs
                    if annotation.start < 0 or annotation.end > b_len:
                        continue
                    b_annotations.append(annotation)

            b_words = b_words[b_left_num_trunc:b_left_num_trunc + b_len]

        word_data = create_word_data(a_words, b_words, self._word_vocab, self._max_seq_length,
            self._masked_lm_prob, self._max_predictions_per_seq)

        if self._target_entity_annotation == 'link':
            entity_data = create_link_data(a_annotations, b_annotations, len(a_words),
                self._entity_vocab, self._max_entity_length, self._masked_entity_prob,
                self._max_entity_predictions_per_seq, self._single_token_per_mention,
                self._max_mention_length)
        else:
            entity_data = create_mention_data(a_annotations, b_annotations, len(a_words),
                self._entity_vocab, self._max_entity_length, self._single_token_per_mention,
                self._max_mention_length, self._link_prob_bin_size, self._prior_prob_bin_size)

        entity_size = np.sum(entity_data['entity_attention_mask'])
        if entity_size == 0:
            return None

        return (entity_size, word_data, entity_data, is_next)

    def _create_batches(self, items):
        items.sort(reverse=True, key=lambda o: o[0])

        buf = []
        current_entity_size = None
        for (entity_size, word_data, entity_data, is_next) in items:
            if current_entity_size is None:
                current_entity_size = entity_size

            entity_data = {k: v[:current_entity_size] for (k, v) in entity_data.items()}

            item = word_data
            item.update(entity_data)
            if not self._single_sentence:
                item['is_random_next'] = int(not is_next)
            buf.append(item)

            if len(buf) == self._batch_size:
                yield {k: np.stack([o[k] for o in buf]) for k in buf[0].keys()}
                buf = []
                current_entity_size = None

        if buf:
            yield {k: np.stack([o[k] for o in buf]) for k in buf[0].keys()}


def create_word_data(a_words, b_words, word_vocab, max_seq_length, masked_lm_prob=0.0,
                     max_predictions_per_seq=0):
    cls_id = word_vocab['[CLS]']
    sep_id = word_vocab['[SEP]']
    mask_id = word_vocab['[MASK]']

    word_ids = [cls_id]
    word_ids += [w.id for w in a_words]
    word_ids.append(sep_id)
    word_len = len(a_words)
    if b_words is not None:
        word_ids += [w.id for w in b_words]
        word_ids.append(sep_id)
        word_len += len(b_words)

    ret = {}

    if max_predictions_per_seq > 0:
        masked_lm_labels = np.full(max_seq_length, -1, dtype=np.int)
        ret['masked_lm_labels'] = masked_lm_labels

        num_to_predict = min(max_predictions_per_seq, max(1, int(round(word_len * masked_lm_prob))))

        for index in np.random.permutation(len(word_ids)):
            if word_ids[index] in (cls_id, sep_id):
                continue

            masked_lm_labels[index] = word_ids[index]
            p = random.random()
            if p < 0.8:
                word_ids[index] = mask_id
            elif p < 0.9:
                word_ids[index] = random.randint(0, len(word_vocab) - 1)

            num_to_predict -= 1
            if num_to_predict == 0:
                break

    output_word_ids = np.zeros(max_seq_length, dtype=np.int)
    output_word_ids[:len(word_ids)] = word_ids
    ret['word_ids'] = output_word_ids

    word_attention_mask = np.ones(max_seq_length, dtype=np.int)
    word_attention_mask[len(word_ids):] = 0
    ret['word_attention_mask'] = word_attention_mask

    word_segment_ids = np.zeros(max_seq_length, dtype=np.int)
    if b_words is not None:
        word_segment_ids[len(a_words) + 2:len(word_ids)] = 1  # 2 for CLS and SEP
    ret['word_segment_ids'] = word_segment_ids

    return ret


def create_link_data(a_links, b_links, a_word_length, entity_vocab, max_entity_length,
                     masked_entity_prob, max_entity_predictions_per_seq, single_token_per_mention,
                     max_mention_length):
    entity_ids = np.zeros(max_entity_length, dtype=np.int)
    entity_segment_ids = np.zeros(max_entity_length, dtype=np.int)
    entity_attention_mask = np.ones(max_entity_length, dtype=np.int)
    masked_entity_labels = np.full(max_entity_length, -1, dtype=np.int)

    if single_token_per_mention:
        entity_position_ids = np.full((max_entity_length, max_mention_length), -1, dtype=np.int)
    else:
        entity_position_ids = np.zeros(max_entity_length, dtype=np.int)

    mask_id = entity_vocab['[MASK]']

    for link in a_links:
        link.start += 1  # 1 for CLS
        link.end += 1
    a_links = [l for l in a_links if l.title in entity_vocab]

    if b_links is None:
        b_links = []
    else:
        for link in b_links:
            link.start += 2 + a_word_length  # 2 for CLS and SEP
            link.end += 2 + a_word_length
        b_links = [l for l in b_links if l.title in entity_vocab]

    entity_len = len(a_links) + len(b_links)
    num_to_predict = min(max_entity_predictions_per_seq,
                         max(1, int(round(entity_len * masked_entity_prob))))
    mask_indices = frozenset(np.random.permutation(range(entity_len))[:num_to_predict])

    index = 0
    for (link_index, (link, segment_id)) in enumerate(chain(zip(a_links, repeat(0)),
                                                            zip(b_links, repeat(1)))):
        entity_id = entity_vocab.get_id(link.title)
        if single_token_per_mention:
            if index >= max_entity_length:
                break

            if link_index in mask_indices:
                entity_ids[index] = mask_id
                masked_entity_labels[index] = entity_id
            else:
                entity_ids[index] = entity_id

            entity_segment_ids[index] = segment_id
            mention_len = min(max_mention_length, link.end - link.start)
            entity_position_ids[index][:mention_len] = range(link.start, link.start + mention_len)

            index += 1

        else:
            for pos in range(link.start, link.end):
                if index >= max_entity_length:
                    break

                if link_index in mask_indices:
                    entity_ids[index] = mask_id
                    masked_entity_labels[index] = entity_id
                else:
                    entity_ids[index] = entity_id

                entity_segment_ids[index] = segment_id
                entity_position_ids[index] = pos

                index += 1

    entity_attention_mask[index:] = 0

    return dict(
        entity_ids=entity_ids,
        entity_position_ids=entity_position_ids,
        entity_segment_ids=entity_segment_ids,
        entity_attention_mask=entity_attention_mask,
        masked_entity_labels=masked_entity_labels
    )


def create_mention_data(a_mentions, b_mentions, a_word_length, entity_vocab, max_entity_length,
                        single_token_per_mention, max_mention_length, link_prob_bin_size,
                        prior_prob_bin_size):
    entity_ids = np.zeros(max_entity_length, dtype=np.int)
    entity_segment_ids = np.zeros(max_entity_length, dtype=np.int)
    entity_labels = np.full(max_entity_length, -1, dtype=np.int)
    entity_link_prob_ids = np.zeros(max_entity_length, dtype=np.int)
    entity_prior_prob_ids = np.zeros(max_entity_length, dtype=np.int)

    if single_token_per_mention:
        entity_position_ids = np.full((max_entity_length, max_mention_length), -1, dtype=np.int)
    else:
        entity_position_ids = np.zeros(max_entity_length, dtype=np.int)

    for mention in a_mentions:
        mention.start += 1  # 1 for CLS
        mention.end += 1
    a_mentions = [m for m in a_mentions if m.title in entity_vocab]

    if b_mentions is None:
        b_mentions = []

    else:
        for mention in b_mentions:
            mention.start += 2 + a_word_length  # 2 for CLS and SEP
            mention.end += 2 + a_word_length
        b_mentions = [m for m in b_mentions if m.title in entity_vocab]

    index = 0
    for (mention, segment_id) in chain(zip(a_mentions, repeat(0)), zip(b_mentions, repeat(1))):
        entity_id = entity_vocab.get_id(mention.title)
        if single_token_per_mention:
            if index >= max_entity_length:
                break

            entity_ids[index] = entity_id
            entity_segment_ids[index] = segment_id

            mention_len = min(max_mention_length, mention.end - mention.start)
            entity_position_ids[index][:mention_len] = range(mention.start, mention.start + mention_len)
            entity_link_prob_ids[index] = int(mention.link_prob * (link_prob_bin_size - 1))
            entity_prior_prob_ids[index] = int(mention.prior_prob * (prior_prob_bin_size - 1))
            entity_labels[index] = mention.label

            index += 1

        else:
            for pos in range(mention.start, mention.end):
                if index >= max_entity_length:
                    break

                entity_ids[index] = entity_id
                entity_segment_ids[index] = segment_id
                entity_position_ids[index] = pos
                entity_link_prob_ids[index] = int(mention.link_prob * (link_prob_bin_size - 1))
                entity_prior_prob_ids[index] = int(mention.prior_prob * (prior_prob_bin_size - 1))
                entity_labels[index] = mention.label

                index += 1

    entity_attention_mask = np.ones(max_entity_length, dtype=np.int)
    entity_attention_mask[index:] = 0

    return dict(
        entity_ids=entity_ids,
        entity_position_ids=entity_position_ids,
        entity_segment_ids=entity_segment_ids,
        entity_attention_mask=entity_attention_mask,
        entity_link_prob_ids=entity_link_prob_ids,
        entity_prior_prob_ids=entity_prior_prob_ids,
        entity_labels=entity_labels
    )
