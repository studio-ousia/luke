import logging
import functools
import multiprocessing
import queue
import random
from itertools import chain, islice, repeat
import numpy as np

from luke.utils.wiki_corpus import WikiCorpus

logger = logging.getLogger(__name__)


class BasePretrainingBatchGenerator(object):
    def generate_batches(self, page_indices=None, queue_size=10000):
        output_queue = multiprocessing.Queue(queue_size)
        is_finished = multiprocessing.Event()

        worker = self._create_worker(output_queue, is_finished, page_indices)
        worker.daemon = True
        worker.start()

        try:
            while True:
                try:
                    yield output_queue.get(True, 1)
                except queue.Empty:
                    if is_finished.is_set():
                        break
                    logger.debug('Queue is empty')

                    if not worker.is_alive():
                        raise RuntimeError('Worker exited unexpectedly')

        finally:
            worker.terminate()
            output_queue.close()

    def _create_worker(self, output_queue, is_finished, page_indices):
        raise NotImplementedError()


class LukePretrainingBatchGenerator(BasePretrainingBatchGenerator):
    def __init__(self, corpus_file, entity_vocab, batch_size, max_seq_length, max_entity_length, max_mention_length,
                 short_seq_prob, masked_lm_prob, masked_entity_prob, single_sentence, batch_buffer_size=100,
                 mmap_mode=None):
        self._worker_cls = functools.partial(LukePretrainingBatchWorker,
                                             corpus_file=corpus_file,
                                             entity_vocab=entity_vocab,
                                             batch_size=batch_size,
                                             max_seq_length=max_seq_length,
                                             max_entity_length=max_entity_length,
                                             max_mention_length=max_mention_length,
                                             short_seq_prob=short_seq_prob,
                                             masked_lm_prob=masked_lm_prob,
                                             masked_entity_prob=masked_entity_prob,
                                             single_sentence=single_sentence,
                                             batch_buffer_size=batch_buffer_size,
                                             mmap_mode=mmap_mode)

    def _create_worker(self, output_queue, is_finished, page_indices):
        return self._worker_cls(output_queue, is_finished, page_indices)


class LukeE2EPretrainingBatchGenerator(BasePretrainingBatchGenerator):
    def __init__(self, corpus_file, entity_vocab, batch_size, max_seq_length, max_entity_length, max_mention_length,
                 max_candidate_length, short_seq_prob, masked_lm_prob, masked_entity_prob, single_sentence,
                 min_candidate_prior_prob, batch_buffer_size=100, mmap_mode=None):
        self._worker_cls = functools.partial(LukeE2EPretrainingBatchWorker,
                                             corpus_file=corpus_file,
                                             entity_vocab=entity_vocab,
                                             batch_size=batch_size,
                                             max_seq_length=max_seq_length,
                                             max_entity_length=max_entity_length,
                                             max_mention_length=max_mention_length,
                                             max_candidate_length=max_candidate_length,
                                             short_seq_prob=short_seq_prob,
                                             masked_lm_prob=masked_lm_prob,
                                             masked_entity_prob=masked_entity_prob,
                                             single_sentence=single_sentence,
                                             min_candidate_prior_prob=min_candidate_prior_prob,
                                             batch_buffer_size=batch_buffer_size,
                                             mmap_mode=mmap_mode)

    def _create_worker(self, output_queue, is_finished, page_indices):
        return self._worker_cls(output_queue, is_finished, page_indices)


class BaseBatchWorker(multiprocessing.Process):
    def __init__(self, output_queue, is_finished, page_indices, corpus_file, batch_size, max_seq_length, short_seq_prob,
                 masked_lm_prob, single_sentence, batch_buffer_size, mmap_mode):
        super(BaseBatchWorker, self).__init__()

        self._output_queue = output_queue
        self._is_finished = is_finished
        self._page_indices = page_indices
        self._corpus_file = corpus_file
        self._batch_size = batch_size
        self._max_seq_length = max_seq_length
        self._short_seq_prob = short_seq_prob
        self._masked_lm_prob = masked_lm_prob
        self._single_sentence = single_sentence
        self._batch_buffer_size = batch_buffer_size
        self._mmap_mode = mmap_mode

        if single_sentence:
            self._max_num_tokens = max_seq_length - 2  # 2 for [CLS] and [SEP]
        else:
            self._max_num_tokens = max_seq_length - 3  # 3 for [CLS], [SEP], and [SEP]

    def run(self):
        # WikiCorpus needs to be initialized here because the BatchWorker class is pickled when the worker starts, and
        # the WikiCorpus instance is too large to be pickled
        self._corpus = WikiCorpus(self._corpus_file, self._mmap_mode)
        self._word_vocab = self._corpus.tokenizer.vocab

        self._cls_id = self._word_vocab['[CLS]']
        self._sep_id = self._word_vocab['[SEP]']
        self._mask_id = self._word_vocab['[MASK]']

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
                sent = sent_stack.pop()
                chunk_sents.append(sent)
                chunk_word_length += len(sent.words)

                if not sent_stack or chunk_word_length + len(sent_stack[-1].words) >= target_seq_length:
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

                    item = self._create_batch_item(a_sents, b_sents, is_next, target_seq_length)
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

    def _create_batch_item(self, a_sents, b_sents, is_next, target_seq_length):
        if b_sents is None:
            a_orig_length = sum([len(s.words) for s in a_sents])
            a_length = min(a_orig_length, target_seq_length)
            a_left_num_trunc = random.randint(0, a_orig_length - a_length)
            b_length = 0
            b_left_num_trunc = 0

            a_words = [w for s in a_sents for w in s.words][a_left_num_trunc:a_left_num_trunc + a_length]
            b_words = None

        else:
            a_orig_length = sum([len(s.words) for s in a_sents])
            b_orig_length = sum([len(s.words) for s in b_sents])

            half_target_length = round(target_seq_length / 2)
            if a_orig_length < b_orig_length:
                a_length = min(half_target_length, a_orig_length)
                b_length = min(target_seq_length - a_length, b_orig_length)
            else:
                b_length = min(half_target_length, b_orig_length)
                a_length = min(target_seq_length - b_length, a_orig_length)
            a_left_num_trunc = random.randint(0, a_orig_length - a_length)
            b_left_num_trunc = random.randint(0, b_orig_length - b_length)

            a_words = [w for s in a_sents for w in s.words][a_left_num_trunc:a_left_num_trunc + a_length]
            b_words = [w for s in b_sents for w in s.words][b_left_num_trunc:b_left_num_trunc + b_length]

        a_links = []
        offset = -a_left_num_trunc
        for sent in a_sents:
            for link in sent.links:
                if link.start + offset >= 0 and link.end + offset <= a_length:
                    link.start += offset + 1  # 1 for [CLS]
                    link.end += offset + 1
                    a_links.append(link)
            offset += len(sent.words)

        b_links = []
        if b_sents is not None:
            offset = -b_left_num_trunc
            for sent in b_sents:
                for link in sent.links:
                    if link.start + offset >= 0 and link.end + offset <= b_length:
                        link.start += offset + a_length + 2  # 2 for [CLS] and [SEP]
                        link.end += offset + a_length + 2
                        b_links.append(link)
                offset += len(sent.words)

        word_inputs = self._create_word_inputs(a_words, b_words)
        entity_inputs = self._create_entity_inputs(a_links, b_links)

        entity_size = np.sum(entity_inputs['entity_attention_mask'])
        if entity_size == 0:
            return None

        return (entity_size, word_inputs, entity_inputs, is_next)

    def _create_batches(self, items):
        items.sort(reverse=True, key=lambda o: o[0])

        buf = []
        current_entity_size = None
        for (entity_size, word_data, entity_data, is_next) in items:
            if current_entity_size is None:
                current_entity_size = max(entity_size, 1)

            entity_data = {k: v[:current_entity_size] for (k, v) in entity_data.items()}

            item = word_data
            item.update(entity_data)
            if not self._single_sentence:
                item['is_random_next'] = int(not is_next)
            buf.append(item)

            if len(buf) % self._batch_size == 0:
                current_entity_size = None

        batches = []
        for i in range(0, len(buf), self._batch_size):
            batches.append({k: np.stack([o[k] for o in buf[i:i + self._batch_size]]) for k in buf[0].keys()})

        random.shuffle(batches)

        return batches

    def _create_word_inputs(self, a_words, b_words):
        word_ids = [self._cls_id] + [w.id for w in a_words] + [self._sep_id]
        word_len = len(a_words)
        if b_words is not None:
            word_ids += [w.id for w in b_words] + [self._sep_id]
            word_len += len(b_words)

        ret = {}
        if self._masked_lm_prob != 0.0:
            masked_lm_labels = np.full(self._max_seq_length, -1, dtype=np.int)
            ret['masked_lm_labels'] = masked_lm_labels

            num_to_predict = max(1, int(round(word_len * self._masked_lm_prob)))

            for index in np.random.permutation(len(word_ids)):
                if word_ids[index] in (self._cls_id, self._sep_id):
                    continue

                masked_lm_labels[index] = word_ids[index]
                p = random.random()
                if p < 0.8:
                    word_ids[index] = self._mask_id
                elif p < 0.9:
                    word_ids[index] = random.randint(0, len(self._word_vocab) - 1)

                num_to_predict -= 1
                if num_to_predict == 0:
                    break

        ret['word_ids'] = np.zeros(self._max_seq_length, dtype=np.int)
        ret['word_ids'][:len(word_ids)] = word_ids

        ret['word_attention_mask'] = np.ones(self._max_seq_length, dtype=np.int)
        ret['word_attention_mask'][len(word_ids):] = 0

        ret['word_segment_ids'] = np.zeros(self._max_seq_length, dtype=np.int)
        if b_words is not None:
            ret['word_segment_ids'][len(a_words) + 2:len(word_ids)] = 1  # 2 for [CLS] and [SEP]

        return ret

    def _create_entity_inputs(self, a_links, b_links):
        raise NotImplementedError()


class LukePretrainingBatchWorker(BaseBatchWorker):
    def __init__(self, output_queue, is_finished, page_indices, corpus_file, entity_vocab, batch_size, max_seq_length,
                 max_entity_length, max_mention_length, short_seq_prob, masked_lm_prob, masked_entity_prob,
                 single_sentence, batch_buffer_size, mmap_mode):
        super(LukePretrainingBatchWorker, self).__init__(
            output_queue, is_finished, page_indices, corpus_file, batch_size, max_seq_length, short_seq_prob,
            masked_lm_prob, single_sentence, batch_buffer_size, mmap_mode)

        self._entity_vocab = entity_vocab
        self._max_entity_length = max_entity_length
        self._max_mention_length = max_mention_length
        self._masked_entity_prob = masked_entity_prob

        self._entity_mask_id = self._entity_vocab['[MASK]']

    def _create_entity_inputs(self, a_links, b_links):
        a_links = [link for link in a_links if link.title in self._entity_vocab]
        b_links = [link for link in b_links if link.title in self._entity_vocab]
        entity_len = len(a_links) + len(b_links)

        if self._masked_entity_prob != 0.0:
            num_to_predict = max(1, int(round(entity_len * self._masked_entity_prob)))
            mask_indices = frozenset(np.random.permutation(range(entity_len))[:num_to_predict])
            masked_entity_labels = np.full(self._max_entity_length, -1, dtype=np.int)
        else:
            mask_indices = frozenset()
            masked_entity_labels = None

        entity_ids = np.zeros(self._max_entity_length, dtype=np.int)
        entity_segment_ids = np.zeros(self._max_entity_length, dtype=np.int)
        entity_position_ids = np.full((self._max_entity_length, self._max_mention_length), -1, dtype=np.int)

        for (index, (link, segment_id)) in enumerate(islice(chain(zip(a_links, repeat(0)), zip(b_links, repeat(1))),
                                                            self._max_entity_length)):
            entity_id = self._entity_vocab.get_id(link.title)

            if index in mask_indices:
                entity_ids[index] = self._entity_mask_id
                masked_entity_labels[index] = entity_id
            else:
                entity_ids[index] = entity_id

            entity_segment_ids[index] = segment_id
            mention_len = min(self._max_mention_length, link.end - link.start)
            entity_position_ids[index][:mention_len] = range(link.start, link.start + mention_len)

        entity_attention_mask = np.ones(self._max_entity_length, dtype=np.int)
        entity_attention_mask[entity_len:] = 0

        ret = dict(
            entity_ids=entity_ids,
            entity_position_ids=entity_position_ids,
            entity_segment_ids=entity_segment_ids,
            entity_attention_mask=entity_attention_mask,
        )
        if masked_entity_labels is not None:
            ret['masked_entity_labels'] = masked_entity_labels

        return ret


class LukeE2EPretrainingBatchWorker(BaseBatchWorker):
    def __init__(self, output_queue, is_finished, page_indices, corpus_file, entity_vocab, batch_size,
                 max_seq_length, max_entity_length, max_mention_length, max_candidate_length, short_seq_prob,
                 masked_lm_prob, masked_entity_prob, single_sentence, min_candidate_prior_prob, batch_buffer_size,
                 mmap_mode):
        super(LukeE2EPretrainingBatchWorker, self).__init__(
            output_queue, is_finished, page_indices, corpus_file, batch_size, max_seq_length, short_seq_prob,
            masked_lm_prob, single_sentence, batch_buffer_size, mmap_mode)

        self._entity_vocab = entity_vocab
        self._max_entity_length = max_entity_length
        self._max_mention_length = max_mention_length
        self._max_candidate_length = max_candidate_length
        self._masked_entity_prob = masked_entity_prob
        self._min_candidate_prior_prob = min_candidate_prior_prob

        self._entity_mask_id = self._entity_vocab['[MASK]']

    def _create_entity_inputs(self, a_links, b_links):
        for link in (a_links + b_links):
            link.candidates = [c for c in link.candidates if c.prior_prob >= self._min_candidate_prior_prob]

        a_links = [link for link in a_links if link.title in self._entity_vocab and link.candidates]
        b_links = [link for link in b_links if link.title in self._entity_vocab and link.candidates]
        entity_len = len(a_links) + len(b_links)

        if self._masked_entity_prob != 0.0:
            num_to_predict = max(1, int(round(entity_len * self._masked_entity_prob)))
            mask_indices = frozenset(np.random.permutation(range(entity_len))[:num_to_predict])
            masked_entity_labels = np.full(self._max_entity_length, -1, dtype=np.int)
        else:
            mask_indices = frozenset()
            masked_entity_labels = None

        entity_candidate_ids = np.zeros((self._max_entity_length, self._max_candidate_length + 1), dtype=np.int)
        entity_segment_ids = np.zeros(self._max_entity_length, dtype=np.int)
        entity_position_ids = np.full((self._max_entity_length, self._max_mention_length), -1, dtype=np.int)
        entity_candidate_labels = np.full(self._max_entity_length, -1, dtype=np.int)

        for (index, (link, segment_id)) in enumerate(islice(chain(zip(a_links, repeat(0)), zip(b_links, repeat(1))),
                                                            self._max_entity_length)):
            entity_candidate_ids[index, 0] = 1  # [UNK]
            entity_candidate_labels[index] = 0  # [UNK]
            candidate_index = 1
            for candidate in link.candidates:
                if candidate.prior_prob >= self._min_candidate_prior_prob and candidate.title in self._entity_vocab:
                    entity_candidate_ids[index, candidate_index] = self._entity_vocab[candidate.title]
                    if candidate.title == link.title:
                        entity_candidate_labels[index] = candidate_index

                    candidate_index += 1
                    if candidate_index == self._max_candidate_length + 1:
                        break

            entity_segment_ids[index] = segment_id

            mention_len = min(self._max_mention_length, link.end - link.start)
            entity_position_ids[index][:mention_len] = range(link.start, link.start + mention_len)

            if index in mask_indices:
                masked_entity_labels[index] = self._entity_vocab[link.title]

        entity_attention_mask = np.ones(self._max_entity_length, dtype=np.int)
        entity_attention_mask[entity_len:] = 0

        ret = dict(
            entity_candidate_ids=entity_candidate_ids,
            entity_position_ids=entity_position_ids,
            entity_segment_ids=entity_segment_ids,
            entity_attention_mask=entity_attention_mask,
            entity_candidate_labels=entity_candidate_labels,
        )

        if self._masked_entity_prob != 0.0:
            ret['masked_entity_labels'] = masked_entity_labels

        return ret
