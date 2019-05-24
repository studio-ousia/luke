# -*- coding: utf-8 -*-

import functools
import logging
import multiprocessing
import queue
import random
import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


class BatchGenerator(object):
    def __init__(self, tf_record_file, batch_size, max_seq_length, max_entity_length,
                 max_mention_length, use_entities, batch_buffer_size=1000):
        self._worker_cls = functools.partial(BatchWorker,
                                             tf_record_file=tf_record_file,
                                             batch_size=batch_size,
                                             max_seq_length=max_seq_length,
                                             max_entity_length=max_entity_length,
                                             max_mention_length=max_mention_length,
                                             use_entities=use_entities,
                                             batch_buffer_size=batch_buffer_size)

    def generate_batches(self, num_skip=0, queue_size=10000):
        output_queue = multiprocessing.Queue(queue_size)
        is_finished = multiprocessing.Event()

        worker = self._worker_cls(output_queue, is_finished, num_skip)
        worker.daemon = True
        worker.start()

        try:
            while True:
                try:
                    yield output_queue.get(True, 3)
                except queue.Empty:
                    if is_finished.is_set():
                        break
                    logger.info('Queue is empty')

                    if not worker.is_alive():
                        raise RuntimeError('Worker exited unexpectedly')

        finally:
            worker.terminate()
            output_queue.close()


class BatchWorker(multiprocessing.Process):
    def __init__(self, output_queue, is_finished, num_skip, tf_record_file, batch_size,
                 max_seq_length, max_entity_length, max_mention_length, use_entities,
                 batch_buffer_size):
        super(BatchWorker, self).__init__()

        self._output_queue = output_queue
        self._is_finished = is_finished
        self._num_skip = num_skip
        self._tf_record_file = tf_record_file
        self._batch_size = batch_size
        self._max_seq_length = max_seq_length
        self._max_entity_length = max_entity_length
        self._max_mention_length = max_mention_length
        self._use_entities = use_entities
        self._batch_buffer_size = batch_buffer_size

    def run(self):
        buf = []

        for item in self._read_records():
            query_word_ids = item['query_word_ids']
            query_entity_ids = item['query_entity_ids']
            query_entity_positions = item['query_entity_positions'].reshape(-1, 2)
            doc_word_ids = item['doc_word_ids']

            word_ids = np.concatenate((query_word_ids, doc_word_ids))
            output_word_ids = np.zeros(self._max_seq_length, dtype=np.int)
            output_word_ids[:word_ids.size] = word_ids

            word_attention_mask = np.ones(self._max_seq_length, dtype=np.int)
            word_attention_mask[word_ids.size:] = 0

            word_segment_ids = np.zeros(self._max_seq_length, dtype=np.int)
            word_segment_ids[query_word_ids.size:word_ids.size] = 1

            if self._use_entities:
                doc_entity_ids = item['doc_entity_ids']
                doc_entity_positions = item['doc_entity_positions'].reshape(-1, 2)

                entity_ids = np.concatenate((query_entity_ids, doc_entity_ids))
                if entity_ids.size > self._max_entity_length:
                    entity_ids = entity_ids[:self._max_entity_length]
                output_entity_ids = np.zeros(self._max_entity_length, dtype=np.int)
                output_entity_ids[:entity_ids.size] = entity_ids

                entity_attention_mask = np.ones(self._max_entity_length, dtype=np.int)
                entity_attention_mask[entity_ids.size:] = 0

                entity_segment_ids = np.zeros(self._max_entity_length, dtype=np.int)
                entity_segment_ids[query_entity_ids.size:entity_ids.size] = 1

                entity_position_ids = np.full((self._max_entity_length, self._max_mention_length), -1,
                                            dtype=np.int)
                for i, (start, end) in enumerate(np.vstack((query_entity_positions,
                                                            doc_entity_positions))):
                    if i == self._max_entity_length:
                        break
                    entity_position_ids[i][:end - start] = range(start, end)[:self._max_mention_length]

            else:
                entity_ids = output_entity_ids = np.zeros(1, dtype=np.int)
                entity_attention_mask = np.zeros(1, dtype=np.int)
                entity_segment_ids = np.zeros(1, dtype=np.int)
                entity_position_ids = np.zeros((1, self._max_mention_length), dtype=np.int)

            buf.append((word_ids.size, entity_ids.size, dict(
                word_ids=output_word_ids,
                word_attention_mask=word_attention_mask,
                word_segment_ids=word_segment_ids,
                entity_ids=output_entity_ids,
                entity_attention_mask=entity_attention_mask,
                entity_segment_ids=entity_segment_ids,
                entity_position_ids=entity_position_ids,
                label=int(item['label'])
            )))
            if len(buf) == self._batch_size * self._batch_buffer_size:
                for batch in self._create_batches(buf):
                    self._output_queue.put(batch, True)
                buf = []

        if buf:
            for batch in self._create_batches(buf):
                self._output_queue.put(batch, True)

        self._is_finished.set()

    def _read_records(self):
        features = {
            'query_word_ids': tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
            'query_entity_ids': tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
            'query_entity_positions': tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
            'doc_word_ids': tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
            'doc_entity_ids': tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
            'doc_entity_positions': tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
            'label': tf.FixedLenFeature([1], tf.int64)
        }

        dataset = tf.data.TFRecordDataset([self._tf_record_file], compression_type='GZIP')
        dataset = dataset.repeat()
        if self._num_skip > 0:
            logger.info('skipping %d records', self._num_skip)
            dataset = dataset.skip(self._num_skip)
        dataset = dataset.map(functools.partial(tf.parse_single_example, features=features))
        it = dataset.make_one_shot_iterator()
        it = it.get_next()

        with tf.Session() as sess:
            try:
                while True:
                    yield sess.run(it)
            except tf.errors.OutOfRangeError:
                pass

    def _create_batches(self, items):
        # sort items by their number of words
        items.sort(reverse=True, key=lambda o: o[0])
        batches = []

        for i in range(0, len(items), self._batch_size):
            target_items = items[i:i+self._batch_size]
            word_size = max(target_items[0][0], 1)
            entity_size = max(max(o[1] for o in target_items), 1)

            for (_, _, data) in target_items:
                for key in ('word_ids', 'word_attention_mask', 'word_segment_ids'):
                    data[key] = data[key][:word_size]

                for key in ('entity_ids', 'entity_attention_mask', 'entity_segment_ids',
                            'entity_position_ids'):
                    data[key] = data[key][:entity_size]

            batch = {k: np.stack([o[2][k] for o in target_items]) for k in target_items[0][2].keys()}
            batches.append(batch)

        random.shuffle(batches)
        return batches
