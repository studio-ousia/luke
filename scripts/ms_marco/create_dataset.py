"""
This code converts MS MARCO train, dev and eval tsv data into the tfrecord files
that will be consumed by BERT.
Based on the code obtained from: https://github.com/nyu-dl/dl4marco-bert/blob/fcd3bf371870fcfa9697e092c80329d5e713865b/convert_msmarco_to_tfrecord.py
"""

import collections
import functools
import itertools
import logging
import multiprocessing
import unicodedata
import os
import click
import numpy as np
import tensorflow as tf
from tensorflow.train import Int64List
from tensorflow.python_io import TFRecordWriter
from tqdm import tqdm
from luke.utils import clean_text
from luke.utils.vocab import WordPieceVocab, EntityVocab
from luke.utils.word_tokenizer import WordPieceTokenizer
from luke.utils.entity_linker import MentionDB, EntityLinker

logger = logging.getLogger(__name__)


@click.command()
@click.argument('word_vocab_file', type=click.Path(exists=True))
@click.argument('entity_vocab_file', type=click.Path(exists=True))
@click.argument('mention_db_file', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path())
@click.option('--data-dir', type=click.Path(exists=True), default='data/ms-marco')
@click.option('--uncased/--cased', default=True)
@click.option('--min-prior-prob', default=0.1)
@click.option('--max-seq-length', default=512)
@click.option('--max-query-length', default=64)
@click.option('--num-eval-docs', default=1000)
@click.option('--pool-size', default=multiprocessing.cpu_count())
def main(word_vocab_file, entity_vocab_file, mention_db_file, output_dir, data_dir, uncased,
         min_prior_prob, max_seq_length, max_query_length, num_eval_docs, pool_size):
    log_format = '[%(asctime)s] [%(levelname)s] %(message)s (%(funcName)s@%(filename)s:%(lineno)s)'
    logging.basicConfig(level=logging.INFO, format=log_format)

    word_vocab = WordPieceVocab(word_vocab_file)
    entity_vocab = EntityVocab(entity_vocab_file)

    tokenizer = WordPieceTokenizer(word_vocab, lowercase=uncased)
    mention_db = MentionDB.load(mention_db_file)
    entity_linker = EntityLinker(mention_db, min_prior_prob)

    os.makedirs(output_dir, exist_ok=True)

    train_dataset_path = os.path.join(data_dir, 'triples.train.small.tsv')
    convert_train_dataset(output_dir, train_dataset_path, tokenizer, entity_linker,
                          entity_vocab, max_seq_length, max_query_length, pool_size)

    dev_dataset_path = os.path.join(data_dir, 'top1000.dev.tsv')
    dev_qrels_path = os.path.join(data_dir, 'qrels.dev.small.tsv')
    convert_eval_dataset('dev', output_dir, dev_dataset_path, tokenizer, entity_linker,
                         entity_vocab, max_seq_length, max_query_length, num_eval_docs, pool_size, dev_qrels_path)

    eval_dataset_path = os.path.join(data_dir, 'top1000.eval.tsv')
    convert_eval_dataset('eval', output_dir, eval_dataset_path, tokenizer, entity_linker,
                         entity_vocab, max_seq_length, max_query_length, num_eval_docs, pool_size)


def convert_train_dataset(output_dir, train_dataset_path, tokenizer, entity_linker, entity_vocab,
                          max_seq_length, max_query_length, pool_size):
    logger.info('Converting the train dataset to tfrecord...')

    num_lines = sum(1 for line in open(train_dataset_path, 'r'))

    def task_generator():
        with open(train_dataset_path, 'r') as f:
            for line in f:
                line = unicodedata.normalize('NFKC', line)
                try:
                    (query, positive_doc, negative_doc) = line.rstrip().split('\t')
                except:
                    logger.warn('Invalid line: %s', line)
                    continue

                yield (query, [positive_doc, negative_doc], [1, 0])

    func = functools.partial(create_record, max_seq_length=max_seq_length,
                             max_query_length=max_query_length)

    output_file = os.path.join(output_dir, 'dataset_train.tf')
    # options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
    with TFRecordWriter(output_file) as writer:
        with tqdm(total=num_lines) as pbar:
            with multiprocessing.pool.Pool(pool_size, initializer=init_worker,
                                           initargs=(tokenizer, entity_linker, entity_vocab)) as pool:
                for items in pool.imap(func, task_generator()):
                    for item in items:
                        writer.write(item[0])

                    pbar.update(1)


def convert_eval_dataset(set_name, output_dir, dataset_path, tokenizer, entity_linker, entity_vocab,
                         max_seq_length, max_query_length, num_eval_docs, pool_size, qrels_path=None):
    if qrels_path:
        relevant_pairs = set()
        with open(qrels_path) as f:
            for line in f:
                (query_id, _, doc_id, _) = line.strip().split('\t')
                relevant_pairs.add((query_id, doc_id))

    queries_docs = collections.defaultdict(list)
    query_ids = {}
    with open(dataset_path, 'r') as f:
        for line in f:
            line = unicodedata.normalize('NFKC', line)
            try:
                (query_id, doc_id, query, doc) = line.strip().split('\t')
            except:
                logger.warn('Invalid line: %s', line)
                continue

            label = 0
            if qrels_path:
                if (query_id, doc_id) in relevant_pairs:
                    label = 1
            queries_docs[query].append((doc_id, doc, label))
            query_ids[query] = query_id

    # Add fake paragraphs to the queries that have less than FLAGS.num_eval_docs.
    queries = list(queries_docs.keys())  # Need to copy keys before iterating.
    for query in queries:
        docs = queries_docs[query]
        docs += max(0, num_eval_docs - len(docs)) * [('00000000', 'FAKE DOCUMENT', 0)]
        queries_docs[query] = docs
        assert len(docs) == num_eval_docs

    query_doc_ids_path = os.path.join(output_dir, 'query_doc_ids_' + set_name + '.txt')

    def task_generator():
        for (query, doc_ids_docs) in queries_docs.items():
            (doc_ids, docs, labels) = zip(*doc_ids_docs)
            query_id = query_ids[query]
            yield (query, docs, labels, query_id, doc_ids)

    func = functools.partial(create_record, max_seq_length=max_seq_length,
                             max_query_length=max_query_length)

    with TFRecordWriter(os.path.join(output_dir, 'dataset_' + set_name + '.tf')) as writer:
    # options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
    # with TFRecordWriter(os.path.join(output_dir, 'dataset_' + set_name + '.tf'),
    #                     options=options) as writer:
        with open(query_doc_ids_path, 'w') as ids_file:
            with tqdm(total=len(queries_docs)) as pbar:
                with multiprocessing.pool.Pool(pool_size, initializer=init_worker,
                                               initargs=(tokenizer, entity_linker, entity_vocab)) as pool:
                    for items in pool.imap(func, task_generator()):
                        for item in items:
                            writer.write(item[0])
                            ids_file.write(item[1] + '\n')

                        pbar.update(1)


_tokenizer = None
_entity_linker = None
_entity_vocab = None


def init_worker(tokenizer, entity_linker, entity_vocab):
    global _tokenizer, _entity_linker, _entity_vocab

    _tokenizer = tokenizer
    _entity_linker = entity_linker
    _entity_vocab = entity_vocab


def create_record(args, max_seq_length, max_query_length):
    if len(args) == 3:
        (query, docs, labels) = args
        query_id = None
        doc_ids = None
    else:
        (query, docs, labels, query_id, doc_ids) = args

    cls_id = _tokenizer.vocab['[CLS]']
    sep_id = _tokenizer.vocab['[SEP]']

    query = clean_text(query, strip_accents=True)
    q_words = _tokenizer.tokenize(query)
    if len(q_words) > max_query_length:
        q_words = q_words[:max_query_length]
    q_word_ids = [cls_id] + [w.id for w in q_words] + [sep_id]

    tf_q_word_ids = tf.train.Feature(int64_list=Int64List(value=q_word_ids))

    q_mentions = detect_mentions(query, q_words, _entity_linker, _entity_vocab)
    tf_q_entity_ids = tf.train.Feature(int64_list=Int64List(value=[m[0] for m in q_mentions]))
    tf_q_entity_positions = tf.train.Feature(int64_list=Int64List(
        value=list(itertools.chain(*[(m[1] + 1, m[2] + 1) for m in q_mentions]))))  # 1 for [CLS]

    max_doc_length = max_seq_length - len(q_words) - 3  # 3 for [CLS], [SEP] and [SEP]

    ret = []

    for (i, (doc_text, label)) in enumerate(zip(docs, labels)):
        doc_text = clean_text(doc_text, strip_accents=True)
        d_words = _tokenizer.tokenize(doc_text)
        if len(d_words) > max_doc_length:
            d_words = d_words[:max_doc_length]

        d_word_ids = [w.id for w in d_words] + [sep_id]
        tf_d_word_ids = tf.train.Feature(int64_list=tf.train.Int64List(value=d_word_ids))

        offset = len(q_word_ids)
        d_mentions = detect_mentions(doc_text, d_words, _entity_linker, _entity_vocab)
        tf_d_entity_ids = tf.train.Feature(int64_list=tf.train.Int64List(
            value=[m[0] for m in d_mentions]))
        tf_d_entity_positions = tf.train.Feature(int64_list=Int64List(
            value=list(itertools.chain(*[(m[1] + offset, m[2] + offset) for m in d_mentions]))))
        tf_labels = tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))

        features = tf.train.Features(feature={
            'query_word_ids': tf_q_word_ids,
            'query_entity_ids': tf_q_entity_ids,
            'query_entity_positions': tf_q_entity_positions,
            'doc_word_ids': tf_d_word_ids,
            'doc_entity_ids': tf_d_entity_ids,
            'doc_entity_positions': tf_d_entity_positions,
            'label': tf_labels,
        })
        example = tf.train.Example(features=features)

        if query_id and doc_ids:
            ret.append((example.SerializeToString(), '\t'.join([query_id, doc_ids[i]])))
        else:
            ret.append((example.SerializeToString(), None))

    return ret


def detect_mentions(text, tokens, entity_linker, entity_vocab):
    token_start_map = np.full(len(text), -1)
    token_end_map = np.full(len(text), -1)

    ret = []
    for (ind, token) in enumerate(tokens):
        token_start_map[token.start] = ind
        token_end_map[token.end - 1] = ind

    for (mention_span, mentions) in entity_linker.detect_mentions(text):
        token_start = token_start_map[mention_span[0]]
        if token_start == -1:
            continue

        token_end = token_end_map[mention_span[1] - 1]
        if token_end == -1:
            continue
        token_end += 1

        for mention in mentions:
            try:
                entity_id = entity_vocab[mention.title]
            except KeyError:
                continue

            ret.append((entity_id, token_start, token_end))

    return ret


if __name__ == '__main__':
    main()
