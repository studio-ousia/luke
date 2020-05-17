from typing import List
import functools
import itertools
import json
import multiprocessing
import os
import random
import re
from contextlib import closing
from multiprocessing.pool import Pool

import click
import tensorflow as tf
from tensorflow.io import TFRecordWriter
from tensorflow.train import Int64List
from transformers import BertTokenizer, RobertaTokenizer, XLMRobertaTokenizer
from tqdm import tqdm
from wikipedia2vec.dump_db import DumpDB

from luke.utils.entity_vocab import UNK_TOKEN, EntityVocab, MultilingualEntityVocab
from luke.utils.sentence_tokenizer import NLTKSentenceTokenizer, OpenNLPSentenceTokenizer, JapaneseSentenceTokenizer

DATASET_FILE = 'dataset.tf'
METADATA_FILE = 'metadata.json'
ENTITY_VOCAB_FILE = 'entity_vocab.tsv'
MULTILINGULA_ENTITY_VOCAB_FILE = 'multilingual_entity_vocab.json'

# global variables used in pool workers
_dump_db = _tokenizer = _sentence_tokenizer = _entity_vocab = _max_num_tokens = _max_entity_length = \
    _max_mention_length = _min_sentence_length = _include_sentences_without_entities = _include_unk_entities = None


@click.command()
@click.argument('dump_db_file', type=click.Path(exists=True))
@click.argument('tokenizer_name')
@click.argument('entity_vocab_file', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path(file_okay=False))
@click.option('--multilingual', is_flag=True)
@click.option('--sentence-tokenizer', default='opennlp', type=click.Choice(['opennlp', 'nltk', 'ja']))
@click.option('--max-seq-length', default=512)
@click.option('--max-entity-length', default=128)
@click.option('--max-mention-length', default=30)
@click.option('--min-sentence-length', default=5)
@click.option('--include-sentences-without-entities', is_flag=True)
@click.option('--include-unk-entities/--skip-unk-entities', default=False)
@click.option('--pool-size', default=multiprocessing.cpu_count())
@click.option('--chunk-size', default=100)
def build_wikipedia_pretraining_dataset(dump_db_file, tokenizer_name, entity_vocab_file, output_dir, multilingual,
                                        sentence_tokenizer, **kwargs):
    dump_db = DumpDB(dump_db_file)
    if 'xlm-roberta' in tokenizer_name:
        tokenizer = XLMRobertaTokenizer.from_pretrained(tokenizer_name)
    elif 'roberta' in tokenizer_name:
        tokenizer = RobertaTokenizer.from_pretrained(tokenizer_name)
    else:
        tokenizer = BertTokenizer.from_pretrained(tokenizer_name)

    if sentence_tokenizer == 'opennlp':
        sentence_tokenizer = OpenNLPSentenceTokenizer()
    elif sentence_tokenizer == 'ja':
        sentence_tokenizer = JapaneseSentenceTokenizer()
    elif sentence_tokenizer == 'nltk':
        sentence_tokenizer = NLTKSentenceTokenizer()
    else:
        raise Exception(f"sentence_tokenizer: {sentence_tokenizer} is not defined.")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if multilingual:
        entity_vocab = MultilingualEntityVocab(entity_vocab_file)
        MultilingualPretrainingDataset.build(dump_db, tokenizer, sentence_tokenizer, entity_vocab, output_dir, **kwargs)
    else:
        entity_vocab = EntityVocab(entity_vocab_file)
        WikipediaPretrainingDataset.build(dump_db, tokenizer, sentence_tokenizer, entity_vocab, output_dir, **kwargs)


class WikipediaPretrainingDataset(object):

    def __init__(self, dataset_dir: str):
        self._dataset_dir = dataset_dir

        with open(os.path.join(dataset_dir, METADATA_FILE)) as metadata_file:
            self.metadata = json.load(metadata_file)

    def __len__(self):
        return self.metadata['number_of_items']

    @property
    def max_seq_length(self):
        return self.metadata['max_seq_length']

    @property
    def max_entity_length(self):
        return self.metadata['max_entity_length']

    @property
    def max_mention_length(self):
        return self.metadata['max_mention_length']

    @property
    def tokenizer(self):
        tokenizer_class = self.metadata.get('tokenizer_class', '').lower()

        if 'xlmroberta' in tokenizer_class:
            return XLMRobertaTokenizer.from_pretrained(self._dataset_dir)
        elif 'roberta' in tokenizer_class:
            return RobertaTokenizer.from_pretrained(self._dataset_dir)
        else:
            return BertTokenizer.from_pretrained(self._dataset_dir)

    @property
    def entity_vocab(self):
        entity_vocab_file_path = os.path.join(self._dataset_dir, ENTITY_VOCAB_FILE)
        if os.path.exists(entity_vocab_file_path):
            return EntityVocab(entity_vocab_file_path)
        else:
            multilingual_entity_vocab_file_path = os.path.join(self._dataset_dir, MULTILINGULA_ENTITY_VOCAB_FILE)
            return MultilingualEntityVocab(multilingual_entity_vocab_file_path)

    def create_iterator(self, skip=0, num_workers=1, worker_index=0, shuffle_buffer_size=1000, shuffle_seed=0,
                        num_parallel_reads=10):
        features = dict(
            word_ids=tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
            entity_ids=tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
            entity_position_ids=tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
            page_id=tf.io.FixedLenFeature([1], tf.int64),
        )
        dataset = tf.data.TFRecordDataset([os.path.join(self._dataset_dir, DATASET_FILE)], compression_type='GZIP',
                                          num_parallel_reads=num_parallel_reads)
        dataset = dataset.repeat()
        dataset = dataset.shuffle(shuffle_buffer_size, seed=shuffle_seed)
        dataset = dataset.skip(skip)
        dataset = dataset.shard(num_workers, worker_index)
        dataset = dataset.map(functools.partial(tf.io.parse_single_example, features=features))
        it = tf.compat.v1.data.make_one_shot_iterator(dataset)
        it = it.get_next()

        with tf.compat.v1.Session() as sess:
            try:
                while True:
                    obj = sess.run(it)
                    yield dict(
                        page_id=obj['page_id'][0],
                        word_ids=obj['word_ids'],
                        entity_ids=obj['entity_ids'],
                        entity_position_ids=obj['entity_position_ids'].reshape(-1, self.metadata['max_mention_length'])
                    )
            except tf.errors.OutOfRangeError:
                pass

    @classmethod
    def build(cls, dump_db, tokenizer, sentence_tokenizer, entity_vocab, output_dir, max_seq_length, max_entity_length,
              max_mention_length, min_sentence_length, include_sentences_without_entities, include_unk_entities,
              pool_size, chunk_size):
        target_titles = [title for title in dump_db.titles()
                         if not (':' in title and title.lower().split(':')[0] in ('image', 'file', 'category'))]
        random.shuffle(target_titles)

        max_num_tokens = max_seq_length - 2  # 2 for [CLS] and [SEP]

        tokenizer.save_pretrained(output_dir)
        entity_vocab.save(os.path.join(output_dir, ENTITY_VOCAB_FILE))

        number_of_items = 0
        tf_file = os.path.join(output_dir, DATASET_FILE)
        options = tf.io.TFRecordOptions(tf.io.TFRecordCompressionType.GZIP)
        with TFRecordWriter(tf_file, options=options) as writer:
            with tqdm(total=len(target_titles)) as pbar:
                initargs = (dump_db, tokenizer, sentence_tokenizer, entity_vocab, max_num_tokens, max_entity_length,
                            max_mention_length, min_sentence_length, include_sentences_without_entities,
                            include_unk_entities)
                with closing(Pool(pool_size, initializer=WikipediaPretrainingDataset._initialize_worker,
                                  initargs=initargs)) as pool:
                    for ret in pool.imap(WikipediaPretrainingDataset._process_page, target_titles,
                                         chunksize=chunk_size):
                        for data in ret:
                            writer.write(data)
                            number_of_items += 1
                        pbar.update()

        with open(os.path.join(output_dir, METADATA_FILE), 'w') as metadata_file:
            json.dump(dict(
                number_of_items=number_of_items,
                max_seq_length=max_seq_length,
                max_entity_length=max_entity_length,
                max_mention_length=max_mention_length,
                min_sentence_length=min_sentence_length,
                tokenizer_class=tokenizer.__class__.__name__,
            ), metadata_file, indent=2)

    @staticmethod
    def _initialize_worker(dump_db, tokenizer, sentence_tokenizer, entity_vocab, max_num_tokens, max_entity_length,
                           max_mention_length, min_sentence_length, include_sentences_without_entities,
                           include_unk_entities):
        global _dump_db, _tokenizer, _sentence_tokenizer, _entity_vocab, _max_num_tokens, _max_entity_length, \
            _max_mention_length, _min_sentence_length, _include_sentences_without_entities, _include_unk_entities

        _dump_db = dump_db
        _tokenizer = tokenizer
        _sentence_tokenizer = sentence_tokenizer
        _entity_vocab = entity_vocab
        _max_num_tokens = max_num_tokens
        _max_entity_length = max_entity_length
        _max_mention_length = max_mention_length
        _min_sentence_length = min_sentence_length
        _include_sentences_without_entities = include_sentences_without_entities
        _include_unk_entities = include_unk_entities

    @staticmethod
    def _process_page(page_title):
        if page_title in _entity_vocab:
            page_id = _entity_vocab[page_title]
        else:
            page_id = -1

        sentences = []

        def tokenize(text, add_prefix_space):
            text = re.sub(r'\s+', ' ', text).rstrip()
            if isinstance(_tokenizer, RobertaTokenizer):
                return _tokenizer.tokenize(text, add_prefix_space=add_prefix_space)
            else:
                return _tokenizer.tokenize(text)

        for paragraph in _dump_db.get_paragraphs(page_title):
            paragraph_text = paragraph.text
            paragraph_links = []
            for link in paragraph.wiki_links:
                link_title = _dump_db.resolve_redirect(link.title)
                # remove category links
                if link_title.startswith('Category:') and link.text.lower().startswith('category:'):
                    paragraph_text = paragraph_text[:link.start] + ' ' * \
                                     (link.end - link.start) + paragraph_text[link.end:]
                else:
                    if link_title in _entity_vocab:
                        paragraph_links.append((link_title, link.start, link.end))
                    elif _include_unk_entities:
                        paragraph_links.append((UNK_TOKEN, link.start, link.end))

            for sent_start, sent_end in _sentence_tokenizer.span_tokenize(paragraph_text):
                cur = sent_start
                sent_words = []
                sent_links = []
                for link_title, link_start, link_end in paragraph_links:
                    if not (sent_start <= link_start < sent_end and link_end <= sent_end):
                        continue

                    entity_id = _entity_vocab[link_title]

                    text = paragraph_text[cur:link_start]
                    if cur == 0 or text.startswith(' ') or paragraph_text[cur - 1] == ' ':
                        sent_words += tokenize(text, True)
                    else:
                        sent_words += tokenize(text, False)

                    link_text = paragraph_text[link_start:link_end]
                    if link_start == 0 or link_text.startswith(' ') or paragraph_text[link_start - 1] == ' ':
                        link_words = tokenize(link_text, True)
                    else:
                        link_words = tokenize(link_text, False)

                    sent_links.append((entity_id, len(sent_words), len(sent_words) + len(link_words)))
                    sent_words += link_words
                    cur = link_end

                text = paragraph_text[cur:sent_end]
                if cur == 0 or text.startswith(' ') or paragraph_text[cur - 1] == ' ':
                    sent_words += tokenize(text, True)
                else:
                    sent_words += tokenize(text, False)

                if len(sent_words) < _min_sentence_length or len(sent_words) > _max_num_tokens:
                    continue
                sentences.append((sent_words, sent_links))

        ret = []
        words = []
        links = []
        for i, (sent_words, sent_links) in enumerate(sentences):
            links += [(id_, start + len(words), end + len(words)) for id_, start, end in sent_links]
            words += sent_words
            if i == len(sentences) - 1 or len(words) + len(sentences[i + 1][0]) > _max_num_tokens:
                if links or _include_sentences_without_entities:
                    links = links[:_max_entity_length]
                    word_ids = _tokenizer.convert_tokens_to_ids(words)
                    assert _min_sentence_length <= len(word_ids) <= _max_num_tokens
                    entity_ids = [id_ for id_, _, _, in links]
                    assert len(entity_ids) <= _max_entity_length
                    entity_position_ids = itertools.chain(
                        *[(list(range(start, end)) + [-1] * (_max_mention_length - end + start))[:_max_mention_length]
                          for _, start, end in links])

                    example = tf.train.Example(features=tf.train.Features(feature=dict(
                        page_id=tf.train.Feature(int64_list=tf.train.Int64List(value=[page_id])),
                        word_ids=tf.train.Feature(int64_list=tf.train.Int64List(value=word_ids)),
                        entity_ids=tf.train.Feature(int64_list=tf.train.Int64List(value=entity_ids)),
                        entity_position_ids=tf.train.Feature(int64_list=Int64List(value=entity_position_ids)),
                    )))
                    ret.append((example.SerializeToString()))

                words = []
                links = []

        return ret


class MultilingualPretrainingDataset(WikipediaPretrainingDataset):

    def __init__(self,
                 dataset_dir_list: List[str]):
        self.dataset_dir_list = dataset_dir_list
        self.dataset_list = [WikipediaPretrainingDataset(d) for d in dataset_dir_list]

        self.data_size_list = [len(dataset) for dataset in self.dataset_list]
        self.total_data_size = sum(self.data_size_list)

    def __len__(self):
        return self.total_data_size

    @property
    def max_seq_length(self):
        return max([dataset.max_seq_length for dataset in self.dataset_list])

    @property
    def max_entity_length(self):
        return max([dataset.max_entity_length for dataset in self.dataset_list])

    @property
    def max_mention_length(self):
        return max([dataset.max_mention_length for dataset in self.dataset_list])

    @property
    def tokenizer(self):
        return self.dataset_list[0].tokenizer

    @property
    def entity_vocab(self):
        return self.dataset_list[0].entity_vocab
