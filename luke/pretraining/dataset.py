import functools
import itertools
import json
import os
import random
import re
from contextlib import closing
import multiprocessing
from multiprocessing.pool import Pool
import click
from pytorch_transformers import BertTokenizer, RobertaTokenizer
import tensorflow as tf
from tensorflow.io import TFRecordWriter
from tensorflow.train import Int64List
from tqdm import tqdm
from wikipedia2vec.dump_db import DumpDB

from luke.utils.entity_linker import EntityLinker
from luke.utils.entity_vocab import EntityVocab, UNK_TOKEN
from luke.utils.sentence_tokenizer import NLTKSentenceTokenizer, OpenNLPSentenceTokenizer

DATASET_FILE = 'dataset.tf'
METADATA_FILE = 'metadata.json'
ENTITY_LINKER_FILE = 'entity_linker.pkl'
ENTITY_VOCAB_FILE = 'entity_vocab.tsv'

# global variables used in pool workers
_dump_db = _tokenizer = _sentence_tokenizer = _entity_vocab = _entity_linker = _max_num_tokens = _max_entity_length =\
    _min_sentence_length = _max_candidate_length = None


class WikipediaPretrainingDataset(object):
    def __init__(self, dataset_dir):
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
    def max_candidate_length(self):
        return self.metadata['max_candidate_length']

    @property
    def tokenizer(self):
        if 'roberta' in self.metadata.get('tokenizer_class', '').lower():
            return RobertaTokenizer.from_pretrained(self._dataset_dir)
        else:
            return BertTokenizer.from_pretrained(self._dataset_dir)

    @property
    def entity_vocab(self):
        return EntityVocab(os.path.join(self._dataset_dir, ENTITY_VOCAB_FILE))

    @property
    def entity_linker(self):
        return EntityLinker(os.path.join(self._dataset_dir, ENTITY_LINKER_FILE))

    def create_iterator(self, skip=0, num_workers=1, worker_index=0, shuffle_buffer_size=1000, shuffle_seed=0,
                        num_parallel_reads=10):
        features = dict(
            word_ids=tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
            entity_ids=tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
            entity_position_ids=tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
            entity_candidate_ids=tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
            entity_candidate_labels=tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
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
                        page_id=obj['page_id'],
                        word_ids=obj['word_ids'],
                        entity_ids=obj['entity_ids'],
                        entity_position_ids=obj['entity_position_ids'].reshape(-1, self.metadata['max_mention_length']),
                        entity_candidate_ids=obj['entity_candidate_ids'].reshape(-1, self.metadata['max_candidate_length']),
                        entity_candidate_labels=obj['entity_candidate_labels']
                    )
            except tf.errors.OutOfRangeError:
                pass

    @staticmethod
    def build(dump_db, tokenizer, sentence_tokenizer, entity_vocab, entity_linker, output_dir, max_seq_length,
              max_entity_length, max_mention_length, min_sentence_length, max_candidate_length,
              skip_sentences_without_entities, pool_size, chunk_size):
        target_titles = [title for title in dump_db.titles()
                         if not (':' in title and title.lower().split(':')[0] in ('image', 'file', 'category'))]
        random.shuffle(target_titles)

        max_num_tokens = max_seq_length - 2  # 2 for [CLS] and [SEP]

        tokenizer.save_pretrained(output_dir)
        entity_vocab.save(os.path.join(output_dir, ENTITY_VOCAB_FILE))
        entity_linker.save(os.path.join(output_dir, ENTITY_LINKER_FILE))

        number_of_items = 0
        tf_file = os.path.join(output_dir, DATASET_FILE)
        options = tf.io.TFRecordOptions(tf.io.TFRecordCompressionType.GZIP)
        with TFRecordWriter(tf_file, options=options) as writer:
            with tqdm(total=len(target_titles)) as pbar:
                initargs = (dump_db, tokenizer, sentence_tokenizer, entity_vocab, entity_linker, max_num_tokens,
                            max_entity_length, max_mention_length, min_sentence_length, max_candidate_length,
                            skip_sentences_without_entities)
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
                max_candidate_length=max_candidate_length,
                tokenizer_class=tokenizer.__class__.__name__,
            ), metadata_file, indent=2)

    @staticmethod
    def _initialize_worker(dump_db, tokenizer, sentence_tokenizer, entity_vocab, entity_linker, max_num_tokens,
                           max_entity_length, max_mention_length, min_sentence_length, max_candidate_length,
                           skip_sentences_without_entities):
        global _dump_db, _tokenizer, _sentence_tokenizer, _entity_vocab, _entity_linker, _max_num_tokens,\
            _max_entity_length, _max_mention_length, _min_sentence_length, _max_candidate_length,\
            _skip_sentences_without_entities

        _dump_db = dump_db
        _tokenizer = tokenizer
        _sentence_tokenizer = sentence_tokenizer
        _entity_vocab = entity_vocab
        _entity_linker = entity_linker
        _max_num_tokens = max_num_tokens
        _max_entity_length = max_entity_length
        _max_mention_length = max_mention_length
        _min_sentence_length = min_sentence_length
        _max_candidate_length = max_candidate_length
        _skip_sentences_without_entities = skip_sentences_without_entities

    @staticmethod
    def _process_page(page_title):
        if page_title in _entity_vocab:
            page_id = _entity_vocab[page_title]
        else:
            page_id = -1

        sentences = []

        def tokenize(text):
            text = re.sub(r'\s+', ' ', text).rstrip()
             # The private _tokenize() method needs to be used here because the public tokenize() method automatically
             # removes the leading whitespace of the first word of the given text
            return _tokenizer._tokenize(text)

        for paragraph in _dump_db.get_paragraphs(page_title):
            paragraph_text = paragraph.text
            paragraph_links = []
            for link in paragraph.wiki_links:
                link_title = _dump_db.resolve_redirect(link.title)
                # remove category links
                if link_title.startswith('Category:') and link.text.lower().startswith('category:'):
                    paragraph_text = paragraph_text[:link.start] + ' ' * \
                        (link.end - link.start) + paragraph_text[link.end:]
                elif link_title in _entity_vocab:
                    paragraph_links.append((link_title, link.start, link.end))

            for sent_start, sent_end in _sentence_tokenizer.span_tokenize(paragraph_text):
                cur = sent_start
                sent_words = []
                sent_links = []
                for link_title, link_start, link_end in paragraph_links:
                    if not (sent_start <= link_start < sent_end and link_end <= sent_end):
                        continue

                    entity_id = _entity_vocab[link_title]

                    text = paragraph_text[cur:link_start]
                    if cur != 0 and paragraph_text[cur - 1] == ' ':
                        text = ' ' + text
                    sent_words += tokenize(text)

                    link_text = paragraph_text[link_start:link_end]
                    if link_start != 0 and paragraph_text[link_start - 1] == ' ':
                        link_words = tokenize(' ' + link_text)
                    else:
                        link_words = tokenize(link_text)

                    candidates = sorted(_entity_linker.query(link_text), reverse=True, key=lambda c: c.prior_prob)
                    candidates = candidates[:_max_candidate_length - 1]
                    candidate_indices = [_entity_vocab[UNK_TOKEN]]
                    candidate_indices += [_entity_vocab[c.title] for c in candidates if c.title in _entity_vocab]
                    candidate_indices += [0] * (_max_candidate_length - len(candidate_indices))
                    try:
                        candidate_label = candidate_indices.index(entity_id)
                    except ValueError:
                        candidate_label = 0

                    sent_links.append((entity_id, len(sent_words), len(sent_words) + len(link_words),
                                       candidate_indices, candidate_label))
                    sent_words += link_words
                    cur = link_end

                text = paragraph_text[cur:sent_end]
                if cur != 0 and paragraph_text[cur - 1] == ' ':
                    text = ' ' + text
                sent_words += tokenize(text)

                if len(sent_words) < _min_sentence_length or len(sent_words) > _max_num_tokens:
                    continue
                sentences.append((sent_words, sent_links))

        ret = []
        words = []
        links = []
        for i, (sent_words, sent_links) in enumerate(sentences):
            links += [(id_, start + len(words), end + len(words), cand, lb) for id_, start, end, cand, lb in sent_links]
            words += sent_words
            if i == len(sentences) - 1 or len(words) + len(sentences[i + 1][0]) > _max_num_tokens:
                if links or not _skip_sentences_without_entities:
                    links = links[:_max_entity_length]
                    word_ids = _tokenizer.convert_tokens_to_ids(words)
                    assert _min_sentence_length <= len(word_ids) <= _max_num_tokens
                    entity_ids = [id_ for id_, _, _, _, _ in links]
                    assert len(entity_ids) <= _max_entity_length
                    entity_position_ids = itertools.chain(
                        *[(list(range(s, e)) + [-1] * (_max_mention_length - e + s))[:_max_mention_length]
                          for _, s, e, _, _ in links])
                    entity_candidate_ids = itertools.chain(*[cands for _, _, _, cands, _ in links])
                    entity_candidate_labels = [label for _, _, _, _, label in links]

                    example = tf.train.Example(features=tf.train.Features(feature=dict(
                        page_id=tf.train.Feature(int64_list=tf.train.Int64List(value=[page_id])),
                        word_ids=tf.train.Feature(int64_list=tf.train.Int64List(value=word_ids)),
                        entity_ids=tf.train.Feature(int64_list=tf.train.Int64List(value=entity_ids)),
                        entity_position_ids=tf.train.Feature(int64_list=Int64List(value=entity_position_ids)),
                        entity_candidate_ids=tf.train.Feature(int64_list=Int64List(value=entity_candidate_ids)),
                        entity_candidate_labels=tf.train.Feature(int64_list=Int64List(value=entity_candidate_labels)),
                    )))
                    ret.append((example.SerializeToString()))

                words = []
                links = []

        return ret


@click.command()
@click.argument('dump_db_file', type=click.Path(exists=True))
@click.argument('tokenizer_name')
@click.argument('entity_vocab_file', type=click.Path(exists=True))
@click.argument('entity_linker_file', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path(file_okay=False))
@click.option('--sentence-tokenizer', default='opennlp', type=click.Choice(['opennlp', 'nltk']))
@click.option('--max-seq-length', default=512)
@click.option('--max-entity-length', default=128)
@click.option('--max-mention-length', default=30)
@click.option('--min-sentence-length', default=5)
@click.option('--max-candidate-length', default=30)
@click.option('--skip-sentences-without-entities', is_flag=True)
@click.option('--pool-size', default=multiprocessing.cpu_count())
@click.option('--chunk-size', default=100)
def build_wikipedia_pretraining_dataset(dump_db_file, tokenizer_name, entity_vocab_file, entity_linker_file, output_dir,
                                        sentence_tokenizer, **kwargs):
    dump_db = DumpDB(dump_db_file)
    if 'roberta' in tokenizer_name:
        tokenizer = RobertaTokenizer.from_pretrained(tokenizer_name)
    else:
        tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
    if sentence_tokenizer == 'opennlp':
        sentence_tokenizer = OpenNLPSentenceTokenizer()
    else:
        sentence_tokenizer = NLTKSentenceTokenizer()
    entity_vocab = EntityVocab(entity_vocab_file)
    entity_linker = EntityLinker(entity_linker_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    WikipediaPretrainingDataset.build(dump_db, tokenizer, sentence_tokenizer, entity_vocab, entity_linker, output_dir,
                                      **kwargs)
