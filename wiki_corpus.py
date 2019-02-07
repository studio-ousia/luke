# -*- coding: utf-8 -*-

import random
from multiprocessing.pool import Pool
import joblib
import numpy as np
from marisa_trie import Trie
from tqdm import tqdm

from utils.entity_linker import EntityLinker


class Entity(object):
    __slots__ = ('title',)

    def __init__(self, title):
        self.title = title

    def __repr__(self):
       return '<Entity %s>' % self.title


class Mention(Entity):
    __slots__ = ('start', 'end', 'is_gold')

    def __init__(self, title, start, end, is_gold):
        super(Mention, self).__init__(title)

        self.start = start
        self.end = end
        self.is_gold = is_gold

    @property
    def span(self):
        return (self.start, self.end)

    def __repr__(self):
       return '<Mention title=%s start=%d end=%d is_gold=%d>' % (self.title, self.start, self.end,
                                                                 self.is_gold)


class Sentence(object):
    __slots__ = ('word_ids', 'mentions', 'page_entity', '_word_vocab')

    def __init__(self, word_ids, mentions, page_entity, word_vocab):
        self.word_ids = word_ids
        self.mentions = mentions
        self.page_entity = page_entity
        self._word_vocab = word_vocab

    @property
    def words(self):
        return [self._word_vocab.get_word_by_id(id_) for id_ in self.word_ids]

    def __repr__(self):
       return '<Sentence %s...>' % (' '.join(self.words)[:100])


class WikiCorpus(object):
    def __init__(self, corpus_file, mmap_mode='r'):
        if mmap_mode is None:
            self._word_arr = np.fromfile(corpus_file + '_word.bin', dtype=np.uint32)
            self._mention_arr = np.fromfile(corpus_file + '_mention.bin', dtype=np.uint32)
        else:
            self._word_arr = np.memmap(corpus_file + '_word.bin', dtype=np.uint32, mode=mmap_mode)
            self._mention_arr = np.memmap(corpus_file + '_mention.bin', dtype=np.uint32,
                                          mode=mmap_mode)

        meta = joblib.load(corpus_file + '_meta.pkl', mmap_mode=mmap_mode)

        self.tokenizer = meta['tokenizer']
        self.word_vocab = self.tokenizer.vocab
        self.title_vocab = meta['title_vocab']

        self._word_offsets = meta['word_offsets']
        self._mention_offsets = meta['mention_offsets']
        self._page_ids = meta['page_ids']
        self._page_offsets = meta['page_offsets']

    @property
    def page_size(self):
        return self._page_offsets.size - 1

    @property
    def word_size(self):
        return self._word_arr.shape[0]

    def iterate_pages(self, page_indices=None, shuffle=True):
        if page_indices is None:
            if shuffle:
                page_indices = np.random.permutation(self._page_offsets.size - 1)
            else:
                page_indices = np.arange(self._page_offsets.size - 1)

        for index in page_indices:
            yield self.read_page(index)

    def read_random_page(self):
        index = random.randint(0, self._page_offsets.size - 2)
        return self.read_page(index)

    def read_page(self, index):
        return [self.read_sentence(i)
                for i in range(self._page_offsets[index], self._page_offsets[index+1])]

    def read_sentence(self, index):
        word_ids = self._word_arr[self._word_offsets[index]:self._word_offsets[index+1]]

        mention_data = self._mention_arr[self._mention_offsets[index]:self._mention_offsets[index+1]]
        mentions = [Mention(self.title_vocab.restore_key(mention_data[n]), *mention_data[n+1:n+4])
                    for n in range(0, mention_data.size, 4)]

        entity_id = self._page_ids[index]
        entity = Entity(self.title_vocab.restore_key(entity_id))

        return Sentence(word_ids, mentions, entity, self.word_vocab)

    def convert_ids_to_words(self, word_ids):
        return [self.word_vocab.get_word_by_id(id_) for id_ in word_ids]

    @staticmethod
    def build_corpus_data(dump_db, mention_db, tokenizer, sentence_tokenizer, out_file,
                          min_sentence_len, min_link_prob, min_prior_prob, pool_size):
        word_offsets = [0]
        mention_offsets = [0]
        page_ids = []
        page_offsets = [0]

        word_offset = 0
        mention_offset = 0

        title_vocab = Trie(dump_db.titles())
        entity_linker = EntityLinker(mention_db, min_link_prob, 0.0)

        try:
            word_file = open(out_file + '_word.bin', mode='wb')
            mention_file = open(out_file + '_mention.bin', mode='wb')

            with tqdm(total=dump_db.page_size()) as bar:
                with Pool(pool_size, initializer=_init_worker,
                          initargs=(dump_db, title_vocab, tokenizer, sentence_tokenizer,
                                    entity_linker, min_sentence_len, min_prior_prob)) as pool:
                    for (page_id, data) in pool.imap(_process_page, dump_db.titles()):
                        if data:
                            for (word_arr, mention_arr) in data:
                                word_file.write(word_arr.tobytes())
                                word_offset += word_arr.size
                                word_offsets.append(word_offset)

                                mention_file.write(mention_arr.tobytes())
                                mention_offset += mention_arr.size
                                mention_offsets.append(mention_offset)

                                page_ids.append(page_id)

                            page_offsets.append(len(word_offsets) - 1)

                        bar.update()

        finally:
            word_file.close()
            mention_file.close()

        word_offsets = np.array(word_offsets, dtype=np.uint64)
        mention_offsets = np.array(mention_offsets, dtype=np.uint64)
        page_ids = np.array(page_ids, dtype=np.uint32)
        page_offsets = np.array(page_offsets, dtype=np.uint32)

        joblib.dump(dict(
            title_vocab=title_vocab,
            tokenizer=tokenizer,
            word_offsets=word_offsets,
            mention_offsets=mention_offsets,
            page_ids=page_ids,
            page_offsets=page_offsets,
        ), out_file + '_meta.pkl')


_dump_db = None
_title_vocab = None
_tokenizer = None
_sentence_tokenizer = None
_entity_linker = None
_min_sentence_len = None
_min_prior_prob = None


def _init_worker(dump_db, title_vocab, tokenizer, sentence_tokenizer, entity_linker,
                 min_sentence_len, min_prior_prob):
    global _dump_db, _title_vocab, _tokenizer, _sentence_tokenizer, _entity_linker,\
        _min_sentence_len, _min_prior_prob

    _dump_db = dump_db
    _title_vocab = title_vocab
    _tokenizer = tokenizer
    _sentence_tokenizer = sentence_tokenizer
    _entity_linker = entity_linker
    _min_sentence_len = min_sentence_len
    _min_prior_prob = min_prior_prob


def _process_page(page_title):
    data = []

    for paragraph in _dump_db.get_paragraphs(page_title):
        paragraph_text = paragraph.text
        links = []
        for link in paragraph.wiki_links:
            link_title = _dump_db.resolve_redirect(link.title)
            if link_title in _title_vocab:
                links.append((link_title, link.span))

        for (sent_start, sent_end) in _sentence_tokenizer.span_tokenize(paragraph_text):
            sent_text = paragraph_text[sent_start:sent_end]

            if len(sent_text.split()) < _min_sentence_len:
                continue

            tokens = _tokenizer.tokenize(sent_text)

            token_start_map = np.full(len(sent_text), -1)
            token_end_map = np.full(len(sent_text), -1)
            for (ind, token) in enumerate(tokens):
                token_start_map[token.start] = ind
                token_end_map[token.end - 1] = ind

            gold_mentions = np.full(len(sent_text), -1)
            for (link_title, (link_start, link_end)) in links:
                if not (sent_start <= link_start < sent_end and link_end <= sent_end):
                    continue

                token_start = token_start_map[link_start - sent_start]
                if token_start == -1:
                    continue

                token_end = token_end_map[link_end - sent_start - 1]
                if token_end == -1:
                    continue
                token_end += 1

                if link_title not in _title_vocab:
                    continue

                id_ = _title_vocab[link_title]
                gold_mentions[token_start:token_end] = id_

            mentions = []
            for mention in _entity_linker.detect_mentions(sent_text):
                token_start = token_start_map[mention.span[0]]
                if token_start == -1:
                    continue

                token_end = token_end_map[mention.span[1] - 1]
                if token_end == -1:
                    continue
                token_end += 1

                if mention.title not in _title_vocab:
                    continue

                id_ = _title_vocab[mention.title]
                is_gold = int(id_ in gold_mentions[token_start:token_end])
                if is_gold or mention.prior_prob >= _min_prior_prob:
                    mentions.extend([id_, token_start, token_end, is_gold])

            word_arr = np.array([t.id for t in tokens], dtype=np.uint32)
            mention_arr = np.array(mentions, dtype=np.uint32)

            data.append((word_arr, mention_arr))

    return (_title_vocab[page_title], data)
