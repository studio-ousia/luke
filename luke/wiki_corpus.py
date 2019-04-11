# -*- coding: utf-8 -*-

import random
from multiprocessing.pool import Pool
import joblib
import numpy as np
from marisa_trie import Trie
from tqdm import tqdm


class Word(object):
    __slots__ = ('id', '_vocab')

    def __init__(self, id_, vocab):
        self.id = id_
        self._vocab = vocab

    @property
    def text(self):
        return self._vocab.get_word_by_id(self.id)

    def __repr__(self):
        return '<Word %s>' % self.text


class Entity(object):
    __slots__ = ('title',)

    def __init__(self, title):
        self.title = title

    def __repr__(self):
       return '<Entity %s>' % self.title


class Link(Entity):
    __slots__ = ('start', 'end')

    def __init__(self, title, start, end):
        super(Link, self).__init__(title)

        self.start = start
        self.end = end

    @property
    def span(self):
        return (self.start, self.end)

    def __repr__(self):
       return '<Link title=%s start=%d end=%d>' % (self.title, self.start, self.end)


class MentionCandidate(Entity):
    __slots__ = ('start', 'end', 'link_prob', 'prior_prob', 'rank', 'label')

    def __init__(self, title, start, end, link_prob, prior_prob, rank, label):
        super(MentionCandidate, self).__init__(title)

        self.start = start
        self.end = end
        self.link_prob = link_prob
        self.prior_prob = prior_prob
        self.rank = rank
        self.label = label

    @property
    def span(self):
        return (self.start, self.end)

    def __repr__(self):
       return '<MentionCandidate title=%s start=%d end=%d label=%d>' % (
           self.title, self.start, self.end, self.label)


class Page(object):
    __slots__ = ('entity', 'sentences')

    def __init__(self, entity, sentences):
        self.entity = entity
        self.sentences = sentences

    def __repr__(self):
        return '<Page %s>' % self.entity.title


class Sentence(object):
    __slots__ = ('words', 'links', 'mentions')

    def __init__(self, words, links, mentions):
        self.words = words
        self.links = links
        self.mentions = mentions

    def __repr__(self):
       return '<Sentence %s...>' % (' '.join(self.words)[:100])


class WikiCorpus(object):
    def __init__(self, corpus_file, mmap_mode='r'):
        self._link_arr = None

        if mmap_mode is None:
            self._word_arr = np.fromfile(corpus_file + '_word.bin', dtype=np.uint32)
            self._link_arr = np.fromfile(corpus_file + '_link.bin', dtype=np.uint32)
            self._mention_arr = np.fromfile(corpus_file + '_mention.bin', dtype=np.int32)
        else:
            self._word_arr = np.memmap(corpus_file + '_word.bin', dtype=np.uint32, mode=mmap_mode)
            self._link_arr = np.memmap(corpus_file + '_link.bin', dtype=np.uint32, mode=mmap_mode)
            self._mention_arr = np.memmap(corpus_file + '_mention.bin', dtype=np.int32,
                                          mode=mmap_mode)

        meta = joblib.load(corpus_file + '_meta.pkl', mmap_mode=mmap_mode)

        self.tokenizer = meta['tokenizer']
        self.entity_linker = meta['entity_linker']
        self.sentence_tokenizer = meta['sentence_tokenizer']
        self.word_vocab = self.tokenizer.vocab
        self.title_vocab = meta['title_vocab']

        self._word_offsets = meta['word_offsets']
        self._link_offsets = meta['link_offsets']
        self._mention_offsets = meta['mention_offsets']
        self._page_ids = meta['page_ids']
        self._page_offsets = meta['page_offsets']

    @property
    def page_size(self):
        return self._page_offsets.size - 1

    @property
    def word_size(self):
        return self._word_arr.shape[0]

    def iterate_links(self):
        for index in range(0, self._link_arr.size, 3):
            yield Link(self.title_vocab.restore_key(self._link_arr[index]),
                       *self._link_arr[index+1:index+3])

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
        entity_id = self._page_ids[index]
        entity = Entity(self.title_vocab.restore_key(entity_id))

        sentences = [self._read_sentence(i)
                     for i in range(self._page_offsets[index], self._page_offsets[index+1])]

        return Page(entity, sentences)

    def _read_sentence(self, index):
        word_ids = self._word_arr[self._word_offsets[index]:self._word_offsets[index+1]]
        words = [Word(i, self.word_vocab) for i in word_ids]

        link_data = self._link_arr[self._link_offsets[index]:self._link_offsets[index+1]]
        links = [Link(self.title_vocab.restore_key(link_data[i]), *link_data[i+1:i+3])
                 for i in range(0, link_data.size, 3)]

        mention_data = self._mention_arr[self._mention_offsets[index]:self._mention_offsets[index+1]]
        mentions = []
        for n in range(0, mention_data.size, 7):
            (title_index, start, end, link_prob, prior_prob, rank, label) = mention_data[n:n+7]
            title = self.title_vocab.restore_key(title_index)
            link_prob /= 100
            prior_prob /= 100
            mentions.append(MentionCandidate(title, start, end, link_prob, prior_prob, rank, label))

        return Sentence(words, links, mentions)

    @staticmethod
    def build_corpus_data(dump_db, tokenizer, sentence_tokenizer, entity_linker, out_file, target,
                          min_sentence_len, pool_size):
        word_offsets = [0]
        link_offsets = [0]
        mention_offsets = [0]
        page_ids = []
        page_offsets = [0]

        word_offset = 0
        link_offset = 0
        mention_offset = 0

        target_titles = [
            title for title in dump_db.titles()
            if not (':' in title and title.lower().split(':')[0] in ('image', 'file', 'category'))
        ]
        title_vocab = Trie(target_titles)

        try:
            word_file = open(out_file + '_word.bin', mode='wb')
            link_file = open(out_file + '_link.bin', mode='wb')
            mention_file = open(out_file + '_mention.bin', mode='wb')

            with tqdm(total=len(target_titles)) as bar:
                with Pool(pool_size, initializer=_init_worker, initargs=(dump_db, title_vocab,
                    tokenizer, sentence_tokenizer, entity_linker, target,
                    min_sentence_len)) as pool:
                    for (page_id, data) in pool.imap(_process_page, target_titles):
                        if data:
                            for (word_arr, link_arr, mention_arr) in data:
                                word_file.write(word_arr.tobytes())
                                word_offset += word_arr.size
                                word_offsets.append(word_offset)

                                link_file.write(link_arr.tobytes())
                                link_offset += link_arr.size
                                link_offsets.append(link_offset)

                                mention_file.write(mention_arr.tobytes())
                                mention_offset += mention_arr.size
                                mention_offsets.append(mention_offset)

                            page_ids.append(page_id)
                            page_offsets.append(len(word_offsets) - 1)

                        bar.update()

        finally:
            word_file.close()
            link_file.close()
            mention_file.close()

        word_offsets = np.array(word_offsets, dtype=np.uint64)
        link_offsets = np.array(link_offsets, dtype=np.uint64)
        mention_offsets = np.array(mention_offsets, dtype=np.uint64)
        page_ids = np.array(page_ids, dtype=np.uint32)
        page_offsets = np.array(page_offsets, dtype=np.uint32)

        joblib.dump(dict(
            tokenizer=tokenizer,
            sentence_tokenizer=sentence_tokenizer,
            entity_linker=entity_linker,
            title_vocab=title_vocab,
            word_offsets=word_offsets,
            link_offsets=link_offsets,
            mention_offsets=mention_offsets,
            page_ids=page_ids,
            page_offsets=page_offsets,
        ), out_file + '_meta.pkl')


_dump_db = None
_title_vocab = None
_tokenizer = None
_sentence_tokenizer = None
_entity_linker = None
_target = None
_min_sentence_len = None


def _init_worker(dump_db, title_vocab, tokenizer, sentence_tokenizer, entity_linker,
                 target, min_sentence_len):
    global _dump_db, _title_vocab, _tokenizer, _sentence_tokenizer, _entity_linker, _target,\
        _min_sentence_len

    _dump_db = dump_db
    _title_vocab = title_vocab
    _tokenizer = tokenizer
    _sentence_tokenizer = sentence_tokenizer
    _entity_linker = entity_linker
    _target = target
    _min_sentence_len = min_sentence_len


def _process_page(page_title):
    data = []

    for paragraph in _dump_db.get_paragraphs(page_title):
        if _target == 'abstract' and not paragraph.abstract:
            continue

        paragraph_links = []
        for link in paragraph.wiki_links:
            link_title = _dump_db.resolve_redirect(link.title)
            if link_title in _title_vocab:
                paragraph_links.append((link_title, link.span))

        for (sent_start, sent_end) in _sentence_tokenizer.span_tokenize(paragraph.text):
            sent_text = paragraph.text[sent_start:sent_end]

            if len(sent_text.split()) < _min_sentence_len:
                continue

            tokens = _tokenizer.tokenize(sent_text)
            word_arr = np.array([t.id for t in tokens], dtype=np.uint32)

            token_start_map = np.full(len(sent_text), -1)
            token_end_map = np.full(len(sent_text), -1)
            for (ind, token) in enumerate(tokens):
                token_start_map[token.start] = ind
                token_end_map[token.end - 1] = ind

            links = []
            gold_mention_map = np.full(len(sent_text), -1)
            for (link_title, (link_start, link_end)) in paragraph_links:
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

                title_id = _title_vocab[link_title]
                links.extend([title_id, token_start, token_end])

                gold_mention_map[token_start:token_end] = title_id

            detected_mentions = []
            for (mention_span, mentions) in _entity_linker.detect_mentions(sent_text):
                token_start = token_start_map[mention_span[0]]
                if token_start == -1:
                    continue

                token_end = token_end_map[mention_span[1] - 1]
                if token_end == -1:
                    continue
                token_end += 1

                title_ids = [_title_vocab.get(m.title) for m in mentions]

                is_gold_mention = False
                gold_title_ids = frozenset([i for i in gold_mention_map[token_start:token_end]
                                            if i != -1])
                if len(gold_title_ids) == 1:
                    gold_title_id = next(iter(gold_title_ids))
                    if gold_title_id in title_ids:
                        is_gold_mention = True

                for (rank, (mention, title_id)) in enumerate(zip(mentions, title_ids)):
                    if title_id is None:
                        continue

                    if is_gold_mention:
                        label = int(title_id == gold_title_id)
                    else:
                        label = -1

                    link_prob = int(mention.link_prob * 100)
                    prior_prob = int(mention.prior_prob * 100)

                    detected_mentions.extend([title_id, token_start, token_end, link_prob,
                                              prior_prob, rank, label])

            link_arr = np.array(links, dtype=np.uint32)
            mention_arr = np.array(detected_mentions, dtype=np.int32)
            data.append((word_arr, link_arr, mention_arr))

    return (_title_vocab[page_title], data)
