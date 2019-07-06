import random
from contextlib import closing
from multiprocessing.pool import Pool
import joblib
import marisa_trie
import numpy as np
import tqdm


class Word(object):
    __slots__ = ('id', 'text')

    def __init__(self, id_, text):
        self.id = id_
        self.text = text

    def __repr__(self):
        return f'<Word {self.text}>'

    @property
    def is_subword(self):
        return self.text.startswith('##')


class Entity(object):
    __slots__ = ('title',)

    def __init__(self, title):
        self.title = title

    def __repr__(self):
        return f'<Entity {self.title}>'


class Link(Entity):
    __slots__ = ('start', 'end', '_candidates')

    def __init__(self, title, start, end, candidates):
        super(Link, self).__init__(title)

        self.start = start
        self.end = end
        self._candidates = candidates  # differ instantiating candidates for efficiency

    @property
    def span(self):
        return (self.start, self.end)

    @property
    def candidates(self):
        return tuple(self._candidates)

    @candidates.setter
    def candidates(self, candidates):
        self._candidates = candidates

    def __repr__(self):
        return f'<Link title={self.title} start={self.start} end={self.end}>'


class Candidate(Entity):
    __slots__ = ('prior_prob',)

    def __init__(self, title, prior_prob):
        super(Candidate, self).__init__(title)

        self.prior_prob = prior_prob

    def __repr__(self):
        return f'<Candidate title={self.title} prior_prob={self.prior_prob:.3f}>'


class Page(object):
    __slots__ = ('entity', 'sentences')

    def __init__(self, entity, sentences):
        self.entity = entity
        self.sentences = sentences

    def __repr__(self):
        return f'<Page {self.entity.title}>'


class Sentence(object):
    __slots__ = ('words', 'links')

    def __init__(self, words, links):
        self.words = words
        self.links = links

    def __repr__(self):
        return '<Sentence {}...>'.format(' '.join(w.text for w in self.words)[:100])


# global variables used in pool workers
_dump_db = _title_vocab = _tokenizer = _sentence_tokenizer = _entity_linker = None
_min_sentence_length = _max_candidate_size = None


class WikiCorpus(object):
    def __init__(self, corpus_file, mmap_mode=None):
        data = joblib.load(corpus_file, mmap_mode=mmap_mode)

        self.tokenizer = data['tokenizer']
        self.sentence_tokenizer = data['sentence_tokenizer']
        self.entity_linker = data['entity_linker']
        self.title_vocab = data['title_vocab']
        self._word_arr = data['word_arr']
        self._link_arr = data['link_arr']
        self._word_offsets = data['word_offsets']
        self._link_offsets = data['link_offsets']
        self._page_ids = data['page_ids']
        self._page_offsets = data['page_offsets']
        self._max_candidate_size = data['max_candidate_size']

    @property
    def page_size(self):
        return self._page_offsets.size - 1

    @property
    def word_size(self):
        return self._word_arr.shape[0]

    @property
    def link_size(self):
        return int(self._link_arr.shape[0] / (3 + self._max_candidate_size * 2))

    def iterate_links(self):
        for i in range(0, self._link_arr.size, self._max_candidate_size * 2 + 3):
            candidates = (Candidate(self.title_vocab.restore_key(self._link_arr[j]), self._link_arr[j + 1] / 10000)
                          for j in range(i + 3, i + 3 + self._max_candidate_size * 2, 2) if self._link_arr[j] != -1)
            yield Link(self.title_vocab.restore_key(self._link_arr[i]), *self._link_arr[i + 1:i + 3],
                       candidates=candidates)

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

        sentences = [self._read_sentence(i) for i in range(self._page_offsets[index], self._page_offsets[index + 1])]

        return Page(entity, sentences)

    def _read_sentence(self, index):
        word_ids = self._word_arr[self._word_offsets[index]:self._word_offsets[index + 1]]
        words = [Word(i, self.tokenizer.ids_to_tokens[i]) for i in word_ids]

        links = []
        link_data = self._link_arr[self._link_offsets[index]:self._link_offsets[index + 1]]
        for i in range(0, link_data.size, self._max_candidate_size * 2 + 3):
            candidates = (Candidate(self.title_vocab.restore_key(link_data[j]), link_data[j + 1] / 10000)
                          for j in range(i + 3, i + 3 + self._max_candidate_size * 2, 2) if link_data[j] != -1)
            links.append(Link(self.title_vocab.restore_key(link_data[i]), *link_data[i + 1:i + 3],
                              candidates=candidates))

        return Sentence(words, links)

    @staticmethod
    def build(dump_db, tokenizer, sentence_tokenizer, entity_linker, out_file, min_sentence_length, max_candidate_size,
              pool_size, chunk_size):
        target_titles = [title for title in dump_db.titles()
                         if not (':' in title and title.lower().split(':')[0] in ('image', 'file', 'category'))]
        title_vocab = marisa_trie.Trie(target_titles)

        word_arr = []
        link_arr = []
        word_offsets = [0]
        link_offsets = [0]
        page_ids = []
        page_offsets = [0]

        word_offset = 0
        link_offset = 0

        with tqdm.tqdm(total=len(target_titles)) as bar:
            initargs = (dump_db, title_vocab, tokenizer, sentence_tokenizer, entity_linker, min_sentence_length,
                        max_candidate_size)
            with closing(Pool(pool_size, initializer=WikiCorpus._initialize_worker, initargs=initargs)) as pool:
                for (page_id, data) in pool.imap(WikiCorpus._process_page, target_titles, chunksize=chunk_size):
                    if data:
                        for (word_data, link_data) in data:
                            word_arr.append(word_data)
                            word_offset += word_data.size
                            word_offsets.append(word_offset)

                            link_arr.append(link_data)
                            link_offset += link_data.size
                            link_offsets.append(link_offset)

                        page_ids.append(page_id)
                        page_offsets.append(len(word_offsets) - 1)

                    bar.update()

        word_arr = np.concatenate(word_arr)
        link_arr = np.concatenate(link_arr)
        word_offsets = np.array(word_offsets, dtype=np.uint64)
        link_offsets = np.array(link_offsets, dtype=np.uint64)
        page_ids = np.array(page_ids, dtype=np.uint32)
        page_offsets = np.array(page_offsets, dtype=np.uint32)

        joblib.dump(dict(
            tokenizer=tokenizer,
            sentence_tokenizer=sentence_tokenizer,
            entity_linker=entity_linker,
            title_vocab=title_vocab,
            word_arr=word_arr,
            link_arr=link_arr,
            word_offsets=word_offsets,
            link_offsets=link_offsets,
            page_ids=page_ids,
            page_offsets=page_offsets,
            max_candidate_size=max_candidate_size,
        ), out_file)

    @staticmethod
    def _initialize_worker(dump_db, title_vocab, tokenizer, sentence_tokenizer, entity_linker, min_sentence_length,
                           max_candidate_size):
        global _dump_db, _title_vocab, _tokenizer, _sentence_tokenizer, _entity_linker, _min_sentence_length,\
            _max_candidate_size

        _dump_db = dump_db
        _title_vocab = title_vocab
        _tokenizer = tokenizer
        _sentence_tokenizer = sentence_tokenizer
        _entity_linker = entity_linker
        _min_sentence_length = min_sentence_length
        _max_candidate_size = max_candidate_size

    @staticmethod
    def _process_page(page_title):
        data = []

        for paragraph in _dump_db.get_paragraphs(page_title):
            paragraph_text = paragraph.text
            paragraph_links = []
            for link in paragraph.wiki_links:
                link_title = _dump_db.resolve_redirect(link.title)
                if link_title.startswith('Category:') and link.text.lower().startswith('category:'):
                    paragraph_text = paragraph_text[:link.start] + ' ' * \
                        (link.end - link.start) + paragraph_text[link.end:]
                else:
                    if link_title in _title_vocab:
                        paragraph_links.append((link_title, link.span))

            for (sent_start, sent_end) in _sentence_tokenizer.span_tokenize(paragraph_text):
                sent_text = paragraph_text[sent_start:sent_end]

                if len(_tokenizer.tokenize(sent_text)) < _min_sentence_length:
                    continue

                cur = 0
                words = []
                links = []
                for (link_title, (link_start, link_end)) in paragraph_links:
                    if not (sent_start <= link_start < sent_end and link_end <= sent_end):
                        continue

                    try:
                        title_id = _title_vocab[link_title]
                    except KeyError:
                        continue

                    link_start -= sent_start
                    link_end -= sent_start

                    words += _tokenizer.tokenize(sent_text[cur:link_start])
                    link_text = sent_text[link_start:link_end]
                    link_words = _tokenizer.tokenize(link_text)
                    links.append((title_id, link_text, len(words), len(words) + len(link_words)))
                    words += link_words

                    cur = link_end

                words += _tokenizer.tokenize(sent_text[cur:])

                link_data = []
                for (title_id, link_text, start, end) in links:
                    candidate_data = [-1] * _max_candidate_size * 2
                    index = 0
                    for mention in sorted(_entity_linker.query(link_text), reverse=True, key=lambda c: c.prior_prob):
                        if mention.title in _title_vocab:
                            mention_title_id = _title_vocab[mention.title]
                            candidate_data[index * 2] = mention_title_id
                            candidate_data[index * 2 + 1] = int(mention.prior_prob * 10000)
                            index += 1
                            if index == _max_candidate_size:
                                break

                    link_data.extend([title_id, start, end] + candidate_data)

                word_data = np.array(_tokenizer.convert_tokens_to_ids(words), dtype=np.int32)
                link_data = np.array(link_data, dtype=np.int32)
                data.append((word_data, link_data))

        return (_title_vocab[page_title], data)
