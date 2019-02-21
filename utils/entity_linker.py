# -*- coding: utf-8 -*-

import logging
import re
from collections import defaultdict, Counter
from contextlib import closing
from functools import partial
from multiprocessing.pool import Pool
import joblib
from marisa_trie import Trie, RecordTrie
from tqdm import tqdm

from .word_tokenizer import BasicTokenizer

logger = logging.getLogger(__name__)


class MentionCandidate(object):
    __slots__ = ('title', 'text', 'link_count', 'total_link_count', 'doc_count')

    def __init__(self, title, text, link_count, total_link_count, doc_count):
        self.title = title
        self.text = text
        self.link_count = link_count
        self.total_link_count = total_link_count
        self.doc_count = doc_count

    @property
    def link_prob(self):
        if self.doc_count > 0:
            return min(1.0, float(self.total_link_count) / self.doc_count)
        else:
            return 0.0

    @property
    def prior_prob(self):
        if self.total_link_count > 0:
            return min(1.0, float(self.link_count) / self.total_link_count)
        else:
            return 0.0

    def __repr__(self):
        return '<MentionCandidate %s -> %s>' % (self.text, self.title)


class EntityLinker(object):
    def __init__(self, mention_db):
        self._mention_db = mention_db
        self._tokenizer = BasicTokenizer()

    def detect_mentions(self, text):
        tokens = self._tokenizer.tokenize(text)
        end_offsets = frozenset(t.span[1] for t in tokens)

        ret = defaultdict(list)
        cur = 0
        for token in tokens:
            start = token.span[0]
            if cur > start:
                continue

            for prefix in self._mention_db.prefix_search(text, start):
                end = start + len(prefix)
                if end in end_offsets:
                    matched = False

                    for mention in self._mention_db.query(prefix):
                        ret[(start, end)].append(mention)
                        cur = end
                        matched = True

                    if matched:
                        break

        return tuple(ret.items())


class MentionDB(object):
    __slots__ = ('_title_trie', '_mention_trie', '_data_trie', '_max_mention_len')

    def __init__(self, title_trie, mention_trie, data_trie, max_mention_len):
        self._title_trie = title_trie
        self._mention_trie = mention_trie
        self._data_trie = data_trie
        self._max_mention_len = max_mention_len

    def query(self, text):
        return [MentionCandidate(self._title_trie.restore_key(args[0]), text, *args[1:])
                for args in self._data_trie[text.lower()]]

    def prefix_search(self, text, start=0):
        target_text = text[start:start+self._max_mention_len].lower()
        return sorted(self._mention_trie.prefixes(target_text), key=len, reverse=True)

    @staticmethod
    def build(dump_db, min_link_prob, max_candidate_size, min_link_count, max_mention_len,
              pool_size, chunk_size):
        name_dict = defaultdict(lambda: Counter())
        init_args = [dump_db, None]

        logger.info('Step 1/4: Starting to iterate over Wikipedia pages...')

        with closing(Pool(pool_size, initializer=init_worker, initargs=init_args)) as pool:
            with tqdm(total=dump_db.page_size(), mininterval=0.5) as bar:
                f = partial(_extract_links, max_mention_len=max_mention_len)
                for ret in pool.imap_unordered(f, dump_db.titles(), chunksize=chunk_size):
                    for (text, title) in ret:
                        name_dict[text][title] += 1
                    bar.update(1)

        logger.info('Step 2/4: Procesing Wikipedia titles...')
        name_counter = Counter()

        disambi_matcher = re.compile(r'\s\(.*\)$')
        for title in dump_db.titles():
            text = disambi_matcher.sub('', title).lower()
            name_dict[text][title] += 1
            name_counter[text] += 1

        for (src, dest) in dump_db.redirects():
            text = disambi_matcher.sub('', src).lower()
            name_dict[text][dest] += 1
            name_counter[text] += 1

        logger.info('Step 3/4: Starting to count occurrences...')

        name_trie = Trie(name_dict.keys())
        init_args[1] = name_trie

        with closing(Pool(pool_size, initializer=init_worker, initargs=init_args)) as pool:
            with tqdm(total=dump_db.page_size(), mininterval=0.5) as bar:
                f = partial(_count_occurrences, max_mention_len=max_mention_len)
                for names in pool.imap_unordered(f, dump_db.titles(), chunksize=chunk_size):
                    name_counter.update(names)
                    bar.update(1)

        logger.info('Step 4/4: Building DB...')

        titles = frozenset([title for cnt in name_dict.values() for title in cnt.keys()])
        title_trie = Trie(titles)

        def item_generator():
            for (name, entity_counter) in name_dict.items():
                doc_count = name_counter[name]
                total_link_count = sum(entity_counter.values())

                if doc_count == 0:
                    continue

                link_prob = float(total_link_count) / doc_count
                if link_prob < min_link_prob:
                    continue

                for (title, link_count) in entity_counter.most_common()[:max_candidate_size]:
                    if link_count < min_link_count:
                        continue
                    yield (name, (title_trie[title], link_count, total_link_count, doc_count))

        data_trie = RecordTrie('<IIII', item_generator())
        mention_trie = Trie(data_trie.keys())

        return MentionDB(title_trie, mention_trie, data_trie, max_mention_len)

    def save(self, out_file):
        joblib.dump(dict(title_trie=self._title_trie.tobytes(),
                         mention_trie=self._mention_trie.tobytes(),
                         data_trie=self._data_trie.tobytes(),
                         max_mention_len=self._max_mention_len), out_file)

    @staticmethod
    def load(in_file):
        obj = joblib.load(in_file)

        title_trie = Trie()
        title_trie = title_trie.frombytes(obj.pop('title_trie'))
        mention_trie = Trie()
        mention_trie = mention_trie.frombytes(obj.pop('mention_trie'))
        data_trie = RecordTrie('<IIII')
        data_trie = data_trie.frombytes(obj.pop('data_trie'))

        return MentionDB(title_trie, mention_trie, data_trie, **obj)


_dump_db = None
_tokenizer = None
_name_trie = None


def init_worker(dump_db, name_trie):
    global _dump_db, _tokenizer, _name_trie

    _dump_db = dump_db
    _name_trie = name_trie
    _tokenizer = BasicTokenizer()


def _extract_links(title, max_mention_len):
    ret = []

    for paragraph in _dump_db.get_paragraphs(title):
        for wiki_link in paragraph.wiki_links:
            text = wiki_link.text.lower()
            if len(text) > max_mention_len:
                continue

            ret.append((text, _dump_db.resolve_redirect(wiki_link.title)))

    return ret


def _count_occurrences(title, max_mention_len):
    ret = []

    for paragraph in _dump_db.get_paragraphs(title):
        text = paragraph.text.lower()
        tokens = _tokenizer.tokenize(text)

        end_offsets = frozenset(token.end for token in tokens)

        for token in tokens:
            start = token.start
            for prefix in _name_trie.prefixes(text[start:start+max_mention_len]):
                if (start + len(prefix)) in end_offsets:
                    ret.append(prefix)

    return frozenset(ret)
