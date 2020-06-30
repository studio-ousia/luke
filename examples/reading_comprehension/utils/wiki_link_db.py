import logging
from contextlib import closing
from multiprocessing.pool import Pool

import joblib
import marisa_trie
from tqdm import tqdm

logger = logging.getLogger(__name__)


class WikiLink(object):
    __slots__ = ("title", "text", "link_prob")

    def __init__(self, title, text, link_prob):
        self.title = title
        self.text = text
        self.link_prob = link_prob


_dump_db = _mention_db = _title_trie = None


class WikiLinkDB(object):
    def __init__(self, wiki_link_db_file):
        self._wiki_link_db_file = wiki_link_db_file
        data = joblib.load(wiki_link_db_file)

        self._title_trie = data["title_trie"]
        self._mention_trie = data["mention_trie"]
        self._data_trie = data["data_trie"]

    def __reduce__(self):
        return (self.__class__, (self._wiki_link_db_file,))

    def __getitem__(self, title):
        return self.get(title)

    def get(self, title):
        if title not in self._data_trie:
            return []
        return [
            WikiLink(self._mention_trie.restore_key(text_id), self._title_trie.restore_key(title_id), link_prob)
            for text_id, title_id, link_prob in self._data_trie[title]
        ]

    def save(self, out_file):
        joblib.dump(
            dict(title_trie=self._title_trie, mention_trie=self._mention_trie, data_trie=self._data_trie), out_file
        )

    @staticmethod
    def build(dump_db, mention_db, out_file, pool_size, chunk_size):
        title_trie = marisa_trie.Trie(dump_db.titles())
        data = {}

        with tqdm(total=dump_db.page_size(), mininterval=0.5) as pbar:
            initargs = (dump_db, mention_db, title_trie)
            with closing(Pool(pool_size, initializer=WikiLinkDB._initialize_worker, initargs=initargs)) as pool:
                for title, links in pool.imap_unordered(
                    WikiLinkDB._extract_wiki_links, title_trie, chunksize=chunk_size
                ):
                    data[title] = links
                    pbar.update()

        mention_trie = marisa_trie.Trie(text for links in data.values() for text, _, _ in links)

        def item_generator():
            for title, links in data.items():
                for mention_text, link_title_id, link_prob in links:
                    yield title, (mention_trie[mention_text], link_title_id, link_prob)

        data_trie = marisa_trie.RecordTrie("<IIf", item_generator())

        joblib.dump(dict(title_trie=title_trie, mention_trie=mention_trie, data_trie=data_trie), out_file)

    @staticmethod
    def _initialize_worker(dump_db, mention_db, title_trie):
        global _dump_db, _mention_db, _title_trie
        _dump_db = dump_db
        _mention_db = mention_db
        _title_trie = title_trie

    @staticmethod
    def _extract_wiki_links(title):
        links = []
        for paragraph in _dump_db.get_paragraphs(title):
            for wiki_link in paragraph.wiki_links:
                link_title = _dump_db.resolve_redirect(wiki_link.title)
                if link_title not in _title_trie:
                    continue

                mentions = _mention_db.query(wiki_link.text)
                if mentions:
                    link_prob = mentions[0].link_prob
                else:
                    link_prob = 0.0

                links.append((wiki_link.text, _title_trie[link_title], link_prob))

        return title, links
