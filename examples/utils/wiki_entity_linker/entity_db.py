import logging
import multiprocessing
from collections import Counter, defaultdict
from contextlib import closing
from multiprocessing.pool import Pool

import click
import joblib
import marisa_trie
from tqdm import tqdm
from wikipedia2vec.dump_db import DumpDB

logger = logging.getLogger(__name__)


@click.group(name="entity-db")
def cli():
    pass


@cli.command()
@click.argument("dump_db_file", type=click.Path(exists=True))
@click.argument("out_file", type=click.Path())
@click.option("--max-candidate-size", default=100)
@click.option("--min-mention-count", default=1)
@click.option("--pool-size", default=multiprocessing.cpu_count())
@click.option("--chunk-size", default=100)
def build_from_wikipedia(dump_db_file, **kwargs):
    dump_db = DumpDB(dump_db_file)
    EntityDB.build_from_wikipedia(dump_db, **kwargs)


# global variables used in pool workers
_dump_db = _name_trie = None


class EntityDB(object):
    def __init__(self, entity_db_file: str):
        self.entity_db_file = entity_db_file

        data = joblib.load(entity_db_file)
        self._title_trie = data["title_trie"]
        self._mention_trie = data["mention_trie"]
        self._data_trie = data["data_trie"]

    def __reduce__(self):
        return (self.__class__, (self.entity_db_file,))

    def query(self, title: str):
        try:
            return [(title, self._mention_trie.restore_key(args[0]), *args[1:]) for args in self._data_trie[title]]
        except KeyError:
            return []

    def save(self, out_file: str):
        joblib.dump(
            dict(title_trie=self._title_trie, mention_trie=self._mention_trie, data_trie=self._data_trie,), out_file,
        )

    @staticmethod
    def build_from_wikipedia(
        dump_db: DumpDB, out_file, max_candidate_size, min_mention_count, pool_size, chunk_size,
    ):
        logger.info("Extracting all entity names...")

        title_dict = defaultdict(Counter)
        with tqdm(total=dump_db.page_size(), mininterval=0.5) as pbar:
            with closing(Pool(pool_size, initializer=EntityDB._initialize_worker, initargs=(dump_db,))) as pool:
                for ret in pool.imap_unordered(
                    EntityDB._extract_name_entity_pairs, dump_db.titles(), chunksize=chunk_size
                ):
                    for (name, title) in ret:
                        title_dict[title][name] += 1
                    pbar.update()

        logger.info("Building DB...")

        mentions = frozenset([mention for mention_counter in title_dict.values() for mention in mention_counter.keys()])
        title_trie = frozenset(title_dict.keys())
        mention_trie = marisa_trie.Trie(mentions)

        def item_generator():
            for (title, mention_counter) in title_dict.items():
                for (mention, mention_count) in mention_counter.most_common()[:max_candidate_size]:
                    if mention_count < min_mention_count:
                        continue
                    yield (title, (mention_trie[mention], mention_count))

        data_trie = marisa_trie.RecordTrie("<II", item_generator())

        joblib.dump(
            dict(title_trie=title_trie, mention_trie=mention_trie, data_trie=data_trie,), out_file,
        )

    @staticmethod
    def _initialize_worker(dump_db, name_trie=None):
        global _dump_db, _name_trie
        _dump_db = dump_db
        _name_trie = name_trie

    @staticmethod
    def _extract_name_entity_pairs(article_title: str):
        ret = []
        for paragraph in _dump_db.get_paragraphs(article_title):
            for wiki_link in paragraph.wiki_links:
                link_title = _dump_db.resolve_redirect(wiki_link.title)
                if link_title.startswith("Category:"):
                    continue
                mention_text = wiki_link.text
                ret.append((mention_text, link_title))
        return ret
