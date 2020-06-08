# -*- coding: utf-8 -*-
from typing import List

import bz2
import logging
import re
import click
import joblib
import numpy as np
import ujson
from marisa_trie import Trie

KEY_RULE = re.compile("^(.*):([^:]+)$")

logger = logging.getLogger(__name__)


@click.command()
@click.argument("wikidata_dump_file", type=click.Path(exists=True))
@click.option("-l", "--language", multiple=True)
@click.argument("out_file", type=click.Path())
def build(wikidata_dump_file: str, language: List[str], out_file: str):
    logging.basicConfig(level=logging.INFO)
    if language:
        language = frozenset(language)
    interwiki_db = InterwikiDB.build(wikidata_dump_file, language)
    interwiki_db.save(out_file)


class InterwikiDB(object):
    def __init__(
        self,
        title_trie: Trie,
        data: np.ndarray,
        indptr: np.ndarray,
        title_indices: np.ndarray,
    ):
        self._title_trie = title_trie
        self._data = data
        self._indptr = indptr
        self._title_indices = title_indices

    def query(self, title: str, lang: str):
        try:
            key = "%s:%s" % (title, lang)
            row = self._title_indices[self._title_trie[key]]
            objs = [
                KEY_RULE.match(self._title_trie.restore_key(ind))
                for ind in self._data[self._indptr[row] : self._indptr[row + 1]]
            ]
            return [(o.group(1), o.group(2)) for o in objs]

        except KeyError:
            return []

    @staticmethod
    def build(wiki_data_file: str, target_languages: List[str] = None):
        data = []
        indptr = [0]
        titles = []
        title_indices = []

        with bz2.BZ2File(wiki_data_file) as f:
            for (n, line) in enumerate(f):
                if n % 1000 == 0 and n != 0:
                    logger.info("Processed %d lines", n)

                if n == 10000:
                    logger.info("Stop")

                    break

                try:
                    line = line.rstrip().decode("utf-8")
                    if line in ("[", "]"):
                        continue

                    if line[-1] == ",":
                        line = line[:-1]
                    obj = ujson.loads(line)
                    if obj["type"] != "item":
                        continue

                    for link_obj in obj["sitelinks"].values():
                        site = link_obj["site"]
                        if not site.endswith("wiki"):
                            continue
                        lang = site[:-4]
                        if target_languages and lang not in target_languages:
                            continue

                        title_indices.append(len(indptr) - 1)
                        data.append(len(titles))

                        title = "%s:%s" % (link_obj["title"], lang)
                        titles.append(title)

                    indptr.append(len(data))

                except BaseException:
                    logging.exception("")
                    print(line)

        title_trie = Trie(titles)
        data = np.fromiter((title_trie[titles[n]] for n in data), dtype=np.int)
        indptr = np.array(indptr, dtype=np.int)
        new_title_indices = np.empty(len(titles), dtype=np.int)
        for (title, index) in zip(titles, title_indices):
            new_title_indices[title_trie[title]] = index

        return InterwikiDB(title_trie, data, indptr, new_title_indices)

    def save(self, out_file: str):
        joblib.dump(
            dict(
                title_trie=self._title_trie.tobytes(),
                data=self._data,
                indptr=self._indptr,
                title_indices=self._title_indices,
            ),
            out_file,
        )

    @staticmethod
    def load(in_file: str, mmap_mode="r"):
        data = joblib.load(in_file, mmap_mode=mmap_mode)
        title_trie = Trie()
        title_trie = title_trie.frombytes(data["title_trie"])
        data["title_trie"] = title_trie

        return InterwikiDB(**data)


if __name__ == "__main__":
    build()
