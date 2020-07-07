from typing import List, TextIO, Dict
import json
import math

import multiprocessing
from collections import Counter, OrderedDict, defaultdict, namedtuple
from contextlib import closing
from multiprocessing.pool import Pool

import click
from tqdm import tqdm
from wikipedia2vec.dump_db import DumpDB

from .interwiki_db import InterwikiDB

PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
MASK_TOKEN = "[MASK]"

Entity = namedtuple("Entity", ["title", "language"])

_dump_db = None  # global variable used in multiprocessing workers


@click.command()
@click.argument("dump_db_file", type=click.Path())
@click.argument("out_file", type=click.Path())
@click.option("--vocab-size", default=1000000)
@click.option("-w", "--white-list", type=click.File(), multiple=True)
@click.option("--white-list-only", is_flag=True)
@click.option("--pool-size", default=multiprocessing.cpu_count())
@click.option("--chunk-size", default=100)
def build_entity_vocab(dump_db_file: str, white_list: List[TextIO], **kwargs):
    dump_db = DumpDB(dump_db_file)
    white_list = [line.rstrip() for f in white_list for line in f]
    EntityVocab.build(dump_db, white_list=white_list, **kwargs)


class EntityVocab(object):
    def __init__(self, vocab_file: str):
        self._vocab_file = vocab_file

        self.vocab: Dict[Entity, int] = {}
        self.counter: Dict[Entity, int] = {}
        self.inv_vocab: Dict[int, List[Entity]] = defaultdict(list)

        if vocab_file.endswith(".tsv"):
            self._parse_tsv_vocab_file(vocab_file)
        else:
            self._parse_json_vocab_file(vocab_file)

    def _parse_tsv_vocab_file(self, vocab_file: str):
        with open(vocab_file, "r") as f:
            for (index, line) in enumerate(f):
                title, count = line.rstrip().split("\t")
                entity = Entity(title, None)
                self.vocab[entity] = index
                self.counter[entity] = int(count)
                self.inv_vocab[index] = [entity]

    def _parse_json_vocab_file(self, vocab_file: str):
        entities_json = json.load(open(vocab_file, "r"))
        for ent_id, record in enumerate(entities_json):
            for title, language in record["entities"]:
                entity = Entity(title, language)
                self.vocab[entity] = ent_id
                self.counter[entity] = record["count"]
                self.inv_vocab[ent_id].append(entity)

    @property
    def size(self) -> int:
        return len(self)

    def __reduce__(self):
        return (self.__class__, (self._vocab_file,))

    def __len__(self):
        return len(self.inv_vocab)

    def __contains__(self, item: str):
        return self.contains(item, language=None)

    def __getitem__(self, key: str):
        return self.get_id(key, language=None)

    def __iter__(self):
        return iter(self.vocab)

    def contains(self, title: str, language: str = None):
        return Entity(title, language) in self.vocab

    def get_id(self, title: str, language: str = None, default: int = None) -> int:
        try:
            return self.vocab[Entity(title, language)]
        except KeyError:
            return default

    def get_title_by_id(self, id_: int, language: str = None) -> str:
        for entity in self.inv_vocab[id_]:
            if entity.language == language:
                return entity.title

    def get_count_by_title(self, title: str, language: str = None) -> int:
        entity = Entity(title, language)
        return self.counter.get(entity, 0)

    def save(self, out_file: str):
        with open(self._vocab_file, "r") as src:
            with open(out_file, "w") as dst:
                dst.write(src.read())

    @staticmethod
    def build(
        dump_db: DumpDB,
        out_file: str,
        vocab_size: int,
        white_list: List[str],
        white_list_only: bool,
        pool_size: int,
        chunk_size: int,
    ):
        counter = Counter()
        with tqdm(total=dump_db.page_size(), mininterval=0.5) as pbar:
            with closing(Pool(pool_size, initializer=EntityVocab._initialize_worker, initargs=(dump_db,))) as pool:
                for ret in pool.imap_unordered(EntityVocab._count_entities, dump_db.titles(), chunksize=chunk_size):
                    counter.update(ret)
                    pbar.update()

        title_dict = OrderedDict()
        title_dict[PAD_TOKEN] = 0
        title_dict[UNK_TOKEN] = 0
        title_dict[MASK_TOKEN] = 0

        for title in white_list:
            if counter[title] != 0:
                title_dict[title] = counter[title]

        if not white_list_only:
            valid_titles = frozenset(dump_db.titles())
            for title, count in counter.most_common():
                if title in valid_titles and not title.startswith("Category:"):
                    title_dict[title] = count
                    if len(title_dict) == vocab_size:
                        break

        with open(out_file, "w") as f:
            for title, count in title_dict.items():
                f.write("%s\t%d\n" % (title, count))

    @staticmethod
    def _initialize_worker(dump_db: DumpDB):
        global _dump_db
        _dump_db = dump_db

    @staticmethod
    def _count_entities(title: str) -> Dict[str, int]:
        counter = Counter()
        for paragraph in _dump_db.get_paragraphs(title):
            for wiki_link in paragraph.wiki_links:
                title = _dump_db.resolve_redirect(wiki_link.title)
                counter[title] += 1
        return counter


@click.command()
@click.option("entity_vocab_files", "-v", multiple=True)
@click.option("inter_wiki_db_path", "-i", type=click.Path())
@click.option("out_file", "-o", type=click.Path())
@click.option("vocab_size", "-s", type=int)
def build_multilingual_entity_vocab(
    entity_vocab_files: List[str], inter_wiki_db_path: str, out_file: str, vocab_size: int = 1000000
):

    try:
        languages, vocab_paths = zip(*map(lambda x: x.split(":"), entity_vocab_files))
    except ValueError:
        raise RuntimeError(
            "Each element of ``vocab_files`` must specify language_code and vocab_file_path"
            "in the form or `{language_code}:{vocab_file_path}`."
        )

    db = InterwikiDB.load(inter_wiki_db_path)

    vocab: Dict[Entity, int] = {}  # title -> index
    inv_vocab = defaultdict(set)  # index -> Set[title]
    count_dict = defaultdict(int)  # index -> count

    new_id = 0
    for vocab_path, lang in zip(vocab_paths, languages):
        with open(vocab_path, "r") as f:
            for line in f:
                title, count = line.strip().split("\t")
                count = int(count)

                # append the language code to the entity name to distinguish homographs across languages
                # e.g., "fr:Apple" -> IT corporation, "en:Apple" -> fruit
                entity = Entity(title, lang)
                multilingual_entities = {entity}
                if title not in {PAD_TOKEN, UNK_TOKEN, MASK_TOKEN}:
                    aligned_entities = {Entity(t, ln) for t, ln in db.query(title, lang)}
                else:
                    # for special tokens, we don't need to ask inter_wiki_db to get aligned entities
                    aligned_entities = {Entity(title, ln) for ln in languages}
                multilingual_entities.update(aligned_entities)

                # judge if we should assign a new id to these entities
                use_new_id = True
                for ent in multilingual_entities:
                    if ent in vocab:
                        # if any of multilingual entities is found in the current vocab, we use the existing index.
                        ent_id = vocab[ent]
                        use_new_id = False
                        break

                if use_new_id:
                    ent_id = new_id
                    new_id += 1

                vocab[entity] = ent_id
                inv_vocab[ent_id].add((entity.title, entity.language))  # Convert Entity to Tuple for json.dump
                count_dict[ent_id] += count
    json_dicts = [{"entities": list(inv_vocab[ent_id]), "count": count_dict[ent_id]} for ent_id in range(new_id)]
    json_dicts.sort(key=lambda x: -x["count"] if x["count"] != 0 else -math.inf)
    json_dicts = json_dicts[:vocab_size]

    with open(out_file, "w") as f:
        json.dump(json_dicts, f, indent=4)
