import json
import logging
import math
import multiprocessing
from collections import Counter, OrderedDict, defaultdict, namedtuple
from contextlib import closing
from multiprocessing.pool import Pool
from pathlib import Path
from typing import Dict, List, TextIO, Optional

import click
from tqdm import tqdm
from wikipedia2vec.dump_db import DumpDB

from .interwiki_db import InterwikiDB

PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
MASK_TOKEN = "[MASK]"

SPECIAL_TOKENS = {PAD_TOKEN, UNK_TOKEN, MASK_TOKEN}

Entity = namedtuple("Entity", ["title", "language"])

_dump_db = None  # global variable used in multiprocessing workers

logger = logging.getLogger(__name__)


@click.command()
@click.argument("dump_db_file", type=click.Path())
@click.argument("out_file", type=click.Path())
@click.option("--vocab-size", default=1000000)
@click.option("--min-count", default=0)
@click.option("--language", type=str)
@click.option("-w", "--white-list", type=click.File(), multiple=True)
@click.option("--white-list-only", is_flag=True)
@click.option("--pool-size", default=multiprocessing.cpu_count())
@click.option("--chunk-size", default=100)
def build_entity_vocab(dump_db_file: str, white_list: List[TextIO], language: Optional[str], **kwargs):
    dump_db = DumpDB(dump_db_file)
    white_list = [line.rstrip() for f in white_list for line in f]
    EntityVocab.build(dump_db, white_list=white_list, language=language, **kwargs)


class EntityVocab:
    def __init__(self, vocab_file: str):
        self._vocab_file = vocab_file

        self.vocab: Dict[Entity, int] = {}
        self.counter: Dict[Entity, int] = {}
        self.inv_vocab: Dict[int, List[Entity]] = defaultdict(list)

        # allow tsv files for backward compatibility
        if vocab_file.endswith(".tsv"):
            logger.info("Detected vocab file type: tsv")
            self._parse_tsv_vocab_file(vocab_file)
        elif vocab_file.endswith(".jsonl"):
            logger.info("Detected vocab file type: jsonl")
            self._parse_jsonl_vocab_file(vocab_file)
        elif "mluke" in vocab_file:
            logger.info("Detected vocab file type: pretrained transformers")
            self._from_pretrained_mluke(vocab_file)
        elif "luke" in vocab_file:
            logger.info("Detected vocab file type: pretrained transformers")
            self._from_pretrained_luke(vocab_file)
        else:
            raise ValueError(f"Unrecognized vocab_file format: {vocab_file}")

        self.special_token_ids = {}
        for special_token in SPECIAL_TOKENS:
            special_token_entity = self.search_across_languages(special_token)[0]
            self.special_token_ids[special_token] = self.get_id(*special_token_entity)

    def _from_pretrained_mluke(self, transformer_model_name: str):
        from transformers.models.mluke.tokenization_mluke import MLukeTokenizer

        mluke_tokenizer = MLukeTokenizer.from_pretrained(transformer_model_name)
        title_to_idx = mluke_tokenizer.entity_vocab
        mluke_special_tokens = SPECIAL_TOKENS | {"[MASK2]"}
        for title, idx in title_to_idx.items():
            if title in mluke_special_tokens:
                entity = Entity(title, None)
            else:
                language, title = title.split(":", maxsplit=1)
                entity = Entity(title, language)
            self.vocab[entity] = idx
            self.counter[entity] = None
            self.inv_vocab[idx].append(entity)

    def _from_pretrained_luke(self, transformer_model_name: str):
        from transformers.models.luke.tokenization_luke import LukeTokenizer

        luke_tokenizer = LukeTokenizer.from_pretrained(transformer_model_name)
        title_to_idx = luke_tokenizer.entity_vocab
        for title, idx in title_to_idx.items():
            entity = Entity(title, None)
            self.vocab[entity] = idx
            self.counter[entity] = None
            self.inv_vocab[idx].append(entity)

    def _parse_tsv_vocab_file(self, vocab_file: str):
        with open(vocab_file, "r") as f:
            for (index, line) in enumerate(f):
                title, count = line.rstrip().split("\t")
                entity = Entity(title, None)
                self.vocab[entity] = index
                self.counter[entity] = int(count)
                self.inv_vocab[index] = [entity]

    def _parse_jsonl_vocab_file(self, vocab_file: str):
        with open(vocab_file, "r") as f:
            entities_json = [json.loads(line) for line in f]

        for item in entities_json:
            for title, language in item["entities"]:
                entity = Entity(title, language)
                self.vocab[entity] = item["id"]
                self.counter[entity] = item["count"]
                self.inv_vocab[item["id"]].append(entity)

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

    def contains(self, title: str, language: Optional[str] = None):
        return Entity(title, language) in self.vocab

    def get_id(self, title: str, language: Optional[str] = None, default: int = None) -> int:
        try:
            return self.vocab[Entity(title, language)]
        except KeyError:
            return default

    def get_title_by_id(self, id_: int, language: Optional[str] = None) -> str:
        for entity in self.inv_vocab[id_]:
            if entity.language == language:
                return entity.title

    def get_count_by_title(self, title: str, language: Optional[str] = None) -> int:
        entity = Entity(title, language)
        return self.counter.get(entity, 0)

    def search_across_languages(self, title: str) -> List[Entity]:
        results = []
        for entity in self.vocab.keys():
            if entity.title == title:
                results.append(entity)
        return results

    def save(self, out_file: str):

        if Path(out_file).suffix != ".jsonl":
            raise ValueError(
                "The saved file has to explicitly have the jsonl extension so that it will be loaded properly,\n"
                f"but the name provided is {out_file}."
            )

        with open(out_file, "w") as f:
            for ent_id, entities in self.inv_vocab.items():
                count = self.counter[entities[0]]
                item = {"id": ent_id, "entities": [(e.title, e.language) for e in entities], "count": count}
                json.dump(item, f)
                f.write("\n")

    @staticmethod
    def build(
        dump_db: DumpDB,
        out_file: str,
        vocab_size: int,
        min_count: int,
        white_list: List[str],
        white_list_only: bool,
        pool_size: int,
        chunk_size: int,
        language: Optional[str],
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

        Path(out_file).parent.mkdir(exist_ok=True, parents=True)
        with open(out_file, "w") as f:
            for ent_id, (title, count) in enumerate(title_dict.items()):
                if 0 < count < min_count:
                    continue
                json.dump({"id": ent_id, "entities": [[title, language]], "count": count}, f, ensure_ascii=False)
                f.write("\n")

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
@click.option("--entity-vocab-files", "-v", multiple=True)
@click.option("--inter-wiki-db-path", "-i", type=click.Path())
@click.option("--out-file", "-o", type=click.Path())
@click.option("--vocab-size", type=int)
@click.option("--min-num-languages", type=int)
def build_multilingual_entity_vocab(
    entity_vocab_files: List[str], inter_wiki_db_path: str, out_file: str, vocab_size: int, min_num_languages: int
):

    for entity_vocab_path in entity_vocab_files:
        if Path(entity_vocab_path).suffix != ".jsonl":
            raise RuntimeError(
                f"entity_vocab_path: {entity_vocab_path}\n"
                "Entity vocab files in this format is not supported."
                "Please use the jsonl file format and try again."
            )

    db = InterwikiDB.load(inter_wiki_db_path)

    vocab: Dict[Entity, int] = {}  # title -> index
    inv_vocab = defaultdict(set)  # ent_id -> Set[title]
    count_dict = defaultdict(int)  # ent_id -> count
    index_mapping = {}  # inter-language index -> ent_id

    special_token_to_idx = {special_token: idx for idx, special_token in enumerate(SPECIAL_TOKENS)}
    current_new_id = len(special_token_to_idx)

    for entity_vocab_path in entity_vocab_files:
        logger.info(f"Reading {entity_vocab_path}")
        with open(entity_vocab_path, "r") as f:
            for line in tqdm(f):
                entity_dict = json.loads(line)
                for title, lang in entity_dict["entities"]:
                    entity = Entity(title, lang)
                    if title not in SPECIAL_TOKENS:
                        try:
                            inter_language_id = db.get_id(title, lang)
                        except KeyError:
                            inter_language_id = None

                        # judge if we should assign a new id to these entities
                        if inter_language_id is not None and inter_language_id in index_mapping:
                            ent_id = index_mapping[inter_language_id]
                        else:
                            ent_id = current_new_id
                            current_new_id += 1
                            index_mapping[inter_language_id] = ent_id
                    else:
                        ent_id = special_token_to_idx[title]

                    vocab[entity] = ent_id
                    inv_vocab[ent_id].add((entity.title, entity.language))  # Convert Entity to Tuple for json.dump
                    count_dict[ent_id] += entity_dict["count"]
    json_dicts = [
        {"entities": list(inv_vocab[ent_id]), "count": count_dict[ent_id]} for ent_id in range(current_new_id)
    ]

    logger.info(f"Vocab size without truncation: {len(json_dicts)}")

    if min_num_languages is not None:
        json_dicts = [d for d in json_dicts if len(d["entities"]) >= min_num_languages]

    json_dicts.sort(key=lambda x: -x["count"] if x["count"] != 0 else -math.inf)
    if vocab_size is not None:
        json_dicts = json_dicts[:vocab_size]

    logger.info(f"Final vocab size: {len(json_dicts)}")
    logger.info(f"Saving to {out_file}")
    with open(out_file, "w") as f:
        for ent_id, item in enumerate(json_dicts):
            json.dump({"id": ent_id, **item}, f, ensure_ascii=False)
            f.write("\n")
