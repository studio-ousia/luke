from typing import List, Union
import logging
from collections import defaultdict, Counter
from contextlib import closing
import multiprocessing
from multiprocessing.pool import Pool
import click
import joblib
import marisa_trie
from tqdm import tqdm
from transformers.models.bert import BasicTokenizer
from wikipedia2vec.dump_db import DumpDB

SEP_CHAR = "\u2581"
REP_CHAR = "_"

logger = logging.getLogger(__name__)


@click.group(name="mention-db")
def cli():
    pass


@cli.command()
@click.argument("dump_db_file", type=click.Path(exists=True))
@click.argument("out_file", type=click.Path())
@click.option("--min-link-prob", default=0.01)
@click.option("--max-candidate-size", default=100)
@click.option("--min-link-count", default=1)
@click.option("--max-mention-length", default=20)
@click.option("--pool-size", default=multiprocessing.cpu_count())
@click.option("--chunk-size", default=100)
def build_from_wikipedia(dump_db_file, **kwargs):
    dump_db = DumpDB(dump_db_file)
    tokenizer = BasicTokenizer(do_lower_case=False)
    normalizer = BertLowercaseNormalizer()
    MentionDB.build_from_wikipedia(dump_db, tokenizer, normalizer, **kwargs)


@cli.command()
@click.argument("p_e_m_file", type=click.Path(exists=True))
@click.argument("dump_db_file", type=click.Path(exists=True))
@click.argument("wiki_mention_db_file", type=click.Path(exists=True))
@click.argument("out_file", type=click.Path())
@click.option("--max-mention-length", default=20)
def build_from_p_e_m_file(p_e_m_file, dump_db_file, wiki_mention_db_file, **kwargs):
    dump_db = DumpDB(dump_db_file)
    tokenizer = BasicTokenizer(do_lower_case=False)
    normalizer = BertLowercaseNormalizer()
    wiki_mention_db = MentionDB(wiki_mention_db_file)
    MentionDB.build_from_p_e_m_file(p_e_m_file, dump_db, wiki_mention_db, tokenizer, normalizer, **kwargs)


class Mention(object):
    __slots__ = ("title", "text", "start", "end", "link_count", "total_link_count", "doc_count")

    def __init__(
        self, title: str, text: str, start: int, end: int, link_count: int, total_link_count: int, doc_count: int
    ):
        self.title = title
        self.text = text
        self.start = start
        self.end = end
        self.link_count = link_count
        self.total_link_count = total_link_count
        self.doc_count = doc_count

    @property
    def span(self):
        return (self.start, self.end)

    @property
    def link_prob(self):
        if self.doc_count > 0:
            return min(1.0, self.total_link_count / self.doc_count)
        else:
            return 0.0

    @property
    def prior_prob(self):
        if self.total_link_count > 0:
            return min(1.0, self.link_count / self.total_link_count)
        else:
            return 0.0

    def __repr__(self):
        return f"<Mention {self.text} -> {self.title}>"


class BertLowercaseNormalizer(object):
    def __init__(self, never_lowercase=("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")):
        self._tokenizer = BasicTokenizer()
        self._never_lowercase = frozenset(never_lowercase)

    def normalize(self, token):
        if token not in self._never_lowercase:
            token = token.lower()
            token = self._tokenizer._run_strip_accents(token)
        return token


# global variables used in pool workers
_dump_db = _tokenizer = _normalizer = _max_mention_length = _name_trie = None


class MentionDB(object):
    def __init__(self, mention_db_file: str):
        self.mention_db_file = mention_db_file

        data = joblib.load(mention_db_file)
        self._title_trie = data["title_trie"]
        self._mention_trie = data["mention_trie"]
        self._data_trie = data["data_trie"]
        self._tokenizer = data["tokenizer"]
        self._normalizer = data["normalizer"]
        self._max_mention_length = data["max_mention_length"]

    def __reduce__(self):
        return (self.__class__, (self.mention_db_file,))

    def query(self, text_or_tokens: Union[str, List[str]]):
        if isinstance(text_or_tokens, str):
            tokens = self._tokenizer.tokenize(text_or_tokens)
        else:
            tokens = text_or_tokens
        tokens = [self._normalizer.normalize(t.replace(SEP_CHAR, REP_CHAR)) for t in tokens]
        name = SEP_CHAR.join(tokens)
        try:
            return [
                Mention(self._title_trie.restore_key(args[0]), name.replace(SEP_CHAR, " "), None, None, *args[1:])
                for args in self._data_trie[name]
            ]
        except KeyError:
            return []

    def save(self, out_file: str):
        joblib.dump(
            dict(
                title_trie=self._title_trie,
                mention_trie=self._mention_trie,
                data_trie=self._data_trie,
                tokenizer=self._tokenizer,
                normalizer=self._normalizer,
                max_mention_length=self._max_mention_length,
            ),
            out_file,
        )

    @staticmethod
    def build_from_wikipedia(
        dump_db,
        tokenizer,
        normalizer,
        out_file,
        min_link_prob,
        max_candidate_size,
        min_link_count,
        max_mention_length,
        pool_size,
        chunk_size,
    ):
        logger.info("Iteration 1/2: Extracting all entity names...")

        name_dict = defaultdict(Counter)
        with tqdm(total=dump_db.page_size(), mininterval=0.5) as pbar:
            initargs = (dump_db, tokenizer, normalizer, max_mention_length)
            with closing(Pool(pool_size, initializer=MentionDB._initialize_worker, initargs=initargs)) as pool:
                for ret in pool.imap_unordered(
                    MentionDB._extract_name_entity_pairs, dump_db.titles(), chunksize=chunk_size
                ):
                    for (name, title) in ret:
                        name_dict[name][title] += 1
                    pbar.update()

        logger.info("Iteration 2/2: Counting occurrences of entity names...")

        with tqdm(total=dump_db.page_size(), mininterval=0.5) as pbar:
            name_doc_counter = Counter()
            initargs = (dump_db, tokenizer, normalizer, max_mention_length, marisa_trie.Trie(name_dict.keys()))
            with closing(Pool(pool_size, initializer=MentionDB._initialize_worker, initargs=initargs)) as pool:
                for names in pool.imap_unordered(
                    MentionDB._extract_name_occurrences, dump_db.titles(), chunksize=chunk_size
                ):
                    name_doc_counter.update(names)
                    pbar.update()

        logger.info("Building DB...")

        titles = frozenset([title for entity_counter in name_dict.values() for title in entity_counter.keys()])
        title_trie = marisa_trie.Trie(titles)

        def item_generator():
            for (name, entity_counter) in name_dict.items():
                doc_count = name_doc_counter[name]
                total_link_count = sum(entity_counter.values())

                if doc_count == 0:
                    continue

                link_prob = total_link_count / doc_count
                if link_prob < min_link_prob:
                    continue

                for (title, link_count) in entity_counter.most_common()[:max_candidate_size]:
                    if link_count < min_link_count:
                        continue
                    yield (name, (title_trie[title], link_count, total_link_count, doc_count))

        data_trie = marisa_trie.RecordTrie("<IIII", item_generator())
        mention_trie = marisa_trie.Trie(data_trie.keys())

        joblib.dump(
            dict(
                title_trie=title_trie,
                mention_trie=mention_trie,
                data_trie=data_trie,
                tokenizer=tokenizer,
                normalizer=normalizer,
                max_mention_length=max_mention_length,
            ),
            out_file,
        )

    @staticmethod
    def build_from_p_e_m_file(
        p_e_m_file, dump_db, wiki_mention_db, tokenizer, normalizer, out_file, max_mention_length
    ):
        with open(p_e_m_file) as f:
            lines = f.readlines()

        name_dict = defaultdict(Counter)

        for line in tqdm(lines):
            (text, total_count, *data) = line.rstrip().split("\t")
            total_count = int(total_count)
            text = text.replace(SEP_CHAR, REP_CHAR)
            tokens = tuple(normalizer.normalize(t) for t in tokenizer.tokenize(text))
            if len(tokens) <= max_mention_length:
                for entry in data:
                    (_, prob, *title_parts) = entry.split(",")
                    title = ",".join(title_parts).replace("_", " ")
                    title = dump_db.resolve_redirect(title)
                    count = int(float(prob) * total_count)
                    name_dict[tokens][title] += count

        titles = frozenset([title for entity_counter in name_dict.values() for title in entity_counter.keys()])
        title_trie = marisa_trie.Trie(titles)

        def item_generator():
            for (tokens, entity_counter) in name_dict.items():
                name = SEP_CHAR.join(tokens)
                total_link_count = sum(entity_counter.values())

                wiki_mentions = wiki_mention_db.query(tokens)
                if wiki_mentions:
                    doc_count = int(total_link_count / wiki_mentions[0].link_prob)
                else:
                    doc_count = 0

                for (title, link_count) in entity_counter.most_common():
                    yield (name, (title_trie[title], link_count, total_link_count, doc_count))

        data_trie = marisa_trie.RecordTrie("<IIII", item_generator())
        mention_trie = marisa_trie.Trie(data_trie.keys())

        joblib.dump(
            dict(
                title_trie=title_trie,
                mention_trie=mention_trie,
                data_trie=data_trie,
                tokenizer=tokenizer,
                normalizer=normalizer,
                max_mention_length=max_mention_length,
            ),
            out_file,
        )

    @staticmethod
    def _initialize_worker(dump_db, tokenizer, normalizer, max_mention_length, name_trie=None):
        global _dump_db, _tokenizer, _normalizer, _max_mention_length, _name_trie
        _dump_db = dump_db
        _tokenizer = tokenizer
        _normalizer = normalizer
        _max_mention_length = max_mention_length
        _name_trie = name_trie

    @staticmethod
    def _extract_name_entity_pairs(title):
        ret = []
        for paragraph in _dump_db.get_paragraphs(title):
            for wiki_link in paragraph.wiki_links:
                title = _dump_db.resolve_redirect(wiki_link.title)
                text = wiki_link.text.replace(SEP_CHAR, REP_CHAR)
                tokens = [_normalizer.normalize(t) for t in _tokenizer.tokenize(text)]
                if len(tokens) <= _max_mention_length:
                    ret.append((SEP_CHAR.join(tokens), title))
        return ret

    @staticmethod
    def _extract_name_occurrences(title):
        ret = []
        for paragraph in _dump_db.get_paragraphs(title):
            text = paragraph.text.replace(SEP_CHAR, REP_CHAR)
            tokens = [_normalizer.normalize(t) for t in _tokenizer.tokenize(text)]
            for n in range(len(tokens)):
                target_text = SEP_CHAR.join(tokens[n : n + _max_mention_length])
                for name in _name_trie.prefixes(target_text):
                    if len(target_text) == len(name) or target_text[len(name)] == SEP_CHAR:
                        ret.append(name)
        return frozenset(ret)
