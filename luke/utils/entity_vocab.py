from collections import Counter, OrderedDict
from contextlib import closing
import multiprocessing
from multiprocessing.pool import Pool
import click
from tqdm import tqdm
from wikipedia2vec.dump_db import DumpDB

PAD_TOKEN = '[PAD]'
UNK_TOKEN = '[UNK]'
MASK_TOKEN = '[MASK]'
MASK2_TOKEN = '[MASK2]'

_dump_db = None  # global variable used in multiprocessing workers


class EntityVocab(object):
    def __init__(self, vocab_file):
        self._vocab_file = vocab_file

        self.vocab = self._load_vocab(vocab_file)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    @property
    def size(self):
        return len(self)

    def __reduce__(self):
        return (self.__class__, (self._vocab_file,))

    def __len__(self):
        return len(self.vocab)

    def __contains__(self, key):
        return key in self.vocab

    def __getitem__(self, key):
        return self.vocab[key]

    def __iter__(self):
        return iter(self.vocab)

    def get_id(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def get_title_by_id(self, id_):
        return self.inv_vocab[id_]

    def save(self, out_file):
        with open(self._vocab_file, 'r') as src:
            with open(out_file, 'w') as dst:
                dst.write(src.read())

    def _load_vocab(self, vocab_file):
        vocab = {}
        with open(vocab_file) as f:
            for (index, line) in enumerate(f):
                title = line.rstrip().split('\t')[0]
                vocab[title] = index

        return vocab

    @staticmethod
    def build(dump_db, out_file, vocab_size, white_list, white_list_only, pool_size, chunk_size):
        counter = Counter()
        with tqdm(total=dump_db.page_size(), mininterval=0.5) as bar:
            with closing(Pool(pool_size, initializer=EntityVocab._initialize_worker, initargs=(dump_db,))) as pool:
                for ret in pool.imap_unordered(EntityVocab._count_entities, dump_db.titles(), chunksize=chunk_size):
                    counter.update(ret)
                    bar.update()

        title_dict = OrderedDict()
        title_dict[PAD_TOKEN] = 0
        title_dict[UNK_TOKEN] = 0
        title_dict[MASK_TOKEN] = 0
        title_dict[MASK2_TOKEN] = 0

        for title in white_list:
            if counter[title] != 0:
                title_dict[title] = counter[title]

        if not white_list_only:
            valid_titles = frozenset(dump_db.titles())
            for (title, count) in counter.most_common():
                if title in valid_titles and not title.startswith('Category:'):
                    title_dict[title] = count
                    if len(title_dict) == vocab_size:
                        break

        with open(out_file, 'w') as f:
            for (title, count) in title_dict.items():
                f.write(f'{title}\t{count}\n')

    @staticmethod
    def _initialize_worker(dump_db):
        global _dump_db
        _dump_db = dump_db

    @staticmethod
    def _count_entities(title):
        counter = Counter()
        for paragraph in _dump_db.get_paragraphs(title):
            for wiki_link in paragraph.wiki_links:
                title = _dump_db.resolve_redirect(wiki_link.title)
                counter[title] += 1
        return counter


from luke.cli import cli

@cli.command()
@click.argument('dump_db_file', type=click.Path())
@click.argument('out_file', type=click.Path())
@click.option('--vocab-size', default=1000000)
@click.option('-w', '--white-list', type=click.File(), multiple=True)
@click.option('--white-list-only', is_flag=True)
@click.option('--pool-size', default=multiprocessing.cpu_count())
@click.option('--chunk-size', default=100)
def build_entity_vocab(dump_db_file, white_list, **kwargs):
    dump_db = DumpDB(dump_db_file)
    white_list = [line.rstrip() for f in white_list for line in f]
    EntityVocab.build(dump_db, white_list=white_list, **kwargs)
