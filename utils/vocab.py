# -*- coding: utf-8 -*-

from collections import Counter
from marisa_trie import Trie
from tqdm import tqdm

UNK_TOKEN = '[UNK]'


class Vocab(object):
    def __init__(self, vocab_file, start_index=0):
        self._start_index = start_index

        self.vocab = self._load_vocab(vocab_file)

    @property
    def size(self):
        return len(self)

    def __len__(self):
        return len(self.vocab) + self._start_index

    def __contains__(self, key):
        return key in self.vocab

    def __getitem__(self, key):
        return self.vocab[key] + self._start_index

    def __iter__(self):
        return iter(self.vocab)

    def get_id(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def _load_vocab(self, vocab_input):
        raise NotImplementedError()


class WordPieceVocab(Vocab):
    def __init__(self, vocab_file):
        super(WordPieceVocab, self).__init__(vocab_file)

        self.inv_vocab = {v: k for k, v in self.vocab.items()}

        self._word_trie = Trie(w for w in self.vocab.keys() if not w.startswith('##'))
        self._subword_trie = Trie(w[2:] for w in self.vocab.keys() if w.startswith('##'))

    def __reduce__(self):
        return (self.__class__, (self.vocab,))

    def get_word_by_id(self, id_):
        return self.inv_vocab[id_ - self._start_index]

    def word_prefix_search(self, text):
        return self._word_trie.prefixes(text)

    def subword_prefix_search(self, text):
        return ['##' + w for w in self._subword_trie.prefixes(text)]

    def _load_vocab(self, vocab_input):
        if isinstance(vocab_input, dict):
            return vocab_input

        with open(vocab_input) as f:
            return {line.strip(): ind for (ind, line) in enumerate(f)}

BertVocab = WordPieceVocab


class EntityVocab(Vocab):
    def __init__(self, vocab_file, start_index=1):
        super(EntityVocab, self).__init__(vocab_file, start_index)

    def get_title_by_id(self, id_):
        return self.vocab.restore_key(id_ - self._start_index)

    def _load_vocab(self, vocab_input):
        vocab = Trie()
        vocab.load(vocab_input)
        return vocab

    @staticmethod
    def build_vocab(dump_db, vocab_size, out_file, white_list=[]):
        counter = Counter()

        for page_title in tqdm(dump_db.titles(), total=dump_db.page_size()):
            for paragraph in dump_db.get_paragraphs(page_title):
                for link in paragraph.wiki_links:
                    title = link.title
                    if title.lower().split(':')[0] in ('image', 'file'):
                        continue
                    title = dump_db.resolve_redirect(title)
                    counter[title] += 1

        titles = set([t for (t, _) in counter.most_common(vocab_size - 1)])
        titles.update(white_list)
        titles.add(UNK_TOKEN)
        vocab = Trie(titles)
        vocab.save(out_file)
