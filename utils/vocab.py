# -*- coding: utf-8 -*-

from collections import Counter
from marisa_trie import Trie
from tqdm import tqdm

MASK_TOKEN = '[MASK]'
UNK_TOKEN = '[UNK]'


class Vocab(object):
    def __init__(self, vocab_file):
        self.vocab = self._load_vocab(vocab_file)

    @property
    def size(self):
        return len(self)

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
        return self.inv_vocab[id_]

    def word_prefix_search(self, text):
        return self._word_trie.prefixes(text)

    def subword_prefix_search(self, text):
        return ['##' + w for w in self._subword_trie.prefixes(text)]

    def _load_vocab(self, vocab_input):
        if isinstance(vocab_input, dict):
            return vocab_input

        with open(vocab_input) as f:
            return {line.strip(): ind for (ind, line) in enumerate(f)}


class EntityVocab(Vocab):
    def __init__(self, vocab_file):
        super(EntityVocab, self).__init__(vocab_file)

    def get_title_by_id(self, id_):
        return self.vocab.restore_key(id_)

    def _load_vocab(self, vocab_input):
        vocab = Trie()
        vocab.load(vocab_input)
        return vocab

    @staticmethod
    def build_vocab(wiki_corpus, vocab_size, out_file, white_list=[]):
        counter = Counter()
        for link in wiki_corpus.iterate_links():
            counter[link.title] += 1

        titles = set()
        for title in white_list:
            if counter[title] != 0:
                titles.add(title)

        titles.add(MASK_TOKEN)
        titles.add(UNK_TOKEN)
        for (title, _) in counter.most_common():
            titles.add(title)
            if len(titles) == vocab_size:
                break

        vocab = Trie(titles)
        vocab.save(out_file)
