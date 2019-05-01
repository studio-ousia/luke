# -*- coding: utf-8 -*-

from collections import Counter, OrderedDict
from marisa_trie import Trie, RecordTrie

PAD_TOKEN = '[PAD]'
UNK_TOKEN = '[UNK]'
MASK_TOKEN = '[MASK]'


class Vocab(object):
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
        self.vocab = self._load_vocab(vocab_file)
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
    def __init__(self, vocab_file, vocab_format='bin'):
        self.vocab = self._load_vocab(vocab_file, vocab_format)
        self.inv_vocab = {v[0]: k for k, v in self.vocab.items()}

    def __reduce__(self):
        return (self.__class__, (self.vocab,))

    def __getitem__(self, key):
        return self.vocab[key][0][0]

    def get_title_by_id(self, id_):
        return self.inv_vocab[id_]

    def get_occurrence_count(self, key):
        return self.vocab[key][0][1]

    def _load_vocab(self, vocab_file, vocab_format):
        if vocab_format == 'bin':
            vocab = RecordTrie('II')
            vocab.load(vocab_file)
        else:
            def item_generator():
                with open(vocab_file) as f:
                    for line in f:
                        (title, index, count) = line.rstrip().split('\t')
                        yield (title, (int(index), int(count)))
            vocab = RecordTrie('II', item_generator())

        return vocab

    def save(self, out_file):
        self.vocab.save(out_file)

    def save_tsv(self, out_file):
        with open(out_file, 'w') as f:
            for (title, (index, count)) in self.vocab.items():
                f.write(f'{title}\t{index}\t{count}\n')

    @staticmethod
    def build_vocab(wiki_corpus, out_file, vocab_size, white_list=[], white_list_only=False):
        counter = Counter()
        for link in wiki_corpus.iterate_links():
            counter[link.title] += 1

        title_dict = OrderedDict()
        title_dict[PAD_TOKEN] = 0
        title_dict[UNK_TOKEN] = 0
        title_dict[MASK_TOKEN] = 0

        for title in white_list:
            if counter[title] != 0:
                title_dict[title] = counter[title]

        if not white_list_only:
            for (title, count) in counter.most_common():
                title_dict[title] = count
                if len(title_dict) == vocab_size:
                    break

        def item_generator():
            for (ind, (title, count)) in enumerate(title_dict.items()):
                yield (title, (ind, count))

        vocab = RecordTrie('II', item_generator())
        vocab.save(out_file)
