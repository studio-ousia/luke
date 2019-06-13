import collections
import tqdm

PAD_TOKEN = '[PAD]'
UNK_TOKEN = '[UNK]'
MASK_TOKEN = '[MASK]'
MASK2_TOKEN = '[MASK2]'


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

    def _load_vocab(self, vocab_file):
        vocab = {}
        with open(vocab_file) as f:
            for (index, line) in enumerate(f):
                title = line.rstrip().split('\t')[0]
                vocab[title] = index

        return vocab

    @staticmethod
    def build(wiki_corpus, out_file, vocab_size, white_list=[], white_list_only=False):
        counter = collections.Counter()
        with tqdm.tqdm(wiki_corpus.iterate_links(), total=wiki_corpus.link_size) as bar:
            for link in bar:
                counter[link.title] += 1

        title_dict = collections.OrderedDict()
        title_dict[PAD_TOKEN] = 0
        title_dict[UNK_TOKEN] = 0
        title_dict[MASK_TOKEN] = 0
        title_dict[MASK2_TOKEN] = 0

        for title in white_list:
            if counter[title] != 0:
                title_dict[title] = counter[title]

        if not white_list_only:
            for (title, count) in counter.most_common():
                title_dict[title] = count
                if len(title_dict) == vocab_size:
                    break

        with open(out_file, 'w') as f:
            for (title, count) in title_dict.items():
                f.write(f'{title}\t{count}\n')
