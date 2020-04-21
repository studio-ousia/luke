import json
from collections import defaultdict

from .interwiki_db import InterwikiDB


class MultilingualEntityVocab(object):
    def __init__(self, vocab_file):
        self._vocab_file = vocab_file

        self.vocab = {}
        self.counter = {}
        self.inv_vocab = {}

        entities_json = json.load(open('test.json', 'r'))
        for ent_id, record in enumerate(entities_json):
            for title in record["entities"]:
                self.vocab[title] = ent_id
                self.counter[title] = record["count"]
            self.inv_vocab[ent_id] = record["entities"]

    @staticmethod
    def build(vocab_files, languages, inter_wiki_db_path, out_file):
        db = InterwikiDB.load(inter_wiki_db_path)

        vocab = {}  # title -> index
        inv_vocab = defaultdict(set)  # index -> List[title]
        count_dict = defaultdict(int)  # index -> count

        new_id = 0
        for vocab_path, lang in zip(vocab_files, languages):
            with open(vocab_path, 'r') as f:
                for line in f:
                    entity, count = line.strip().split('\t')
                    count = int(count)
                    multilingual_entities = db.query(entity, lang)
                    multilingual_entities = {e for e, _ in multilingual_entities}
                    multilingual_entities.add(entity)

                    use_new_id = True
                    for e in multilingual_entities:
                        if e in vocab:
                            ent_id = vocab[e]
                            use_new_id = False
                    if use_new_id:
                        ent_id = new_id
                        new_id += 1
                    vocab[entity] = ent_id
                    inv_vocab[ent_id].add(entity)
                    count_dict[ent_id] += count
        json_dicts = [{"entities": list(inv_vocab[ent_id]), "count": count_dict[ent_id]} for ent_id in range(new_id)]

        with open(out_file, 'w') as f:
            json.dump(json_dicts, f, indent=4)

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
        raise not NotImplementedError
        return self.inv_vocab[id_]

    def get_count_by_title(self, title):
        return self.counter.get(title, 0)

    def save(self, out_file):
        with open(self._vocab_file, 'r') as src:
            with open(out_file, 'w') as dst:
                dst.write(src.read())
