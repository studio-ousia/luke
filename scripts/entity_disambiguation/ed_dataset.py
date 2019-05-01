# -*- coding: utf-8 -*-
# This code is based on the code obtained from here: https://github.com/lephong/mulrel-nel/blob/db14942450f72c87a4d46349860e96ef2edf353d/nel/dataset.py

import os
import re
from collections import defaultdict


class EntityDisambiguationDataset:
    def __init__(self, dataset_dir):
        person_names = frozenset(load_person_names(os.path.join(dataset_dir, 'persons.txt')))

        self.train = load_documents(os.path.join(dataset_dir, 'aida_train.csv'),
            os.path.join(dataset_dir, 'aida_train.txt'), person_names)
        self.test_a = load_documents(os.path.join(dataset_dir, 'aida_testA.csv'),
            os.path.join(dataset_dir, 'testa_testb_aggregate_original'), person_names)
        self.test_b = load_documents(os.path.join(dataset_dir, 'aida_testB.csv'),
            os.path.join(dataset_dir, 'testa_testb_aggregate_original'), person_names)
        self.ace2004 = load_documents(os.path.join(dataset_dir, 'wned-ace2004.csv'),
            os.path.join(dataset_dir, 'ace2004.conll'), person_names)
        self.aquaint = load_documents(os.path.join(dataset_dir, 'wned-aquaint.csv'),
            os.path.join(dataset_dir, 'aquaint.conll'), person_names)
        self.clueweb = load_documents(os.path.join(dataset_dir, 'wned-clueweb.csv'),
            os.path.join(dataset_dir, 'clueweb.conll'), person_names)
        self.msnbc = load_documents(os.path.join(dataset_dir, 'wned-msnbc.csv'),
            os.path.join(dataset_dir, 'msnbc.conll'), person_names)
        self.wikipedia = load_documents(os.path.join(dataset_dir, 'wned-wikipedia.csv'),
            os.path.join(dataset_dir, 'wikipedia.conll'), person_names)

    def get_all_datasets(self):
        return (self.train, self.test_a, self.test_b, self.ace2004, self.aquaint, self.clueweb,
                self.msnbc, self.wikipedia)


class Document(object):
    __slots__ = ('id', 'words', 'mentions')

    def __init__(self, id_, words, mentions):
        self.id = id_
        self.words = words
        self.mentions = mentions

    def __repr__(self):
        return '<Document %s...>' % (' '.join(self.words[:3]),)


class Mention(object):
    __slots__ = ('text', 'title', 'start', 'end', 'candidates')

    def __init__(self, text, title, start, end, candidates):
        self.text = text
        self.start = start
        self.end = end
        self.title = title
        self.candidates = candidates

    @property
    def span(self):
        return (self.start, self.end)

    def __repr__(self):
        return '<Mention %s->%s>' % (self.text, self.title)


class Candidate(object):
    __slots__ = ('title', 'prior_prob')

    def __init__(self, title, prior_prob):
        self.title = title
        self.prior_prob = prior_prob

    def __repr__(self):
        return '<Candidate %s (prior prob: %.3f)>' % (self.title, self.prior_prob)


def load_person_names(input_file):
    with open(input_file) as f:
        return [l.strip() for l in f]


def load_documents(csv_path, conll_path, person_names):
    document_data = {}
    mention_data = load_mentions_from_csv_file(csv_path, person_names)

    with open(conll_path, 'r') as f:
        cur_doc = {}

        for line in f:
            line = line.strip()
            if line.startswith('-DOCSTART-'):
                doc_name = line.split()[1][1:]
                document_data[doc_name] = dict(words=[], mentions=[], mention_spans=[])
                cur_doc = document_data[doc_name]

            else:
                comps = line.split('\t')
                if len(comps) >= 6:
                    tag = comps[1]
                    if tag == 'I':
                        cur_doc['mention_spans'][-1]['end'] += 1
                    else:
                        cur_doc['mention_spans'].append(dict(start=len(cur_doc['words']),
                                                             end=len(cur_doc['words']) + 1))

                cur_doc['words'].append(comps[0])

    documents = []

    # merge with the mention_data
    punc_remover = re.compile(r'[\W]+')
    for (doc_name, mentions) in mention_data.items():
        # This document is excluded in Le and Titov 2018: https://github.com/lephong/mulrel-nel/blob/db14942450f72c87a4d46349860e96ef2edf353d/nel/dataset.py#L221
        if doc_name == 'Jiří_Třanovský Jiří_Třanovský':
            continue
        document = document_data[doc_name.split()[0]]

        mention_span_index = 0
        for mention in mentions:
            mention_text = punc_remover.sub('', mention['text'].lower())

            while True:
                doc_mention_span = document['mention_spans'][mention_span_index]
                doc_mention_text = ' '.join(
                    document['words'][doc_mention_span['start']:doc_mention_span['end']])
                doc_mention_text = punc_remover.sub('', doc_mention_text.lower())
                if doc_mention_text == mention_text:
                    mention.update(doc_mention_span)
                    document['mentions'].append(mention)
                    mention_span_index += 1
                    break
                else:
                    mention_span_index += 1

        mentions = [Mention(**o) for o in document['mentions']]
        documents.append(Document(doc_name, document['words'], mentions))

    return documents


def load_mentions_from_csv_file(path, person_names):
    mention_data = defaultdict(list)
    with open(path, 'r') as f:
        for line in f:
            comps = line.strip().split('\t')
            doc_name = comps[0] + ' ' + comps[1]
            mention_text = comps[2]

            if comps[6] != 'EMPTYCAND':
                candidates = [c.split(',') for c in comps[6:-2]]
                candidates = [Candidate(','.join(c[2:]), float(c[1])) for c in candidates]
                candidates = [c for c in candidates if c.title]
                candidates = sorted(candidates, key=lambda c: c.prior_prob, reverse=True)
            else:
                candidates = []

            title = comps[-1].split(',')
            if title[0] == '-1':
                title = ','.join(title[2:])
            else:
                title = ','.join(title[3:])

            title = title.replace('&amp;', '&')

            if not title:  # we use only mentions with valid referent entities
                continue

            mention_data[doc_name].append(dict(text=mention_text, candidates=candidates,
                                               title=title))

    def find_coreference(target_mention, mention_list):
        target_mention_text = target_mention['text'].lower()
        ret = []

        for mention in mention_list:
            if not mention['candidates'] or mention['candidates'][0].title not in person_names:
                continue

            mention_text = mention['text'].lower()
            if mention_text == target_mention_text:
                continue

            start_pos = mention_text.find(target_mention_text)
            if start_pos == -1:
                continue

            end_pos = start_pos + len(target_mention_text) - 1
            if (start_pos == 0 or mention_text[start_pos - 1] == ' ') and\
               (end_pos == len(mention_text) - 1 or mention_text[end_pos + 1] == ' '):
                ret.append(mention)

        return ret

    for (_, mentions) in mention_data.items():
        for mention in mentions:
            coref_mentions = find_coreference(mention, mentions)
            if coref_mentions:
                new_cands = defaultdict(int)
                for coref_mention in coref_mentions:
                    for candidate in coref_mention['candidates']:
                        new_cands[candidate.title] += candidate.prior_prob

                for candidate_title in new_cands.keys():
                    new_cands[candidate_title] /= len(coref_mentions)

                mention['candidates'] = sorted([Candidate(t, p) for (t, p) in new_cands.items()],
                                               key=lambda c: c.prior_prob, reverse=True)

    return mention_data
