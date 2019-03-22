# -*- coding: utf-8 -*-

import logging
import os
import re

CORPUS_FILE = 'AIDA-YAGO2-dataset.tsv'
CANDIDATES_DIR = 'PPRforNED/AIDA_candidates/'

logger = logging.getLogger(__name__)


class Document(object):
    __slots__ = ('id', 'words', 'mentions', 'fold')

    def __init__(self, id_, words, mentions, fold=None):
        self.id = id_
        self.words = words
        self.mentions = mentions
        self.fold = fold

    def __repr__(self):
        return '<Document %s...>' % (' '.join(self.words[:3]),)


class Mention(object):
    __slots__ = ('text', 'title', 'span', 'candidates')

    def __init__(self, text, title, span, candidates):
        self.text = text
        self.span = span
        self.title = title
        self.candidates = candidates

    def __repr__(self):
        return '<Mention %s->%s>' % (self.text, self.title)


class AIDACoNLLReader(object):
    def __init__(self, dataset_dir):
        self._dataset_dir = dataset_dir

    def get_documents(self, target_fold=None):
        with open(os.path.join(self._dataset_dir, CORPUS_FILE)) as f:
            corpus_text = f.read()

        documents = []
        for document in re.split(r'-DOCSTART-\s', corpus_text)[1:]:
            (data, document) = document.split('\n', 1)

            if 'testa' in data:
                fold = 'dev'
            elif 'testb' in data:
                fold = 'test'
            else:
                fold = 'train'

            if target_fold and fold != target_fold:
                continue

            document_id = int(re.match(r'\((\d+)', data).group(1))

            words = []
            mention_texts = []
            mention_spans = []
            begin = None

            lines = document.split('\n')
            for (n, line) in enumerate(lines):
                items = line.split('\t')
                words.append(items[0])

                if begin is not None:
                    if len(items) == 1 or (len(items) >= 4 and items[1] == 'B'):
                        mention_spans.append((begin, n))
                        begin = None

                if len(items) >= 4:
                    (label, text, _) = items[1:4]
                    if label == 'B':
                        begin = n
                        mention_texts.append(text)

                if n == len(lines) - 1 and begin is not None:
                    mention_spans.append((begin, n))

            mentions = self._create_mentions(document_id, mention_texts, mention_spans)
            documents.append(Document(document_id, words, mentions, fold))

        logger.info('created %d mentions in %d documents',
                    len([m for d in documents for m in d.mentions]), len(documents))

        return documents

    def _create_mentions(self, document_id, texts, spans):
        candidates_dir = os.path.join(self._dataset_dir, CANDIDATES_DIR)

        if document_id <= 1000:
            dir_name = 'PART_1_1000'
        else:
            dir_name = 'PART_1001_1393'

        target_file = os.path.join(os.path.join(candidates_dir, dir_name), str(document_id))
        mentions = []
        cur = 0
        with open(target_file) as f:
            for line in f:
                if line.startswith('ENTITY'):
                    split_line = line.rstrip().split('\t')

                    text = split_line[7][9:]
                    # skip mentions with no candidates
                    while text != texts[cur]:
                        cur += 1

                    wiki_url = split_line[8][4:]
                    if wiki_url == 'NIL':
                        title = 'NIL'
                    else:
                        title = wiki_url[29:].replace('_', ' ')
                    mentions.append((text, title, spans[cur], []))
                    cur += 1

                elif line.startswith('CANDIDATE'):
                    wiki_url = line.split('\t')[5][4:]
                    title = wiki_url[29:].replace('_', ' ')
                    mentions[-1][3].append(title)

        return [Mention(*args) for args in mentions if args[1] != 'NIL']
