# This code is based on the code obtained from here:
# https://github.com/lephong/mulrel-nel/blob/db14942450f72c87a4d46349860e96ef2edf353d/nel/dataset.py

import logging
import math
import os
import re
from collections import defaultdict
import numpy as np

from ..utils.mention_db import MentionDB
from ..utils.mention_db import BertLowercaseNormalizer

logger = logging.getLogger(__name__)


class EntityLinkingDataset(object):
    def __init__(self, dataset_dir, mention_db_path, wikipedia_titles_file=None, wikipedia_redirects_file=None):
        person_names = frozenset(load_person_names(os.path.join(dataset_dir, "persons.txt")))
        self.mention_DB = MentionDB(mention_db_path)
        self.train = load_documents(
            os.path.join(dataset_dir, "aida_train.csv"),
            os.path.join(dataset_dir, "aida_train.txt"),
            person_names,
            self.mention_DB,
        )
        self.test_a = load_documents(
            os.path.join(dataset_dir, "aida_testA.csv"),
            os.path.join(dataset_dir, "testa_testb_aggregate_original"),
            person_names,
            self.mention_DB,
        )
        self.test_b = load_documents(
            os.path.join(dataset_dir, "aida_testB.csv"),
            os.path.join(dataset_dir, "testa_testb_aggregate_original"),
            person_names,
            self.mention_DB,
        )
        self.ace2004 = load_documents(
            os.path.join(dataset_dir, "wned-ace2004.csv"),
            os.path.join(dataset_dir, "ace2004.conll"),
            person_names,
            self.mention_DB,
        )
        self.aquaint = load_documents(
            os.path.join(dataset_dir, "wned-aquaint.csv"),
            os.path.join(dataset_dir, "aquaint.conll"),
            person_names,
            self.mention_DB,
        )
        self.clueweb = load_documents(
            os.path.join(dataset_dir, "wned-clueweb.csv"),
            os.path.join(dataset_dir, "clueweb.conll"),
            person_names,
            self.mention_DB,
        )
        self.msnbc = load_documents(
            os.path.join(dataset_dir, "wned-msnbc.csv"),
            os.path.join(dataset_dir, "msnbc.conll"),
            person_names,
            self.mention_DB,
        )
        self.wikipedia = load_documents(
            os.path.join(dataset_dir, "wned-wikipedia.csv"),
            os.path.join(dataset_dir, "wikipedia.conll"),
            person_names,
            self.mention_DB,
        )

        valid_titles = None
        if wikipedia_titles_file:
            with open(wikipedia_titles_file) as f:
                valid_titles = frozenset([l.rstrip() for l in f])

        redirects = {}
        if wikipedia_redirects_file:
            with open(wikipedia_redirects_file) as f:
                for line in f:
                    (src, dest) = line.rstrip().split("\t")
                    redirects[src] = dest

        logger.info("Building entity vocab and resolving wikipedia vocabularies")
        # build entity vocabulary and resolve Wikipedia redirects
        for documents in self.get_all_datasets():
            for document in documents:
                new_mentions = []
                for mention in document.mentions:
                    mention.title = redirects.get(mention.title, mention.title)
                    if valid_titles and mention.title not in valid_titles:
                        logger.debug("Invalid title: %s", mention.title)
                        continue
                    new_mentions.append(mention)
                    for candidate in mention.candidates:
                        candidate.title = redirects.get(candidate.title, candidate.title)
                document.mentions = new_mentions

        logger.info("Done")

    def get_all_datasets(self):
        return (
            self.train,
            self.test_a,
            self.test_b,
            self.ace2004,
            self.aquaint,
            self.clueweb,
            self.msnbc,
            self.wikipedia,
        )


class Document(object):
    def __init__(self, id_, words, mentions):
        self.id = id_
        self.words = words
        self.mentions = mentions

    def __repr__(self):
        return "<Document %s...>" % (" ".join(self.words[:3]),)


class Mention(object):
    def __init__(self, text, title, start, end, candidates):
        self.text = text
        self.start = start
        self.end = end
        self.title = title
        self.candidates = candidates

    @property
    def span(self):
        return self.start, self.end

    def __repr__(self):
        return "<Mention %s->%s>" % (self.text, self.title)


class Candidate(object):
    def __init__(self, title, prior_prob):
        self.title = title
        self.prior_prob = prior_prob

    def __repr__(self):
        return "<Candidate %s (prior prob: %.3f)>" % (self.title, self.prior_prob)


class InputFeatures(object):
    def __init__(
        self,
        document,
        mentions,
        word_ids,
        word_segment_ids,
        word_attention_mask,
        entity_ids,
        entity_position_ids,
        entity_segment_ids,
        entity_attention_mask,
        entity_candidate_ids,
        target_mention_indices,
    ):
        self.document = document
        self.mentions = mentions
        self.word_ids = word_ids
        self.word_segment_ids = word_segment_ids
        self.word_attention_mask = word_attention_mask
        self.entity_ids = entity_ids
        self.entity_position_ids = entity_position_ids  #
        self.entity_segment_ids = entity_segment_ids  #
        self.entity_attention_mask = entity_attention_mask  #
        self.entity_candidate_ids = entity_candidate_ids  #
        self.target_mention_indices = target_mention_indices  #


def load_person_names(input_file):
    with open(input_file) as f:
        return [l.strip() for l in f]


def load_documents(csv_path, conll_path, person_names, mention_DB):
    document_data = {}
    mention_data = load_mentions_from_csv_file(csv_path, person_names)

    with open(conll_path, "r") as f:
        cur_doc = {}

        for line in f:
            if line == "\n":
                cur_doc["words"].append("\n")
                continue

            line = line.strip()
            if line.startswith("-DOCSTART-"):
                doc_name = line.split()[1][1:]
                document_data[doc_name] = dict(words=[], mentions=[], mention_spans=[])
                cur_doc = document_data[doc_name]

            else:
                comps = line.split("\t")
                if len(comps) >= 6:
                    tag = comps[1]
                    if tag == "I":
                        cur_doc["mention_spans"][-1]["end"] += 1
                    else:
                        cur_doc["mention_spans"].append(
                            dict(start=len(cur_doc["words"]), end=len(cur_doc["words"]) + 1)
                        )
                cur_doc["words"].append(comps[0])

    for doc_id in document_data.keys():
        current_document = document_data[doc_id]
        for i in range(len(current_document["words"])):
            if current_document["words"][i] == "\n":  # 改行から始まるメンションはパス #
                continue
            for j in range(i + 1, len(current_document["words"])):
                if current_document["words"][j - 1] == "\n":  # 改行をまたぐメンションはパス
                    break
                if {"start": i, "end": j} in current_document["mention_spans"]:
                    continue
                if mention_DB.query(" ".join(current_document["words"][i:j])):
                    current_document["mention_spans"].append(dict(start=i, end=j))
                else:
                    break

    def person_name_coreference(target_mention, mention_list):
        target_mention_text = target_mention["text"].lower()
        ret = []

        for mention in mention_list:
            if not mention["candidates"] or mention["candidates"][0].title not in person_names:
                continue

            mention_text = mention["text"].lower()
            if mention_text == target_mention_text:
                continue

            start_pos = mention_text.find(target_mention_text)
            if start_pos == -1:
                continue

            end_pos = start_pos + len(target_mention_text) - 1
            if (start_pos == 0 or mention_text[start_pos - 1] == " ") and (
                end_pos == len(mention_text) - 1 or mention_text[end_pos + 1] == " "
            ):
                ret.append(mention)

        return ret

    documents = []

    # merge with the mention_data
    punc_remover = re.compile(r"[\W]+")
    for (doc_name, mentions) in mention_data.items():
        # This document is excluded in Le and Titov 2018:
        # https://github.com/lephong/mulrel-nel/blob/db14942450f72c87a4d46349860e96ef2edf353d/nel/dataset.py#L221
        if doc_name == "Jiří_Třanovský Jiří_Třanovský":
            continue
        document = document_data[doc_name.split()[0]]

        mention_span_index = 0
        for mention in mentions:
            mention_text = punc_remover.sub("", mention["text"].lower())  # spaceを潰す #

            while True:
                doc_mention_span = document["mention_spans"][mention_span_index]
                doc_mention_text = " ".join(document["words"][doc_mention_span["start"] : doc_mention_span["end"]])
                doc_mention_text = punc_remover.sub("", doc_mention_text.lower())  # space 潰す #
                if doc_mention_text == mention_text:

                    doc_mention_cand = sorted(
                        [Candidate(m.title, m.prior_prob) for m in mention_DB.query(doc_mention_text)],
                        key=lambda c: c.prior_prob,
                        reverse=True,
                    )
                    document["mentions"].append(
                        {
                            "text": mention["text"],
                            "candidates": doc_mention_cand,
                            "title": mention["title"],
                            "start": doc_mention_span["start"],
                            "end": doc_mention_span["end"],
                        }
                    )

                    mention_span_index += 1
                    break
                else:
                    mention_span_index += 1

        while mention_span_index < len(document["mention_spans"]):
            doc_mention_span = document["mention_spans"][mention_span_index]
            doc_mention_text = " ".join(document["words"][doc_mention_span["start"] : doc_mention_span["end"]])
            doc_mention_cand = sorted(
                [Candidate(m.title, m.prior_prob) for m in mention_DB.query(doc_mention_text)],
                key=lambda c: c.prior_prob,
                reverse=True,
            )
            document["mentions"].append(
                {
                    "text": doc_mention_text,
                    "candidates": doc_mention_cand,
                    "title": "[NO_E]",
                    "start": doc_mention_span["start"],
                    "end": doc_mention_span["end"],
                }
            )
            mention_span_index += 1

        for m in document["mentions"]:
            coref_mentions = person_name_coreference(m, document["mentions"])
            if coref_mentions:
                new_cands = defaultdict(int)
                for coref_m in coref_mentions:
                    for cand in coref_m["candidates"]:
                        new_cands[cand.title] += cand.prior_prob
                for cand_title in new_cands.keys():
                    new_cands[cand_title] /= len(coref_mentions)
                m["candidates"] = sorted(
                    [Candidate(t, p) for (t, p) in new_cands.items()], key=lambda c: c.prior_prob, reverse=True
                )

        mentions = [Mention(**o) for o in document["mentions"]]
        documents.append(Document(doc_name, document["words"], mentions))

    return documents


def load_mentions_from_csv_file(path, person_names):
    mention_data = defaultdict(list)
    with open(path, "r") as f:
        for line in f:
            comps = line.strip().split("\t")
            doc_name = comps[0] + " " + comps[1]
            mention_text = comps[2]

            if comps[6] != "EMPTYCAND":
                candidates = [c.split(",") for c in comps[6:-2]]
                candidates = [Candidate(",".join(c[2:]), float(c[1])) for c in candidates]
                candidates = [c for c in candidates if c.title]
                candidates = sorted(candidates, key=lambda c: c.prior_prob, reverse=True)
            else:
                candidates = []

            title = comps[-1].split(",")
            if title[0] == "-1":
                title = ",".join(title[2:])
            else:
                title = ",".join(title[3:])

            title = title.replace("&amp;", "&")
            if not title:  # we use only mentions with valid referent entities
                continue

            mention_data[doc_name].append(dict(text=mention_text, candidates=candidates, title=title))

    def find_coreference(target_mention, mention_list):
        target_mention_text = target_mention["text"].lower()
        ret = []

        for mention in mention_list:
            if not mention["candidates"] or mention["candidates"][0].title not in person_names:
                continue

            mention_text = mention["text"].lower()
            if mention_text == target_mention_text:
                continue

            start_pos = mention_text.find(target_mention_text)
            if start_pos == -1:
                continue

            end_pos = start_pos + len(target_mention_text) - 1

            if (start_pos == 0 or mention_text[start_pos - 1] == " ") and (
                end_pos == len(mention_text) - 1 or mention_text[end_pos + 1] == " "
            ):
                ret.append(mention)

        return ret

    for _, mentions in mention_data.items():
        for mention in mentions:
            coref_mentions = find_coreference(mention, mentions)
            if coref_mentions:
                new_cands = defaultdict(int)
                for coref_mention in coref_mentions:
                    for candidate in coref_mention["candidates"]:
                        new_cands[candidate.title] += candidate.prior_prob
                for candidate_title in new_cands.keys():
                    new_cands[candidate_title] /= len(coref_mentions)

                mention["candidates"] = sorted(
                    [Candidate(t, p) for (t, p) in new_cands.items()], key=lambda c: c.prior_prob, reverse=True
                )
    return mention_data


def convert_documents_to_features(
    documents,
    tokenizer,
    entity_vocab,
    mode,
    document_split_mode,
    max_seq_length,
    max_candidate_length,
    max_mention_length,
):
    max_num_tokens = max_seq_length - 2

    def generate_feature_dict(tokens, mentions, ctx_start, ctx_end, doc_start, doc_end):

        ################################################################
        # TODO : UNKO と word_ids のトークナイズを外している部分を戻す #
        ################################################################
        # all_tokens = [tokenizer.cls_token] + tokens[doc_start:doc_end] + [tokenizer.sep_token]
        all_tokens = ["unko1"] + tokens[ctx_start:ctx_end] + ["unko2"]
        # word_ids = np.array(tokenizer.convert_tokens_to_ids(all_tokens), dtype=np.int)
        word_ids = np.ones(len(all_tokens), dtype=np.int)
        word_attention_mask = np.ones(len(all_tokens), dtype=np.int)
        word_segment_ids = np.zeros(len(all_tokens), dtype=np.int)

        target_mention_data = []
        for start, end, mention in mentions:
            if start >= doc_start and end <= doc_end:
                candidates = [c.title for c in mention.candidates[:max_candidate_length]]
                candidates.append("[NO_E]")

                if mode == "train" and mention.title not in candidates:
                    continue
                if end - start > max_mention_length:
                    continue
                target_mention_data.append((start - ctx_start, end - ctx_start, mention, candidates))

        entity_ids = np.empty(len(target_mention_data), dtype=np.int)
        entity_attention_mask = np.ones(len(target_mention_data), dtype=np.int)
        entity_segment_ids = np.zeros(len(target_mention_data), dtype=np.int)
        entity_position_ids = np.full((len(target_mention_data), max_mention_length), -1, dtype=np.int)
        entity_candidate_ids = np.zeros((len(target_mention_data), max_candidate_length + 1), dtype=np.int)

        for index, (start, end, mention, candidates) in enumerate(target_mention_data):
            print(mention.title, "---- ", start + 1, ":", end + 1)
            entity_ids[index] = entity_vocab[mention.title]
            entity_position_ids[index][: end - start] = range(start + 1, end + 1)  # +1 for [CLS]
            entity_candidate_ids[index, : len(candidates)] = [entity_vocab[cand] for cand in candidates]
        output_mentions = [mention for _, _, mention, _ in target_mention_data]

        return (
            output_mentions,
            dict(
                word_ids=word_ids,
                word_segment_ids=word_segment_ids,
                word_attention_mask=word_attention_mask,
                entity_ids=entity_ids,
                entity_position_ids=entity_position_ids,
                entity_segment_ids=entity_segment_ids,
                entity_attention_mask=entity_attention_mask,
                entity_candidate_ids=entity_candidate_ids,
            ),
        )

    ret = []
    for document in documents:
        tokens = []
        mention_data = []
        cur = 0

        for mention in document.mentions:
            tokens = [w for ws in document.words[: mention.start] for w in tokenizer.tokenize(ws)]
            mention_tokens = [w for ws in document.words[mention.start : mention.end] for w in tokenizer.tokenize(ws)]
            mention_data.append((len(tokens), len(tokens) + len(mention_tokens), mention))
        tokens = [w for ws in document.words for w in tokenizer.tokenize(ws)]

        feed_flag = [False] * len(tokens)  # 改行文字なしのlength #
        quasi_tokens = []  # 改行文字込みの token 列
        for ws in document.words:
            if ws == "\n":
                quasi_tokens.append(ws)
                continue
            for w in tokenizer.tokenize(ws):
                quasi_tokens.append(w)
        feed_count = 0
        for i, ws in enumerate(quasi_tokens):
            if ws == "\n":
                feed_count += 1
                feed_flag[i - feed_count] = True

        seq_record = []
        s_ = 0
        for i, flg in enumerate(feed_flag):
            if flg == True:
                seq_record.append((s_, i + 1))
                s_ = i + 1

        for doc_start, doc_end in seq_record:
            seq_len = doc_end - doc_start
            rest_len = max_num_tokens - seq_len
            if rest_len > 0:
                left_ctx_len = math.ceil(rest_len / 2)
                if doc_start < left_ctx_len:
                    left_ctx_len = doc_start
                right_ctx_len = rest_len - left_ctx_len
                if len(tokens) - doc_end < right_ctx_len:
                    right_ctx_len = len(tokens) - doc_end
            else:
                left_ctx_len = right_ctx_len = 0
            output_mentions, feature_dict = generate_feature_dict(
                tokens, mention_data, doc_start - left_ctx_len, doc_end + right_ctx_len, doc_start, doc_end
            )
            if output_mentions:
                ret.append(
                    InputFeatures(
                        document=document,
                        mentions=output_mentions,
                        target_mention_indices=range(len(output_mentions)),
                        **feature_dict
                    )
                )
    return ret
