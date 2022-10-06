# This code is based on the code obtained from here:
# https://github.com/lephong/mulrel-nel/blob/db14942450f72c87a4d46349860e96ef2edf353d/nel/dataset.py

import copy
import dataclasses
import glob
import logging
import os
import re
from collections import defaultdict
from typing import AbstractSet, Dict, List, Optional


logger = logging.getLogger(__name__)

DATASET_FILE_NAMES = {
    "train": ("aida_train.csv", "aida_train.txt"),
    "test_a": ("aida_testA.csv", "testa_testb_aggregate_original"),
    "test_b": ("aida_testB.csv", "testa_testb_aggregate_original"),
    "ace2004": ("wned-ace2004.csv", "ace2004.conll"),
    "aquaint": ("wned-aquaint.csv", "aquaint.conll"),
    "clueweb": ("wned-clueweb.csv", "clueweb.conll"),
    "msnbc": ("wned-msnbc.csv", "msnbc.conll"),
    "wikipedia": ("wned-wikipedia.csv", "wikipedia.conll"),
}


@dataclasses.dataclass
class Candidate:
    title: str
    prior_prob: float


@dataclasses.dataclass
class Mention:
    text: str
    title: str
    index: int
    candidates: List[Candidate]
    start: Optional[int] = None
    end: Optional[int] = None


@dataclasses.dataclass
class Document:
    id: str
    words: List[str]
    mentions: List[Mention]


@dataclasses.dataclass
class EntityDisambiguationDataset:
    train: List[Document]
    test_a: List[Document]
    test_b: List[Document]
    ace2004: List[Document]
    aquaint: List[Document]
    clueweb: List[Document]
    msnbc: List[Document]
    wikipedia: List[Document]
    test_a_ppr: Optional[List[Document]] = None  # test_a dataset with ppr candidates
    test_b_ppr: Optional[List[Document]] = None  # test_b dataset with ppr candidates

    def get_all_datasets(self) -> List[List[Document]]:
        all_datasets = [
            self.train,
            self.test_a,
            self.test_b,
            self.ace2004,
            self.aquaint,
            self.clueweb,
            self.msnbc,
            self.wikipedia,
        ]
        if self.test_a_ppr is not None:
            all_datasets += [self.test_a_ppr, self.test_b_ppr]

        return all_datasets

    def get_dataset(self, dataset_name: str) -> List[Document]:
        return getattr(self, dataset_name)


def load_dataset(
    dataset_dir: str,
    titles_file: Optional[str] = None,
    redirects_file: Optional[str] = None,
    ppr_for_ned_dir: Optional[str] = None,
) -> EntityDisambiguationDataset:
    with open(os.path.join(dataset_dir, "persons.txt")) as f:
        person_names = frozenset([line.strip() for line in f])
    if titles_file:
        with open(titles_file) as f:
            valid_titles = frozenset([line.rstrip() for line in f])
    else:
        valid_titles = None
    redirects = {}
    if redirects_file:
        with open(redirects_file) as f:
            for line in f:
                src, dest = line.rstrip().split("\t")
                redirects[src] = dest

    all_datasets = {}
    for dataset_name, (tsv_file_name, conll_file_name) in DATASET_FILE_NAMES.items():
        tsv_file = os.path.join(dataset_dir, tsv_file_name)
        conll_file = os.path.join(dataset_dir, conll_file_name)
        all_datasets[dataset_name] = _load_documents(tsv_file, conll_file, person_names, valid_titles, redirects)

    if ppr_for_ned_dir:
        all_datasets["test_a_ppr"] = _load_ppr_candidates(
            copy.deepcopy(all_datasets["test_a"]), ppr_for_ned_dir, redirects
        )
        all_datasets["test_b_ppr"] = _load_ppr_candidates(
            copy.deepcopy(all_datasets["test_b"]), ppr_for_ned_dir, redirects
        )

    return EntityDisambiguationDataset(**all_datasets)


def _load_documents(
    tsv_file: str,
    conll_file: str,
    person_names: AbstractSet[str],
    valid_titles: Optional[AbstractSet[str]],
    redirects: Dict[str, str],
) -> List[Document]:
    doc_id2mentions: Dict[str, List[Mention]] = defaultdict(list)
    with open(tsv_file, "r") as f:
        mention_index = 0
        for line in f:
            items = line.strip().split("\t")
            doc_id = items[0]
            mention_text = items[2]

            if items[6] != "EMPTYCAND":
                candidates = []
                for candidate_str in items[6:-2]:
                    candidate_data = candidate_str.split(",")
                    candidate_title = ",".join(candidate_data[2:])
                    prior_prob = float(candidate_data[1])
                    if candidate_title:
                        candidates.append(Candidate(candidate_title, prior_prob))
                candidates = sorted(candidates, key=lambda c: c.prior_prob, reverse=True)
            else:
                candidates = []

            gold_title_data = items[-1].split(",")
            if gold_title_data[0] == "-1":
                gold_title = ",".join(gold_title_data[2:])
            else:
                gold_title = ",".join(gold_title_data[3:])
            if not gold_title:  # we use only mentions with valid referent entities
                continue
            gold_title = gold_title.replace("&amp;", "&")

            mention = Mention(text=mention_text, title=gold_title, index=mention_index, candidates=candidates)
            doc_id2mentions[doc_id].append(mention)
            mention_index += 1

    for doc_id, mentions in doc_id2mentions.items():
        orig_mentions = copy.deepcopy(mentions)
        for mention in mentions:
            coref_mentions = _find_coreference(mention, orig_mentions, person_names=person_names)
            if coref_mentions:
                new_prior_probs: Dict[str, float] = defaultdict(float)
                for coref_mention in coref_mentions:
                    for candidate in coref_mention.candidates:
                        new_prior_probs[candidate.title] += candidate.prior_prob

                for candidate_title in new_prior_probs.keys():
                    new_prior_probs[candidate_title] /= len(coref_mentions)

                mention.candidates = sorted(
                    [Candidate(title, prior_prob) for (title, prior_prob) in new_prior_probs.items()],
                    key=lambda c: c.prior_prob,
                    reverse=True,
                )

    doc_name2words: Dict[str, List[str]] = defaultdict(list)
    doc_name2mention_spans: Dict[str, List[List[int]]] = defaultdict(list)
    with open(conll_file, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("-DOCSTART-"):
                doc_name = line.split()[1][1:]
            else:
                items = line.split("\t")
                if len(items) >= 6:
                    tag = items[1]
                    if tag == "I":
                        doc_name2mention_spans[doc_name][-1][1] += 1
                    elif tag == "B":
                        start = len(doc_name2words[doc_name])
                        doc_name2mention_spans[doc_name].append([start, start + 1])
                    else:
                        raise RuntimeError(f"Invalid format: {conll_file}")

                doc_name2words[doc_name].append(items[0])

    documents = []
    punc_remover = re.compile(r"[\W]+")
    for doc_id, mentions in doc_id2mentions.items():
        # This document is excluded in Le and Titov 2018:
        # https://github.com/lephong/mulrel-nel/blob/db14942450f72c87a4d46349860e96ef2edf353d/nel/dataset.py#L221
        if doc_id == "Jiří_Třanovský":
            continue
        doc_name = doc_id.split()[0]

        mention_index = 0
        for mention in mentions:
            mention_text = punc_remover.sub("", mention.text.lower())
            while True:
                mention_start, mention_end = doc_name2mention_spans[doc_name][mention_index]
                mention_index += 1
                conll_mention_text = " ".join(doc_name2words[doc_name][mention_start:mention_end])
                conll_mention_text = punc_remover.sub("", conll_mention_text.lower())
                if conll_mention_text == mention_text:
                    mention.start = mention_start
                    mention.end = mention_end
                    break

        valid_mentions = []
        for mention in mentions:
            mention.title = redirects.get(mention.title, mention.title)
            if valid_titles is not None and mention.title not in valid_titles:
                logger.debug(f"Missing entity: {mention.title}")
                continue
            valid_mentions.append(mention)

        for mention in valid_mentions:
            for candidate in mention.candidates:
                candidate.title = redirects.get(candidate.title, candidate.title)

        document = Document(id=doc_id, words=doc_name2words[doc_name], mentions=valid_mentions)
        documents.append(document)

    return documents


def _find_coreference(mention: Mention, all_mentions: List[Mention], person_names: AbstractSet[str]) -> List[Mention]:
    mention_text = mention.text.lower()
    ret = []
    for target_mention in all_mentions:
        if not target_mention.candidates or target_mention.candidates[0].title not in person_names:
            continue

        target_mention_text = target_mention.text.lower()
        if target_mention_text == mention_text:
            continue

        start = target_mention_text.find(mention_text)
        if start == -1:
            continue

        end = start + len(mention_text) - 1
        if (start == 0 or target_mention_text[start - 1] == " ") and (
            end == len(target_mention_text) - 1 or target_mention_text[end + 1] == " "
        ):
            ret.append(target_mention)

    return ret


def _load_ppr_candidates(documents: List[Document], ppr_for_ned_dir: str, redirects: Dict[str, str]):
    for document in documents:
        file_name = re.match(r"^\d*", document.id).group(0)
        file_path = glob.glob(f"{ppr_for_ned_dir}/**/{file_name}", recursive=True)[0]
        mention_texts = []
        candidate_titles: List[List[str]] = []
        with open(file_path) as f:
            for line in f:
                if line.startswith("ENTITY"):
                    mention_text = line.split("\t")[7][9:]
                    mention_texts.append(mention_text)
                    candidate_titles.append([])

                elif line.startswith("CANDIDATE"):
                    uri = line.split("\t")[5][4:]
                    title = uri[29:].replace("_", " ")
                    candidate_titles[-1].append(title)

        index = 0
        punc_remover = re.compile(r"[\W]+")
        for mention in document.mentions:
            mention_text = punc_remover.sub("", mention.text.lower())
            while mention_text != punc_remover.sub("", mention_texts[index].lower()):
                index += 1

            candidates = []
            for title in candidate_titles[index]:
                title = redirects.get(title, title)
                candidates.append(Candidate(title, -1))
            mention.candidates = candidates
            index += 1

    return documents
