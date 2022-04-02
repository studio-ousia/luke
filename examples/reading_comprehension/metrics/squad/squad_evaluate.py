import functools
import json
import sys
from collections import Counter
from dataclasses import dataclass
from typing import Callable, List

from examples.reading_comprehension.metrics.squad.mlqa_extensions.mlqa_languages import MLQALanguage
from examples.reading_comprehension.readers.squad_reader import SQuADBasedDataset, SQuADFile

from .answer_string import AnswerString
from .mlqa_extensions import MLQAAnswerString, MLQALanguage


@dataclass
class Prediction:
    id: str
    answer_string: AnswerString


class PredictionList:
    def __init__(self, predictions: List[Prediction]):
        self.id_to_prediction = {p.id: p for p in predictions}
        assert len(predictions) == len(self.id_to_prediction)

    def __contains__(self, item: str):
        return item in self.id_to_prediction

    def __getitem__(self, item: str) -> Prediction:
        return self.id_to_prediction[item]

    def __iter__(self):
        return iter(self.id_to_prediction.values())

    @classmethod
    def from_file(cls, prediction_file_path: str, answer_string: Callable = None):
        if answer_string is None:
            answer_string = AnswerString
        with open(prediction_file_path, "r") as prediction_file:
            predictions = json.load(prediction_file)
        predictions = [Prediction(id_, answer_string(answer)) for id_, answer in predictions.items()]
        return cls(predictions)


@dataclass
class GroundTruth:
    id: str
    answer_strings: List[AnswerString]


class GroundTruthList:
    def __init__(self, ground_truth_list: List[GroundTruth]):
        self.id_to_ground_truth = {g.id: g for g in ground_truth_list}
        assert len(ground_truth_list) == len(self.id_to_ground_truth)

    def __contains__(self, item: str):
        return item in self.id_to_ground_truth

    def __getitem__(self, item: str) -> GroundTruth:
        return self.id_to_ground_truth[item]

    def __iter__(self):
        return iter(self.id_to_ground_truth.values())

    @classmethod
    def from_file(cls, squad_file_path: str, answer_string: Callable = None):
        if answer_string is None:
            answer_string = AnswerString

        with open(squad_file_path, "r") as dataset_file:
            dataset_json = json.load(dataset_file)
            data = dataset_json["data"]

        ground_truth_list = []
        for article in data:
            for paragraph in article["paragraphs"]:
                for qa in paragraph["qas"]:
                    ground_truth = GroundTruth(qa["id"], [answer_string(a["text"]) for a in qa["answers"]])
                    ground_truth_list.append(ground_truth)
        return cls(ground_truth_list)


def f1_score(prediction: AnswerString, ground_truth: AnswerString) -> float:
    prediction_tokens = prediction.get_tokens()
    ground_truth_tokens = ground_truth.get_tokens()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction: AnswerString, ground_truth: AnswerString) -> bool:
    return prediction.get_normalized_answer() == ground_truth.get_normalized_answer()


def metric_max_over_ground_truths(metric_fn: Callable, prediction: Prediction, ground_truths: GroundTruth):
    assert prediction.id == ground_truths.id
    scores_for_ground_truths = []
    for ground_truth_answer in ground_truths.answer_strings:
        score = metric_fn(prediction.answer_string, ground_truth_answer)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def evaluate(ground_truth_list: GroundTruthList, predictions_list: PredictionList):
    f1 = exact_match = total = 0

    for ground_truth in ground_truth_list:
        id_ = ground_truth.id

        if id_ not in predictions_list:
            message = "Unanswered question " + id_ + " will receive score 0."
            print(message, file=sys.stderr)
            continue

        prediction = predictions_list[id_]

        exact_match += metric_max_over_ground_truths(exact_match_score, prediction, ground_truth)
        f1 += metric_max_over_ground_truths(f1_score, prediction, ground_truth)
        total += 1

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    return {"exact_match": exact_match, "f1": f1}


def evaluate_from_files(gold_file_path: str, prediction_file_path: str):
    squad_file = SQuADFile.from_path(gold_file_path)

    if squad_file.dataset == SQuADBasedDataset.MLQA:
        answer_string = functools.partial(
            MLQAAnswerString, language=MLQALanguage.from_code(squad_file.context_language)
        )
    else:
        answer_string = AnswerString

    ground_truth_list = GroundTruthList.from_file(gold_file_path, answer_string)
    prediction_list = PredictionList.from_file(prediction_file_path, answer_string)

    return evaluate(ground_truth_list, prediction_list)
