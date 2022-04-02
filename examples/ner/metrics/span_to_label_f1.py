import json
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
from allennlp.data import Vocabulary
from allennlp.training.metrics import Metric
from seqeval.metrics import f1_score, precision_score, recall_score
from seqeval.scheme import IOB2


class SpanToLabelF1(Metric):
    def __init__(self, vocab: Vocabulary, label_namespace: str = "labels", prediction_save_path: str = None):
        self.vocab = vocab
        self.label_namespace = label_namespace

        self.prediction = defaultdict(list)
        self.gold_labels = defaultdict(list)
        self.doc_id_to_words: Dict[str, List[str]] = {}
        self.prediction_save_path = prediction_save_path

    def __call__(
        self,
        prediction: torch.Tensor,
        gold_labels: torch.Tensor,
        prediction_scores: torch.Tensor,
        original_entity_spans: torch.Tensor,
        doc_id: List[str],
        input_words: List[List[str]] = None,
    ):

        if self.prediction_save_path is not None and input_words is None:
            raise RuntimeError("If you want to dump predictions, you need input_words.")

        prediction, gold_labels, prediction_scores, original_entity_spans = self.detach_tensors(
            prediction, gold_labels, prediction_scores, original_entity_spans
        )

        if input_words is not None:
            for id_, words in zip(doc_id, input_words):
                self.doc_id_to_words[id_] = words

        for pred, gold, scores, spans, id_ in zip(
            prediction, gold_labels, prediction_scores, original_entity_spans, doc_id
        ):
            pred = pred.tolist()
            gold = gold.tolist()
            scores = scores.tolist()
            spans = spans.tolist()
            for p, g, score, span in zip(pred, gold, scores, spans):
                if g == -1:
                    continue
                p = self.vocab.get_token_from_index(p, namespace=self.label_namespace)
                g = self.vocab.get_token_from_index(g, namespace=self.label_namespace)

                self.prediction[id_].append((score, span, p))
                self.gold_labels[id_].append((0, span, g))

    def reset(self):
        self.prediction = defaultdict(list)
        self.gold_labels = defaultdict(list)

    def get_metric(self, reset: bool):
        if not reset:
            return {}

        all_prediction_sequence = []
        all_gold_sequence = []
        results = []
        for doc_id in self.gold_labels.keys():
            prediction = self.span_to_label_sequence(self.prediction[doc_id])
            gold = self.span_to_label_sequence(self.gold_labels[doc_id])
            all_prediction_sequence.append(prediction)
            all_gold_sequence.append(gold)
            results.append({"words": self.doc_id_to_words[doc_id], "gold": gold, "prediction": prediction})

        if self.prediction_save_path is not None:
            with open(self.prediction_save_path, "w") as f:
                json.dump(results, f)

        return dict(
            f1=f1_score(all_gold_sequence, all_prediction_sequence, scheme=IOB2),
            precision=precision_score(all_gold_sequence, all_prediction_sequence, scheme=IOB2),
            recall=recall_score(all_gold_sequence, all_prediction_sequence, scheme=IOB2),
        )

    @staticmethod
    def span_to_label_sequence(span_and_labels: List[Tuple[float, Tuple[int, int], str]]) -> List[str]:
        sequence_length = max([end for score, (start, end), label in span_and_labels])
        label_sequence = ["O"] * sequence_length
        for score, (start, end), label in sorted(span_and_labels, key=lambda x: -x[0]):
            if label == "O" or any([l != "O" for l in label_sequence[start:end]]):
                continue
            label_sequence[start:end] = ["I-" + label] * (end - start)
            label_sequence[start] = "B-" + label

        return label_sequence
