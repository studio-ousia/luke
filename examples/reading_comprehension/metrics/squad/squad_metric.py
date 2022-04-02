import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Union

import numpy as np
import torch
from allennlp.common.util import sanitize_wordpiece
from allennlp.data import Token
from allennlp_models.rc.metrics import SquadEmAndF1
from transformers import AutoTokenizer

from examples.reading_comprehension.metrics.qa_metric import QAMetric
from examples.reading_comprehension.metrics.squad.squad_evaluate import evaluate_from_files

logger = logging.getLogger(__name__)


class SQuADPrediction(NamedTuple):
    prediction: str
    score: float


@QAMetric.register("squad-v1.1")
class SQuADMetric(QAMetric):
    def __init__(
        self, gold_data_path: str, prediction_dump_path: str, transformers_tokenizer_name: str,
    ):
        super().__init__()
        self.document_predictions: Dict[str, List[SQuADPrediction]] = defaultdict(list)
        self.passage_answer_candidates = {}
        self.example_to_language = {}
        assert Path(gold_data_path).exists()
        self.gold_data_path = gold_data_path
        self.prediction_dump_path = prediction_dump_path

        self._tokenizer = AutoTokenizer.from_pretrained(transformers_tokenizer_name)

        self.per_instance_metric = SquadEmAndF1()

        self.count = 0

    def __call__(self, output_dict: Dict[str, torch.Tensor], metadata_list: List[Dict]):
        prediction_logits = output_dict["prediction_logits"].tolist()
        for metadata, (best_span), score, cspan in zip(
            metadata_list, output_dict["best_span"], prediction_logits, output_dict["context_span"]
        ):
            best_span_strings = _collect_best_span_string(
                best_span, cspan, metadata["context_tokens"], metadata["context"]
            )
            self.document_predictions[metadata["example_id"]].append(SQuADPrediction(best_span_strings, score))
            self.per_instance_metric(best_span_strings, metadata["answers"])

    def get_metric(self, reset: bool) -> Dict[str, float]:

        exact_match, f1_score = self.per_instance_metric.get_metric(reset=reset)
        result_dict = {"per_instance_em": exact_match, "per_instance_f1": f1_score}

        if not reset:
            return result_dict

        prediction_dict = {}
        for example_id, predictions in self.document_predictions.items():
            prediction, score = max(predictions, key=lambda x: x.score)
            if prediction is not None:
                prediction_dict[example_id] = prediction

        prediction_dump_path = self.prediction_dump_path + f"_{self.count}"
        Path(prediction_dump_path).parent.mkdir(exist_ok=True, parents=True)
        with open(prediction_dump_path, "w") as f:
            json.dump(prediction_dict, f, indent=4)

        self.count += 1

        result_dict.update(evaluate_from_files(self.gold_data_path, prediction_dump_path))

        return result_dict


def _collect_best_span_string(
    best_span: torch.Tensor,
    cspan: torch.IntTensor,
    context_tokens: List[Token],
    context_string: str,
    cls_ind: Optional[Union[torch.LongTensor, int]] = 0,
) -> str:
    """
    Collect the string of the best predicted span from the context metadata and
    update `self._per_instance_metrics`, which in the case of SQuAD v1.1 / v2.0
    includes the EM and F1 score.

    This returns a `Tuple[List[str], torch.Tensor]`, where the `List[str]` is the
    predicted answer for each instance in the batch, and the tensor is just the input
    tensor `best_spans` after adjustments so that each answer span corresponds to the
    context tokens only, and not the question tokens. Spans that correspond to the
    `[CLS]` token, i.e. the question was predicted to be impossible, will be set
    to `(-1, -1)`.
    """
    best_span = best_span.detach().cpu().numpy()

    if best_span[0] == cls_ind:
        # Predicting [CLS] is interpreted as predicting the question as unanswerable.
        best_span_string = ""

    else:
        best_span -= int(cspan[0])
        assert np.all(best_span >= 0)

        predicted_start, predicted_end = tuple(best_span)

        while predicted_start >= 0 and context_tokens[predicted_start].idx is None:
            predicted_start -= 1
        if predicted_start < 0:
            logger.warning(
                f"Could not map the token '{context_tokens[best_span[0]].text}' at index "
                f"'{best_span[0]}' to an offset in the original text."
            )
            character_start = 0
        else:
            character_start = context_tokens[predicted_start].idx

        while predicted_end < len(context_tokens) and context_tokens[predicted_end].idx is None:
            predicted_end += 1
        if predicted_end >= len(context_tokens):
            logger.warning(
                f"Could not map the token '{context_tokens[best_span[1]].text}' at index "
                f"'{best_span[1]}' to an offset in the original text."
            )
            character_end = len(context_string)
        else:
            end_token = context_tokens[predicted_end]
            character_end = end_token.idx + len(sanitize_wordpiece(end_token.text))

        best_span_string = context_string[character_start:character_end]

    return best_span_string
