import itertools
from typing import Dict, List

import torch
import torch.nn as nn
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models import Model
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.training.metrics import Average, BooleanAccuracy, CategoricalAccuracy

from examples.reading_comprehension.metrics.qa_metric import QAMetric


@Model.register("transformers_qa")
class TransformersQAModel(Model):
    """
    Model based on
    ``A BERT Baseline for the Natural Questions``
    (https://arxiv.org/abs/1901.08634)
    """

    def __init__(
        self,
        vocab: Vocabulary,
        embedder: TextFieldEmbedder,
        encoder: Seq2SeqEncoder = None,
        dropout: float = 0.1,
        answer_type_name_space: str = "answer_type",
        max_sequence_length: int = 512,
        num_best_predictions: int = 20,
        max_answer_length: int = 30,
        qa_metric: QAMetric = None,
    ):

        super().__init__(vocab=vocab)
        self.embedder = embedder
        self.encoder = encoder

        feature_size = self.encoder.get_output_dim() if self.encoder else self.embedder.get_output_dim()
        self.max_sequence_length = max_sequence_length
        self.span_scoring_layer = nn.Linear(feature_size, 2)

        self.answer_type_name_space = answer_type_name_space
        if self.answer_type_name_space is not None:
            self.answer_type_classifier = nn.Linear(feature_size, vocab.get_vocab_size(answer_type_name_space))

        self.dropout = nn.Dropout(p=dropout)
        self.criterion = nn.CrossEntropyLoss()

        self.metrics = {
            "span_loss": Average(),
            "answer_type_loss": Average(),
            "answer_type_accuracy": CategoricalAccuracy(),
            "span_accuracy": BooleanAccuracy(),
            "span_start_accuracy": CategoricalAccuracy(),
            "span_end_accuracy": CategoricalAccuracy(),
        }

        self.num_best_predictions = num_best_predictions
        self.max_answer_length = max_answer_length

        self.qa_metric = qa_metric

    def forward(
        self,
        question_with_context: TextFieldTensors,
        context_span: torch.LongTensor,
        answer_span: torch.LongTensor = None,
        answer_type: torch.LongTensor = None,
        metadata: List = None,
        entity_ids: torch.LongTensor = None,
        entity_position_ids: torch.LongTensor = None,
        entity_segment_ids: torch.LongTensor = None,
        entity_attention_mask: torch.LongTensor = None,
        **kwargs
    ):

        if entity_ids is not None:
            question_with_context["tokens"]["entity_ids"] = entity_ids
            question_with_context["tokens"]["entity_position_ids"] = entity_position_ids
            question_with_context["tokens"]["entity_segment_ids"] = entity_segment_ids
            question_with_context["tokens"]["entity_attention_mask"] = entity_attention_mask

        token_embeddings = self.embedder(question_with_context)
        if self.encoder is not None:
            token_embeddings = self.encoder(token_embeddings)

        # compute logits for span prediction
        # shape: (batch_size, sequence_length, feature_size) -> (batch_size, sequence_length, 2)
        span_start_end_logits = self.span_scoring_layer(token_embeddings)

        # shape: (batch_size, sequence_length)
        span_start_logits = span_start_end_logits[:, :, 0]
        span_end_logits = span_start_end_logits[:, :, 1]

        output_dict = {
            "span_start_logits": span_start_logits,
            "span_end_logits": span_end_logits,
            "context_span": context_span,
        }

        if self.answer_type_name_space is not None:
            # compute logits for answer type prediction
            cls_embeddings = token_embeddings[:, 0]
            answer_type_logits = self.answer_type_classifier(cls_embeddings)
            answer_type_prediction = answer_type_logits.argmax(dim=1)
            output_dict.update(
                {"answer_type_logits": answer_type_logits, "answer_type_prediction": answer_type_prediction}
            )

        output_dict.update(self._get_best_span(span_start_logits, span_end_logits, context_span))
        if answer_span is not None:
            span_loss = self._evaluate_span(output_dict["best_span"], span_start_logits, span_end_logits, answer_span)
            self.metrics["span_loss"](span_loss.item())
            output_dict["loss"] = span_loss

            if self.answer_type_name_space is not None and answer_type is not None:
                # predict answer type
                answer_type_loss = self.criterion(answer_type_logits, answer_type)
                self.metrics["answer_type_loss"](span_loss.item())
                self.metrics["answer_type_accuracy"](answer_type_logits, answer_type)
                output_dict["loss"] += answer_type_loss

        if not self.training and self.qa_metric:
            self.qa_metric(output_dict, metadata)

        return output_dict

    def _evaluate_span(
        self,
        best_spans: torch.Tensor,
        span_start_logits: torch.Tensor,
        span_end_logits: torch.Tensor,
        answer_span: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate the loss against the `answer_span` and also update the span metrics.
        """
        span_start = answer_span[:, 0]
        span_end = answer_span[:, 1]
        self.metrics["span_accuracy"](best_spans, answer_span)

        start_loss = self.criterion(span_start_logits, span_start)
        end_loss = self.criterion(span_end_logits, span_end)

        self.metrics["span_start_accuracy"](span_start_logits, span_start)
        self.metrics["span_end_accuracy"](span_end_logits, span_end)

        return (start_loss + end_loss) / 2

    def _get_best_span(
        self, span_start_logits: torch.Tensor, span_end_logits: torch.Tensor, context_span: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        batch_size = span_start_logits.size(0)
        # span_start_prediction_logits, span_start_prediction = torch.max(span_start_logits, dim=1)
        # span_end_prediction_logits, span_end_prediction = torch.max(span_end_logits, dim=1)

        # masking out indices outside the context_span
        for i in range(batch_size):
            s, e = context_span[i]
            small_constant = torch.finfo(span_start_logits.dtype).min
            span_start_logits[i][1:s] = small_constant
            span_end_logits[i][e + 1 :] = small_constant

        start_topk = torch.topk(span_start_logits, k=self.num_best_predictions, dim=1, sorted=True)
        end_topk = torch.topk(span_end_logits, k=self.num_best_predictions, dim=1, sorted=True)

        null_prediction_logits = (span_start_logits[:, 0] + span_end_logits[:, 0]).detach()

        prediction_logits_list = []
        prediction_list = []
        # the first loop is loop over the batch dimension
        for i, (topk_start, topk_end, c_span) in enumerate(zip(zip(*start_topk), zip(*end_topk), context_span)):
            prediction_candidates = []
            for (s_score, s_idx), (e_score, e_idx) in itertools.product(zip(*topk_start), zip(*topk_end)):
                # skip if the predicted span is ill-formed.
                if s_idx > e_idx:
                    continue

                # skip if the predicted span is not pointed to CLS (null prediction) and outside the context span.
                if not s_idx == e_idx == 0 and (s_idx < c_span[0] or e_idx > c_span[1]):
                    continue

                # skip if the predicted span is logner than max_answer_length.
                if e_idx - s_idx + 1 > self.max_answer_length:
                    continue

                prediction_candidates.append((s_idx, e_idx, s_score + e_score))

            if len(prediction_candidates) > 0:
                best_s_idx, best_e_idx, best_score = max(prediction_candidates, key=lambda x: x[2])
                prediction_logits_list.append(best_score)
                prediction_list.append((best_s_idx, best_e_idx))
            else:
                prediction_logits_list.append(null_prediction_logits[i])
                prediction_list.append((0, 0))

        prediction_logits = torch.Tensor(prediction_logits_list).to(null_prediction_logits.device)
        best_span = torch.LongTensor(prediction_list).to(context_span.device)
        return {
            "prediction_logits": prediction_logits - null_prediction_logits,
            "best_span": best_span,
        }

    def get_metrics(self, reset: bool = False):
        metric_results = {k: metric.get_metric(reset=reset) for k, metric in self.metrics.items()}
        if self.qa_metric is not None:
            metric_results.update(self.qa_metric.get_metric(reset=reset))
        return metric_results
