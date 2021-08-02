from typing import List
import torch
import torch.nn as nn

from allennlp.data import Vocabulary, TextFieldTensors
from allennlp.models import Model
from allennlp.training.metrics import CategoricalAccuracy

from .modules.feature_extractor import NERFeatureExtractor
from .metrics.span_to_label_f1 import SpanToLabelF1


@Model.register("span_ner")
class ExhaustiveNERModel(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        feature_extractor: NERFeatureExtractor,
        dropout: float = 0.1,
        label_name_space: str = "labels",
        text_field_key: str = "tokens",
        prediction_save_path: str = None
    ):
        super().__init__(vocab=vocab)
        self.feature_extractor = feature_extractor

        self.text_field_key = text_field_key
        self.classifier = nn.Linear(self.feature_extractor.get_output_dim(), vocab.get_vocab_size(label_name_space))

        self.dropout = nn.Dropout(p=dropout)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)

        self.span_f1 = SpanToLabelF1(self.vocab, prediction_save_path=prediction_save_path)
        self.span_accuracy = CategoricalAccuracy()

    def forward(
        self,
        word_ids: TextFieldTensors,
        entity_start_positions: torch.LongTensor,
        entity_end_positions: torch.LongTensor,
        original_entity_spans: torch.LongTensor,
        doc_id: List[str],
        labels: torch.LongTensor = None,
        entity_ids: torch.LongTensor = None,
        entity_position_ids: torch.LongTensor = None,
        input_words: List[List[str]] = None,
        **kwargs,
    ):
        feature_vector = self.feature_extractor(
            word_ids[self.text_field_key], entity_start_positions, entity_end_positions, entity_ids, entity_position_ids
        )

        feature_vector = self.dropout(feature_vector)
        logits = self.classifier(feature_vector)
        prediction_logits, prediction = logits.max(dim=-1)
        output_dict = {"logits": logits, "prediction": prediction, "input": input_words}

        if labels is not None:
            output_dict["loss"] = self.criterion(logits.flatten(0, 1), labels.flatten())
            self.span_accuracy(logits, labels, mask=(labels != -1))
            self.span_f1(prediction, labels, prediction_logits, original_entity_spans, doc_id, input_words)

        return output_dict

    def get_metrics(self, reset: bool = False):
        output_dict = self.span_f1.get_metric(reset)
        output_dict["span_accuracy"] = self.span_accuracy.get_metric(reset)
        return output_dict
