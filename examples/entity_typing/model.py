from typing import Dict, List

import torch
import torch.nn as nn
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models import Model
from allennlp.training.metrics import F1MultiLabelMeasure

from .modules.feature_extractor import ETFeatureExtractor


@Model.register("entity_typing")
class EntityTypeClassifier(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        feature_extractor: ETFeatureExtractor,
        dropout: float = 0.1,
        label_name_space: str = "labels",
        text_field_key: str = "tokens",
    ):

        super().__init__(vocab=vocab)
        self.feature_extractor = feature_extractor
        self.classifier = nn.Linear(self.feature_extractor.get_output_dim(), vocab.get_vocab_size(label_name_space))

        self.text_field_key = text_field_key
        self.label_name_space = label_name_space

        self.dropout = nn.Dropout(p=dropout)
        self.criterion = nn.BCEWithLogitsLoss()

        self.metrics = {}
        self.f1_score = F1MultiLabelMeasure(average="micro", threshold=0.0)

    def forward(
        self,
        word_ids: TextFieldTensors,
        entity_span: torch.LongTensor,
        labels: torch.LongTensor = None,
        entity_ids: torch.LongTensor = None,
        input_sentence: List[str] = None,
        **kwargs,
    ):
        feature_vector = self.feature_extractor(word_ids[self.text_field_key], entity_span, entity_ids)
        feature_vector = self.dropout(feature_vector)
        logits = self.classifier(feature_vector)

        output_dict = {"input": input_sentence, "prediction": torch.softmax(logits, dim=-1)}

        if labels is not None:
            output_dict["loss"] = self.criterion(logits, labels.float())
            output_dict["gold_label"] = labels
            self.f1_score(logits, labels)
        return output_dict

    def make_output_human_readable(self, output_dict: Dict[str, torch.Tensor]):
        output_dict["prediction"] = self.make_label_human_readable(output_dict["prediction"])

        if "gold_label" in output_dict:
            output_dict["gold_label"] = self.make_label_human_readable(output_dict["gold_label"])
        return output_dict

    def make_label_human_readable(self, labels: torch.Tensor) -> List[List[str]]:
        human_readable_labels = []
        for onehot in labels:
            indices = torch.nonzero(onehot > 0.5).squeeze(1).tolist()
            label_texts = [self.vocab.get_token_from_index(i, namespace=self.label_name_space) for i in indices]
            human_readable_labels.append(label_texts)
        return human_readable_labels

    def get_metrics(self, reset: bool = False):
        output_dict = {k: metric.get_metric(reset=reset) for k, metric in self.metrics.items()}
        output_dict.update({"micro_" + k: v for k, v in self.f1_score.get_metric(reset).items()})
        return output_dict
