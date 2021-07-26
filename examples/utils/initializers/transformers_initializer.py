from typing import Dict
import torch
from allennlp.nn.initializers import Initializer, PretrainedModelInitializer

from transformers import AutoModel


@Initializer.register("transformers")
class TransformersPretrainedModelInitializer(PretrainedModelInitializer):
    """
    An initializer which allows initializing parameters using a pretrained model in in Hugging Face Model Hub.
    """

    def __init__(self, transformers_model_name: str, parameter_name_overrides: Dict[str, str] = None) -> None:
        model = AutoModel.from_pretrained(transformers_model_name)
        self.weights = model.named_parameters
        self.parameter_name_overrides = parameter_name_overrides or {}
