import json
import logging
from pathlib import Path

import click
import torch
from transformers import AutoConfig, AutoTokenizer
from transformers.models.luke.modeling_luke import LukeForEntitySpanClassification

logging.basicConfig(format="[%(asctime)s] [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command()
@click.argument("serialization-dir", type=click.Path(exists=True))
@click.argument("save-dir", type=click.Path())
def convert_allennlp_to_huggingface_model(serialization_dir: str, save_dir: str):
    logger.info(f"Loading allennlp data from {serialization_dir}...")
    model_weights_path = Path(serialization_dir) / "best.th"
    config_path = Path(serialization_dir) / "config.json"
    vocabulary_path = Path(serialization_dir) / "vocabulary/labels.txt"

    model_weights = torch.load(model_weights_path)
    config = json.load(open(config_path))

    # Check if the right tokenizer is used in the config
    if config["dataset_reader"]["tokenizer"]["type"] != "pretrained_transformer":
        raise ValueError("Only models that use a HuggingFace tokenizer can be converted.")
    huggingface_tokenizer_name = config["dataset_reader"]["tokenizer"]["model_name"]

    # Check if the right model is used in the config
    if config["model"]["type"] != "span_ner":
        raise ValueError("This script converts the weights of ExhaustiveNERModel (registered as `span_ner`).")
    if config["model"]["feature_extractor"]["embedder"]["type"] != "transformers-luke":
        raise ValueError(
            "Only models that use TransformersLukeEmbedder (registered as `transformers-luke`) can be converted."
        )
    huggingface_model_name = config["model"]["feature_extractor"]["embedder"]["model_name"]

    config = AutoConfig.from_pretrained(huggingface_model_name)
    tokenizer = AutoTokenizer.from_pretrained(huggingface_tokenizer_name)

    setattr(config, "classifier_bias", "classifier.bias" in model_weights)
    setattr(config, "num_labels", model_weights["classifier.weight"].size(0))
    labels = [label.strip() for label in open(vocabulary_path, "r")]
    config.id2label = {id_: label for id_, label in enumerate(labels)}
    config.label2id = {label: id_ for id_, label in enumerate(labels)}
    downstream_luke_model = LukeForEntitySpanClassification(config)

    huggingface_model_weights = {}
    for key, w in model_weights.items():
        huggingface_model_weights[key.replace("feature_extractor.embedder.luke_model", "luke")] = w

    downstream_luke_model.load_state_dict(huggingface_model_weights, strict=True)
    downstream_luke_model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    logger.info(f"Saved hugging face model in {save_dir}.")


if __name__ == "__main__":
    convert_allennlp_to_huggingface_model()
