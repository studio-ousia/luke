import json
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModel

from luke.model import LukeConfig, LukeModel


def load_model_from_config(serialization_directory: str, weight_file: str = None):
    serialization_directory = Path(serialization_directory)
    metadata_path = serialization_directory / "metadata.json"
    model_config = json.load(open(metadata_path, "r"))["model_config"]

    bert_config = AutoConfig.from_pretrained(model_config["bert_model_name"])

    config = LukeConfig(
        entity_vocab_size=model_config["entity_vocab_size"],
        bert_model_name=model_config["bert_model_name"],
        entity_emb_size=model_config["entity_emb_size"],
        **bert_config.to_dict(),
    )

    model = LukeModel(config)

    if weight_file is not None:
        model_state_dict = torch.load(weight_file, map_location="cpu")
        model.load_state_dict(model_state_dict, strict=False)
    else:
        bert_model = AutoModel.from_pretrained(model_config["bert_model_name"])
        bert_state_dict = bert_model.state_dict()
        model.load_bert_weights(bert_state_dict)

    return model
