import click
import json
import os
from collections import OrderedDict

import torch
from transformers import LukeConfig, AutoTokenizer

from luke.utils.entity_vocab import EntityVocab

from model import LukeForEntityDisambiguation


@click.command
@click.option("--checkpoint-file", type=click.Path(exists=True), required=True)
@click.option("--metadata-file", type=click.File(), required=True)
@click.option("--entity-vocab-file", type=click.Path(exists=True), required=True)
@click.option("--output-dir", type=click.Path(), required=True)
def convert_checkpoint(checkpoint_file, metadata_file, entity_vocab_file, output_dir):
    metadata = json.load(metadata_file)
    if "entity_emb_size" not in metadata["model_config"]:
        metadata["model_config"]["entity_emb_size"] = metadata["model_config"]["hidden_size"]
    if "pad_token_id" not in metadata["model_config"]:
        metadata["model_config"]["pad_token_id"] = 0

    config = LukeConfig(use_entity_aware_attention=False, **metadata["model_config"],)
    model = LukeForEntityDisambiguation(config=config).eval()

    state_dict = torch.load(checkpoint_file, map_location="cpu")
    if "module" in state_dict:
        state_dict = state_dict["module"]

    entity_vocab = EntityVocab(entity_vocab_file)

    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith("lm_head") or key.startswith("cls"):
            continue
        if key.startswith("entity_predictions"):
            new_state_dict[key] = value
        else:
            new_state_dict[f"luke.{key}"] = value
    entity_embeddings = state_dict["entity_embeddings.entity_embeddings.weight"]
    new_state_dict["luke.entity_embeddings.mask_embedding"] = entity_embeddings[entity_vocab["[MASK]"]]

    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
    assert not missing_keys or missing_keys == ["luke.embeddings.position_ids"]
    assert not unexpected_keys

    model.tie_weights()

    bert_model_name = metadata["model_config"]["bert_model_name"]
    tokenizer = AutoTokenizer.from_pretrained(bert_model_name)

    os.makedirs(output_dir, exist_ok=True)
    entity_vocab.save(os.path.join(output_dir, "entity_vocab.jsonl"))
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    convert_checkpoint()
