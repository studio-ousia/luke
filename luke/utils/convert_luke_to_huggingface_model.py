from typing import Union, Tuple, TypeVar
import json
import os
from collections import OrderedDict

import click
import torch
from transformers import LukeConfig, LukeForMaskedLM, AutoTokenizer, LukeTokenizer, MLukeTokenizer
from transformers.tokenization_utils_base import AddedToken

LukeTokenizerType = TypeVar("LukeTokenizerType", bound=Union[LukeTokenizer, MLukeTokenizer])


def remove_entity_embeddings_from_luke(
    model: LukeForMaskedLM, tokenizer: LukeTokenizerType
) -> Tuple[LukeForMaskedLM, LukeTokenizerType]:
    """
    This script modifies the Luke model into one without entity embeddings.
    """

    if model.entity_predictions.decoder.bias:
        raise ValueError(
            "The bias paramter should not present at 'model.entity_predictions.decoder.bias'."
            "We assume the bias weight is at 'model.entity_predictions.bias'."
        )

    if not (model.entity_predictions.decoder.weight == model.luke.entity_embeddings.entity_embeddings.weight).all():
        raise ValueError("Currently this script only supports model with tied entity embeddings.")

    entity_vocab = tokenizer.entity_vocab
    old_special_token_indices = [
        entity_vocab["[MASK]"],
        entity_vocab["[UNK]"],
        entity_vocab["[PAD]"],
        entity_vocab["[MASK2]"],
    ]

    new_entity_embeddings = model.luke.entity_embeddings.entity_embeddings.weight.data[old_special_token_indices]
    new_entity_prediction_bias = model.entity_predictions.bias.data[old_special_token_indices]

    new_entity_vocab = {"[MASK]": 0, "[UNK]": 1, "[PAD]": 2, "[MASK2]": 3}

    model.luke.entity_embeddings.entity_embeddings.weight.data = new_entity_embeddings
    model.luke.entity_embeddings.entity_embeddings.num_embeddings = len(new_entity_vocab)

    model.entity_predictions.decoder.weight.data = new_entity_embeddings
    model.entity_predictions.decoder.out_features = len(new_entity_vocab)

    model.entity_predictions.bias.data = new_entity_prediction_bias

    tokenizer.entity_vocab = new_entity_vocab

    model.config.entity_vocab_size = len(new_entity_vocab)
    return model, tokenizer


@click.command()
@click.option(
    "--checkpoint-path", type=click.Path(exists=True), help="Path to a pytorch_model.bin file.", required=True
)
@click.option(
    "--metadata-path",
    type=click.Path(exists=True),
    help="Path to a metadata.json file, defining the configuration.",
    required=True,
)
@click.option(
    "--entity-vocab-path",
    type=click.Path(exists=True),
    help="Path to an entity_vocab.jsonl file, containing the entity vocabulary.",
    required=True,
)
@click.option(
    "--transformers-model-save-path",
    type=click.Path(),
    help="Path to where to dump the output PyTorch model.",
    required=True,
)
@click.option(
    "--tokenizer-class",
    type=click.Choice(["LukeTokenizer", "MLukeTokenizer"]),
    help="The Tokenizer class to use in transformers.",
    required=True,
)
@click.option(
    "--set-entity-aware-attention-default",
    type=bool,
    help="If true, use_entity_aware_attention is set to true in the model config.",
    required=True,
)
@click.option(
    "--remove-entity-embeddings",
    is_flag=True,
    help="If true, the entity embeddings will be removed to make a lite-weight model.",
)
def convert_luke_to_huggingface_model(
    checkpoint_path: str,
    metadata_path: str,
    entity_vocab_path: str,
    transformers_model_save_path: str,
    tokenizer_class: str,
    set_entity_aware_attention_default: bool,
    remove_entity_embeddings: bool,
):
    # Load configuration defined in the metadata file
    with open(metadata_path) as metadata_file:
        metadata = json.load(metadata_file)
    config = LukeConfig(use_entity_aware_attention=True, **metadata["model_config"])

    # Load in the weights from the checkpoint_path
    state_dict = torch.load(checkpoint_path, map_location="cpu")["module"]

    # Load the entity vocab file
    entity_vocab = load_original_entity_vocab(entity_vocab_path)
    # add an entry for [MASK2]
    entity_vocab["[MASK2]"] = max(entity_vocab.values()) + 1
    config.entity_vocab_size += 1

    tokenizer = AutoTokenizer.from_pretrained(metadata["model_config"]["bert_model_name"])

    # Add special tokens to the token vocabulary for downstream tasks
    entity_token_1 = AddedToken("<ent>", lstrip=False, rstrip=False)
    entity_token_2 = AddedToken("<ent2>", lstrip=False, rstrip=False)
    tokenizer.add_special_tokens(dict(additional_special_tokens=[entity_token_1, entity_token_2]))
    config.vocab_size += 2

    print(f"Saving tokenizer to {transformers_model_save_path}")
    tokenizer.save_pretrained(transformers_model_save_path)
    with open(os.path.join(transformers_model_save_path, "tokenizer_config.json"), "r") as f:
        tokenizer_config = json.load(f)
    tokenizer_config["tokenizer_class"] = tokenizer_class
    with open(os.path.join(transformers_model_save_path, "tokenizer_config.json"), "w") as f:
        json.dump(tokenizer_config, f)

    with open(os.path.join(transformers_model_save_path, "entity_vocab.json"), "w") as f:
        json.dump(entity_vocab, f)

    tokenizer = AutoTokenizer.from_pretrained(transformers_model_save_path)

    # Initialize the embeddings of the special tokens
    ent_init_index = tokenizer.convert_tokens_to_ids(["@"])[0]
    ent2_init_index = tokenizer.convert_tokens_to_ids(["#"])[0]

    word_emb = state_dict["embeddings.word_embeddings.weight"]
    ent_emb = word_emb[ent_init_index].unsqueeze(0)
    ent2_emb = word_emb[ent2_init_index].unsqueeze(0)
    state_dict["embeddings.word_embeddings.weight"] = torch.cat([word_emb, ent_emb, ent2_emb])
    # add special tokens for 'entity_predictions.bias'
    for bias_name in ["lm_head.decoder.bias", "lm_head.bias"]:
        decoder_bias = state_dict[bias_name]
        ent_decoder_bias = decoder_bias[ent_init_index].unsqueeze(0)
        ent2_decoder_bias = decoder_bias[ent2_init_index].unsqueeze(0)
        state_dict[bias_name] = torch.cat([decoder_bias, ent_decoder_bias, ent2_decoder_bias])

    # If the model is pretrained without the entity-aware self-attention mechanism,
    # the normal attention weights are copied to the ones for entity aware attention
    # so that you can use them during fine-tuning
    for layer_index in range(config.num_hidden_layers):
        for matrix_name in ["query.weight", "query.bias"]:
            prefix = f"encoder.layer.{layer_index}.attention.self."
            for infix in ["w2e_", "e2w_", "e2e_"]:
                entity_aware_weight_name = prefix + infix + matrix_name
                if entity_aware_weight_name not in state_dict:
                    state_dict[entity_aware_weight_name] = state_dict[prefix + matrix_name]

    # Initialize the embedding of the [MASK2] entity using that of the [MASK] entity for downstream tasks
    entity_emb = state_dict["entity_embeddings.entity_embeddings.weight"]
    entity_mask_emb = entity_emb[entity_vocab["[MASK]"]].unsqueeze(0)
    state_dict["entity_embeddings.entity_embeddings.weight"] = torch.cat([entity_emb, entity_mask_emb])
    # add [MASK2] for 'entity_predictions.bias'
    entity_prediction_bias = state_dict["entity_predictions.bias"]
    entity_mask_bias = entity_prediction_bias[entity_vocab["[MASK]"]].unsqueeze(0)
    state_dict["entity_predictions.bias"] = torch.cat([entity_prediction_bias, entity_mask_bias])

    model = LukeForMaskedLM(config=config).eval()

    state_dict.pop("entity_predictions.decoder.weight")
    state_dict.pop("lm_head.decoder.weight")
    state_dict.pop("lm_head.decoder.bias")
    state_dict_for_hugging_face = OrderedDict()
    for key, value in state_dict.items():
        if not (key.startswith("lm_head") or key.startswith("entity_predictions")):
            state_dict_for_hugging_face[f"luke.{key}"] = state_dict[key]
        else:
            state_dict_for_hugging_face[key] = state_dict[key]

    missing_keys, unexpected_keys = model.load_state_dict(state_dict_for_hugging_face, strict=False)

    if set(unexpected_keys) != {"luke.embeddings.position_ids"}:
        raise ValueError(f"Unexpected unexpected_keys: {unexpected_keys}")
    if set(missing_keys) != {
        "lm_head.decoder.weight",
        "lm_head.decoder.bias",
        "entity_predictions.decoder.weight",
    }:
        raise ValueError(f"Unexpected missing_keys: {missing_keys}")

    model.tie_weights()
    assert (model.luke.embeddings.word_embeddings.weight == model.lm_head.decoder.weight).all()
    assert (model.luke.entity_embeddings.entity_embeddings.weight == model.entity_predictions.decoder.weight).all()

    if remove_entity_embeddings:
        model, tokenizer = remove_entity_embeddings_from_luke(model, tokenizer)

    # Finally, save our PyTorch model and tokenizer
    print(f"Saving PyTorch model to {transformers_model_save_path}")
    model.config.use_entity_aware_attention = set_entity_aware_attention_default
    model.save_pretrained(transformers_model_save_path)
    tokenizer.save_pretrained(transformers_model_save_path)


def load_original_entity_vocab(entity_vocab_path):
    SPECIAL_TOKENS = ["[MASK]", "[PAD]", "[UNK]"]

    data = [json.loads(line) for line in open(entity_vocab_path)]

    new_mapping = {}
    for entry in data:
        entity_id = entry["id"]
        for entity_name, language in entry["entities"]:
            if entity_name in SPECIAL_TOKENS:
                new_mapping[entity_name] = entity_id
                break
            new_entity_name = f"{language}:{entity_name}"
            new_mapping[new_entity_name] = entity_id
    return new_mapping


if __name__ == "__main__":
    convert_luke_to_huggingface_model()
