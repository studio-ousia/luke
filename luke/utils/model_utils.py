import json
import os
import tarfile
import tempfile
from typing import Dict

import click
import torch
from transformers import AutoTokenizer, BertTokenizer, RobertaTokenizer, PreTrainedTokenizer
from transformers.modeling_bert import BertConfig, BertForPreTraining, BertPreTrainedModel
from transformers.modeling_roberta import RobertaConfig, RobertaForMaskedLM
from transformers.modeling_xlm_roberta import XLMRobertaConfig, XLMRobertaForMaskedLM

from luke.model import LukeConfig
from .entity_vocab import EntityVocab
from .word_tokenizer import XLMRobertaTokenizer

MODEL_FILE = "pytorch_model.bin"
METADATA_FILE = "metadata.json"
ENTITY_VOCAB_FILE = "entity_vocab.tsv"
MULTILINGUAL_ENTITY_VOCAB_FILE = "multilingual_entity_vocab.json"


@click.command()
@click.argument("model_file", type=click.Path())
@click.argument("out_file", type=click.Path())
@click.option("--compress", type=click.Choice(["", "gz", "bz2", "xz"]), default="")
def create_model_archive(model_file: str, out_file: str, compress: str):
    model_dir = os.path.dirname(model_file)
    json_file = os.path.join(model_dir, METADATA_FILE)
    with open(json_file) as f:
        model_data = json.load(f)
        del model_data["arguments"]

    file_ext = ".tar" if not compress else ".tar." + compress
    if not out_file.endswith(file_ext):
        out_file = out_file + file_ext

    with tarfile.open(out_file, mode="w:" + compress) as archive_file:
        archive_file.add(model_file, arcname=MODEL_FILE)

        entity_vocab_file_path = os.path.join(model_dir, ENTITY_VOCAB_FILE)
        if os.path.exists(entity_vocab_file_path):
            archive_file.add(entity_vocab_file_path, arcname=ENTITY_VOCAB_FILE)
        else:
            archive_file.add(
                os.path.join(model_dir, MULTILINGUAL_ENTITY_VOCAB_FILE), arcname=MULTILINGUAL_ENTITY_VOCAB_FILE
            )

        with tempfile.NamedTemporaryFile(mode="w") as metadata_file:
            json.dump(model_data, metadata_file, indent=2)
            metadata_file.flush()
            os.fsync(metadata_file.fileno())
            archive_file.add(metadata_file.name, arcname=METADATA_FILE)


class ModelArchive(object):
    def __init__(self, state_dict: Dict[str, torch.Tensor], metadata: dict, entity_vocab: EntityVocab):
        self.state_dict = state_dict
        self.metadata = metadata
        self.entity_vocab = entity_vocab

    @property
    def bert_model_name(self):
        return self.metadata["model_config"]["bert_model_name"]

    @property
    def config(self):
        return LukeConfig(**self.metadata["model_config"])

    @property
    def tokenizer(self):
        return AutoTokenizer.from_pretrained(self.bert_model_name)

    @property
    def max_seq_length(self):
        return self.metadata["max_seq_length"]

    @property
    def max_mention_length(self):
        return self.metadata["max_mention_length"]

    @property
    def max_entity_length(self):
        return self.metadata["max_entity_length"]

    @classmethod
    def load(cls, archive_path: str):
        if os.path.isdir(archive_path):
            return cls._load(archive_path, MODEL_FILE)
        elif archive_path.endswith(".bin"):
            return cls._load(os.path.dirname(archive_path), os.path.basename(archive_path))

        with tempfile.TemporaryDirectory() as temp_path:
            f = tarfile.open(archive_path)
            f.extractall(temp_path)
            return cls._load(temp_path, MODEL_FILE)

    @staticmethod
    def _load(path: str, model_file: str):
        state_dict = torch.load(os.path.join(path, model_file), map_location="cpu")
        with open(os.path.join(path, METADATA_FILE)) as metadata_file:
            metadata = json.load(metadata_file)
        entity_vocab = EntityVocab(os.path.join(path, ENTITY_VOCAB_FILE))

        return ModelArchive(state_dict, metadata, entity_vocab)


def get_pretrained_model(mode_name: str) -> BertPreTrainedModel:
    if "xlm-roberta" in mode_name:
        return XLMRobertaForMaskedLM.from_pretrained(mode_name)
    elif "roberta" in mode_name:
        return RobertaForMaskedLM.from_pretrained(mode_name)
    elif "bert" in mode_name:
        return BertForPreTraining.from_pretrained(mode_name)
    else:
        raise NotImplementedError


def get_tokenizer(tokenizer_name: str) -> PreTrainedTokenizer:
    if "xlm-roberta" in tokenizer_name:
        return XLMRobertaTokenizer.from_pretrained(tokenizer_name)
    elif "roberta" in tokenizer_name:
        return RobertaTokenizer.from_pretrained(tokenizer_name)
    elif "bert" in tokenizer_name:
        return BertTokenizer.from_pretrained(tokenizer_name)
    else:
        raise NotImplementedError


def get_config(config_name: str):
    if "xlm-roberta" in config_name:
        return XLMRobertaConfig.from_pretrained(config_name)
    elif "roberta" in config_name:
        return RobertaConfig.from_pretrained(config_name)
    elif "bert" in config_name:
        return BertConfig.from_pretrained(config_name)
    else:
        raise NotImplementedError
