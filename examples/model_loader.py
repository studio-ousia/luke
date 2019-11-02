import os
import json

import torch
from transformers import AutoTokenizer

from luke.model import LukeConfig
from luke.utils.entity_vocab import EntityVocab
from luke.pretraining.dataset import ENTITY_VOCAB_FILE, METADATA_FILE


class LukeModelLoader(object):
    def __init__(self, model_config, state_dict, tokenizer, entity_vocab, max_seq_length, max_entity_length,
                 max_mention_length):
        self.model_config = model_config
        self.state_dict = state_dict
        self.tokenizer = tokenizer
        self.entity_vocab = entity_vocab
        self.max_seq_length = max_seq_length
        self.max_entity_length = max_entity_length
        self.max_mention_length = max_mention_length

    @staticmethod
    def load(model_file_or_dir):
        if os.path.isfile(model_file_or_dir):
            model_dir = os.path.dirname(model_file_or_dir)
            state_dict = torch.load(model_file_or_dir, map_location='cpu')
        else:
            model_dir = model_file_or_dir
            state_dict = None

        json_file = os.path.join(model_dir, METADATA_FILE)
        with open(json_file) as f:
            model_data = json.load(f)

        if model_data['model_config'].get('entity_emb_size') is None:
            model_data['model_config']['entity_emb_size'] = model_data['model_config']['hidden_size']

        return LukeModelLoader(
            model_config=LukeConfig(**model_data['model_config']),
            state_dict=state_dict,
            tokenizer=AutoTokenizer.from_pretrained(model_data['model_config']['bert_model_name']),
            entity_vocab=EntityVocab(os.path.join(model_dir, ENTITY_VOCAB_FILE)),
            max_seq_length=model_data['max_seq_length'],
            max_entity_length=model_data['max_entity_length'],
            max_mention_length=model_data['max_mention_length'],
        )
