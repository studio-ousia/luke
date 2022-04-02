import functools
import operator

import pytest
import torch
from transformers import AutoConfig, AutoModel

from luke.model import EntityEmbeddings, LukeConfig, LukeModel

BERT_MODEL_NAME = "bert-base-uncased"


@pytest.fixture
def bert_config():
    bert_config = AutoConfig.from_pretrained(BERT_MODEL_NAME)
    bert_config.hidden_dropout_prob = 0.0
    return bert_config


def _create_luke_config(bert_config, entity_vocab_size, entity_emb_size):
    return LukeConfig(
        entity_vocab_size=entity_vocab_size,
        bert_model_name=BERT_MODEL_NAME,
        entity_emb_size=entity_emb_size,
        **bert_config.to_dict()
    )


def test_entity_embedding(bert_config):
    config = _create_luke_config(bert_config, 5, bert_config.hidden_size)
    entity_embeddings = EntityEmbeddings(config)
    entity_ids = torch.LongTensor([2, 3, 0])
    position_ids = torch.LongTensor(
        [
            [0, 1] + [-1] * (config.max_position_embeddings - 2),
            [3] + [-1] * (config.max_position_embeddings - 1),
            [-1] * config.max_position_embeddings,
        ]
    )
    token_type_ids = torch.LongTensor([0, 1, 0])

    emb = entity_embeddings(entity_ids, position_ids, token_type_ids)
    assert emb.size() == (3, config.hidden_size)

    for n, (entity_id, position_id_list, token_type_id) in enumerate(zip(entity_ids, position_ids, token_type_ids)):
        entity_emb = entity_embeddings.entity_embeddings.weight[entity_id]
        token_type_emb = entity_embeddings.token_type_embeddings.weight[token_type_id]
        position_emb_list = [entity_embeddings.position_embeddings.weight[p] for p in position_id_list if p != -1]
        if position_emb_list:
            position_emb = functools.reduce(operator.add, position_emb_list) / len(position_emb_list)
            target_emb = entity_embeddings.LayerNorm((entity_emb + position_emb + token_type_emb))
        else:
            target_emb = entity_embeddings.LayerNorm((entity_emb + token_type_emb))

        assert torch.equal(emb[n], target_emb)


def test_load_bert_weights(bert_config):
    bert_model = AutoModel.from_pretrained(BERT_MODEL_NAME)
    bert_state_dict = bert_model.state_dict()

    config = _create_luke_config(bert_config, 5, bert_config.hidden_size)
    model = LukeModel(config)
    model.load_bert_weights(bert_state_dict)
    luke_state_dict = model.state_dict()

    for key, tensor in bert_state_dict.items():
        assert torch.equal(luke_state_dict[key], tensor)
