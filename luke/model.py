import logging

import torch
from torch import nn
from transformers.modeling_bert import BertConfig, BertEmbeddings, BertEncoder, BertLayerNorm, BertPooler
from transformers.modeling_roberta import RobertaEmbeddings

logger = logging.getLogger(__name__)

EPS = 1e-7


class LukeConfig(BertConfig):
    def __init__(self,
                 vocab_size: int,
                 entity_vocab_size: int,
                 bert_model_name: str,
                 entity_emb_size: int = None,
                 **kwargs):
        super(LukeConfig, self).__init__(vocab_size, **kwargs)

        self.entity_vocab_size = entity_vocab_size
        self.bert_model_name = bert_model_name
        self.entity_emb_size = entity_emb_size


class EntityEmbeddings(nn.Module):
    def __init__(self,
                 config: LukeConfig,
                 entity_vocab_size: int = None):
        super(EntityEmbeddings, self).__init__()
        self.config = config
        if entity_vocab_size is None:
            entity_vocab_size = config.entity_vocab_size

        if config.entity_emb_size is None:
            self.entity_embeddings = nn.Embedding(entity_vocab_size, config.hidden_size, padding_idx=0)
        else:
            self.entity_embeddings = nn.Embedding(entity_vocab_size, config.entity_emb_size, padding_idx=0)
            if config.entity_emb_size != config.hidden_size:
                self.entity_embedding_dense = nn.Linear(config.entity_emb_size, config.hidden_size, bias=False)

        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self,
                entity_ids: torch.LongTensor,
                position_ids: torch.LongTensor,
                token_type_ids: torch.LongTensor = None):
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(entity_ids)

        entity_embeddings = self.entity_embeddings(entity_ids)
        if self.config.entity_emb_size is not None and self.config.entity_emb_size != self.config.hidden_size:
            entity_embeddings = self.entity_embedding_dense(entity_embeddings)

        position_embeddings = self.position_embeddings(position_ids.clamp(min=0))
        position_embedding_mask = (position_ids != -1).type_as(position_embeddings).unsqueeze(-1)
        position_embeddings = position_embeddings * position_embedding_mask
        position_embeddings = torch.sum(position_embeddings, dim=-2)
        position_embeddings = position_embeddings / position_embedding_mask.sum(dim=-2).clamp(min=EPS)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = entity_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class LukeModel(nn.Module):
    def __init__(self, config: LukeConfig):
        super(LukeModel, self).__init__()

        self.config = config

        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        if self.config.bert_model_name and self.config.bert_model_name.startswith('roberta'):
            self.embeddings = RobertaEmbeddings(config)
            self.embeddings.token_type_embeddings.requires_grad = False
        else:
            self.embeddings = BertEmbeddings(config)
        self.entity_embeddings = EntityEmbeddings(config)

    def forward(self,
                word_ids: torch.LongTensor,
                word_segment_ids: torch.LongTensor,
                word_attention_mask: torch.LongTensor,
                entity_ids: torch.LongTensor,
                entity_position_ids: torch.LongTensor,
                entity_segment_ids: torch.LongTensor,
                entity_attention_mask: torch.LongTensor):
        word_seq_size = word_ids.size(1)
        extended_attention_mask = self._compute_extended_attention_mask(word_attention_mask, entity_attention_mask)

        word_embedding_output = self.embeddings(word_ids, word_segment_ids)
        entity_embedding_output = self.entity_embeddings(entity_ids, entity_position_ids, entity_segment_ids)
        embedding_output = torch.cat([word_embedding_output, entity_embedding_output], dim=1)
        encoder_outputs = self.encoder(embedding_output, extended_attention_mask,
                                       [None] * self.config.num_hidden_layers)
        sequence_output = encoder_outputs[0]
        word_sequence_output = sequence_output[:, :word_seq_size, :]
        entity_sequence_output = sequence_output[:, word_seq_size:, :]
        pooled_output = self.pooler(sequence_output)

        return (word_sequence_output, entity_sequence_output, pooled_output,) + encoder_outputs[1:]

    def init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.Embedding):
            if module.embedding_dim == 1:  # embedding for bias parameters
                module.weight.data.zero_()
            else:
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def load_bert_weights(self, state_dict):
        state_dict = state_dict.copy()
        for key in list(state_dict.keys()):
            new_key = key.replace('gamma', 'weight').replace('beta', 'bias')
            if new_key.startswith('roberta.'):
                new_key = new_key[8:]
            elif new_key.startswith('bert.'):
                new_key = new_key[5:]

            if key != new_key:
                state_dict[new_key] = state_dict[key]
                del state_dict[key]

        missing_keys = []
        unexpected_keys = []
        error_msgs = []

        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys,
                                         error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        load(self, prefix='')
        if len(unexpected_keys) > 0:
            logger.info("Weights from pretrained model not used in {}: {}".format(self.__class__.__name__,
                                                                                  sorted(unexpected_keys)))
        if len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(self.__class__.__name__,
                                                                                     "\n\t".join(error_msgs)))

    def _compute_extended_attention_mask(self,
                                         word_attention_mask: torch.LongTensor,
                                         entity_attention_mask: torch.LongTensor):
        attention_mask = torch.cat([word_attention_mask, entity_attention_mask], dim=1)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        return extended_attention_mask
