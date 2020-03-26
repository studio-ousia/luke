import functools
import math

import click
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_bert import BertIntermediate, BertOutput, BertSelfOutput

from luke.model import LukeModel


def word_entity_model_args(func):
    @click.option('--word-entity-query', is_flag=True)
    @click.option('--word-entity-key', is_flag=True)
    @click.option('--word-entity-value', is_flag=True)
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


class LukeWordEntityAttentionModel(LukeModel):
    def __init__(self, args):
        super(LukeWordEntityAttentionModel, self).__init__(args.model_config)
        self.encoder = WordEntityEncoder(args)
        self.args = args

    def forward(self, word_ids, word_segment_ids, word_attention_mask, entity_ids, entity_position_ids,
                entity_segment_ids, entity_attention_mask):
        word_embeddings = self.embeddings(word_ids, word_segment_ids)
        entity_embeddings = self.entity_embeddings(entity_ids, entity_position_ids, entity_segment_ids)
        attention_mask = self._compute_extended_attention_mask(word_attention_mask, entity_attention_mask)

        return self.encoder(word_embeddings, entity_embeddings, attention_mask)

    def load_state_dict(self, state_dict, *args, **kwargs):
        new_state_dict = state_dict.copy()

        for num in range(self.args.model_config.num_hidden_layers):
            for mat_name in ('query', 'key', 'value'):
                for attr_name in ('weight', 'bias'):
                    if f'encoder.layer.{num}.attention.self.w2e_{mat_name}.{attr_name}' not in state_dict:
                        new_state_dict[f'encoder.layer.{num}.attention.self.w2e_{mat_name}.{attr_name}'] =\
                            state_dict[f'encoder.layer.{num}.attention.self.{mat_name}.{attr_name}']
                    if f'encoder.layer.{num}.attention.self.e2w_{mat_name}.{attr_name}' not in state_dict:
                        new_state_dict[f'encoder.layer.{num}.attention.self.e2w_{mat_name}.{attr_name}'] =\
                            state_dict[f'encoder.layer.{num}.attention.self.{mat_name}.{attr_name}']
                    if f'encoder.layer.{num}.attention.self.e2e_{mat_name}.{attr_name}' not in state_dict:
                        new_state_dict[f'encoder.layer.{num}.attention.self.e2e_{mat_name}.{attr_name}'] =\
                            state_dict[f'encoder.layer.{num}.attention.self.{mat_name}.{attr_name}']

        kwargs['strict'] = False
        super(LukeWordEntityAttentionModel, self).load_state_dict(new_state_dict, *args, **kwargs)


class WordEntitySelfAttention(nn.Module):
    def __init__(self, args):
        super(WordEntitySelfAttention, self).__init__()
        self.args = args

        config = args.model_config
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        args.word_entity_attention = (args.word_entity_query or args.word_entity_key or args.word_entity_value)

        if args.word_entity_attention:
            if args.word_entity_query:
                self.w2e_query = nn.Linear(config.hidden_size, self.all_head_size)
                self.e2w_query = nn.Linear(config.hidden_size, self.all_head_size)
                self.e2e_query = nn.Linear(config.hidden_size, self.all_head_size)

            if args.word_entity_key:
                self.w2e_key = nn.Linear(config.hidden_size, self.all_head_size)
                self.e2w_key = nn.Linear(config.hidden_size, self.all_head_size)
                self.e2e_key = nn.Linear(config.hidden_size, self.all_head_size)

            if args.word_entity_value:
                self.w2e_value = nn.Linear(config.hidden_size, self.all_head_size)
                self.e2w_value = nn.Linear(config.hidden_size, self.all_head_size)
                self.e2e_value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        # batch, num_attention_heads, seq_length, attention_head_size
        return x.permute(0, 2, 1, 3)

    def forward(self, word_hidden_states, entity_hidden_states, attention_mask):
        word_size = word_hidden_states.size(1)

        if self.args.word_entity_attention and self.args.word_entity_query:
            # batch_size, num_heads, word_seq_size, head_size
            w2w_query_layer = self.transpose_for_scores(self.query(word_hidden_states))
            w2e_query_layer = self.transpose_for_scores(self.w2e_query(word_hidden_states))
            # batch_size, num_heads, entity_seq_size, head_size
            e2w_query_layer = self.transpose_for_scores(self.e2w_query(entity_hidden_states))
            e2e_query_layer = self.transpose_for_scores(self.e2e_query(entity_hidden_states))
        else:
            query_layer = self.transpose_for_scores(
                self.query(torch.cat([word_hidden_states, entity_hidden_states], dim=1)))
            if self.args.word_entity_attention:
                # batch_size, num_heads, word_seq_size, head_size
                w2w_query_layer = query_layer[:, :, :word_size, :]
                w2e_query_layer = query_layer[:, :, :word_size, :]
                # batch_size, num_heads, entity_seq_size, head_size
                e2w_query_layer = query_layer[:, :, word_size:, :]
                e2e_query_layer = query_layer[:, :, word_size:, :]

        if self.args.word_entity_attention and self.args.word_entity_key:
            # batch_size, num_heads, word_seq_size, head_size
            w2w_key_layer = self.transpose_for_scores(self.key(word_hidden_states))
            e2w_key_layer = self.transpose_for_scores(self.e2w_key(word_hidden_states))
            # batch_size, num_heads, entity_seq_size, head_size
            w2e_key_layer = self.transpose_for_scores(self.w2e_key(entity_hidden_states))
            e2e_key_layer = self.transpose_for_scores(self.e2e_key(entity_hidden_states))
        else:
            key_layer = self.transpose_for_scores(
                self.key(torch.cat([word_hidden_states, entity_hidden_states], dim=1)))
            if self.args.word_entity_attention:
                # batch_size, num_heads, word_seq_size, head_size
                w2w_key_layer = key_layer[:, :, :word_size, :]
                e2w_key_layer = key_layer[:, :, :word_size, :]
                # batch_size, num_heads, entity_seq_size, head_size
                w2e_key_layer = key_layer[:, :, word_size:, :]
                e2e_key_layer = key_layer[:, :, word_size:, :]

        if self.args.word_entity_attention:
            # batch_size, num_heads, word_seq_size, word_seq_size
            w2w_attention_scores = torch.matmul(w2w_query_layer, w2w_key_layer.transpose(-1, -2))
            # batch_size, num_heads, word_seq_size, entity_seq_size
            w2e_attention_scores = torch.matmul(w2e_query_layer, w2e_key_layer.transpose(-1, -2))
            # batch_size, num_heads, entity_seq_size, word_seq_size
            e2w_attention_scores = torch.matmul(e2w_query_layer, e2w_key_layer.transpose(-1, -2))
            # batch_size, num_heads, entity_seq_size, entity_seq_size
            e2e_attention_scores = torch.matmul(e2e_query_layer, e2e_key_layer.transpose(-1, -2))

            # batch_size, num_heads, word_seq_size, word_seq_size + entity_seq_size
            word_attention_scores = torch.cat([w2w_attention_scores, w2e_attention_scores], dim=3)
            # batch_size, num_heads, entity_seq_size, word_seq_size + entity_seq_size
            entity_attention_scores = torch.cat([e2w_attention_scores, e2e_attention_scores], dim=3)
            # batch_size, num_heads, word_seq_size + entity_seq_size, word_seq_size + entity_seq_size
            attention_scores = torch.cat([word_attention_scores, entity_attention_scores], dim=2)
        else:
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores + attention_mask

        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        if self.args.word_entity_attention and self.args.word_entity_value:
            w2w_value_layer = self.value(word_hidden_states)
            w2e_value_layer = self.w2e_value(entity_hidden_states)
            # batch_size, num_heads, word_seq_size + entity_seq_size, head_size
            word_value_layer = self.transpose_for_scores(torch.cat([w2w_value_layer, w2e_value_layer], dim=1))
            e2w_value_layer = self.e2w_value(word_hidden_states)
            e2e_value_layer = self.e2e_value(entity_hidden_states)
            # batch_size, num_heads, word_seq_size + entity_seq_size, head_size
            entity_value_layer = self.transpose_for_scores(torch.cat([e2w_value_layer, e2e_value_layer], dim=1))

            word_context_layer = torch.matmul(attention_probs[:, :, :word_size, :], word_value_layer)
            entity_context_layer = torch.matmul(attention_probs[:, :, word_size:, :], entity_value_layer)
            context_layer = torch.cat([word_context_layer, entity_context_layer], dim=2)
        else:
            value_layer = self.transpose_for_scores(self.value(
                torch.cat([word_hidden_states, entity_hidden_states], dim=1)))
            context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer[:, :word_size, :], context_layer[:, word_size:, :]


class WordEntityAttention(nn.Module):
    def __init__(self, args):
        super(WordEntityAttention, self).__init__()
        self.self = WordEntitySelfAttention(args)
        self.output = BertSelfOutput(args.model_config)

    def forward(self, word_hidden_states, entity_hidden_states, attention_mask):
        word_self_output, entity_self_output = self.self(word_hidden_states, entity_hidden_states, attention_mask)
        hidden_states = torch.cat([word_hidden_states, entity_hidden_states], dim=1)
        self_output = torch.cat([word_self_output, entity_self_output], dim=1)
        output = self.output(self_output, hidden_states)
        return output[:, :word_hidden_states.size(1), :], output[:, word_hidden_states.size(1):, :]


class WordEntityLayer(nn.Module):
    def __init__(self, args):
        super(WordEntityLayer, self).__init__()
        self.args = args

        self.attention = WordEntityAttention(args)
        self.intermediate = BertIntermediate(args.model_config)
        self.output = BertOutput(args.model_config)

    def forward(self, word_hidden_states, entity_hidden_states, attention_mask):
        word_attention_output, entity_attention_output = self.attention(word_hidden_states, entity_hidden_states,
                                                                        attention_mask)
        attention_output = torch.cat([word_attention_output, entity_attention_output], dim=1)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)

        return layer_output[:, :word_hidden_states.size(1), :], layer_output[:, word_hidden_states.size(1):, :]


class WordEntityEncoder(nn.Module):
    def __init__(self, args):
        super(WordEntityEncoder, self).__init__()
        self.layer = nn.ModuleList([WordEntityLayer(args) for _ in range(args.model_config.num_hidden_layers)])

    def forward(self, word_hidden_states, entity_hidden_states, attention_mask):
        for layer_module in self.layer:
            word_hidden_states, entity_hidden_states = layer_module(word_hidden_states, entity_hidden_states,
                                                                    attention_mask)
        return word_hidden_states, entity_hidden_states
