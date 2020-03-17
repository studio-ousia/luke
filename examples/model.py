import contextlib
import copy
import functools
import itertools
import math

import click
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_bert import BertEncoder, BertIntermediate, BertOutput, BertSelfOutput

from luke.model import LukeModel
from luke.pretraining.model import EntityPredictionHead
from luke.utils.entity_vocab import MASK_TOKEN, UNK_TOKEN


def two_stage_model_args(func):
    @click.option('--min-context-entity-prob', default=0.0)
    @click.option('--use-softmax-average', is_flag=True)
    @click.option('--entity-softmax-temp', default=0.1)
    @click.option('--update-params-in-disambi', is_flag=True)
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


class LukeTwoStageBaseModel(LukeModel):
    def __init__(self, args):
        super(LukeTwoStageBaseModel, self).__init__(args.model_config)

        self.args = args
        self.entity_mask_id = args.entity_vocab[MASK_TOKEN]
        self.entity_unk_id = args.entity_vocab[UNK_TOKEN]

        if args.use_softmax_average and args.update_params_in_disambi:
            self.ed_embeddings = self.embeddings
            self.ed_entity_embeddings = self.entity_embeddings
            self.ed_encoder = self.encoder
        else:
            self.ed_embeddings = copy.deepcopy(self.embeddings)
            self.ed_entity_embeddings = copy.deepcopy(self.entity_embeddings)
            self.ed_encoder = copy.deepcopy(self.encoder)
            for param in itertools.chain(self.ed_embeddings.parameters(), self.ed_entity_embeddings.parameters(),
                                         self.ed_encoder.parameters()):
                param.requires_grad = False

        self.entity_predictions = EntityPredictionHead(self.config)
        self.entity_predictions.decoder.weight = self.entity_embeddings.entity_embeddings.weight
        self.entity_prediction_bias = nn.Embedding(args.model_config.entity_vocab_size, 1, padding_idx=0)
        self.entity_prediction_bias.weight.data = self.entity_predictions.bias.data.view(-1, 1)

    def forward(self, word_ids, word_segment_ids, word_attention_mask, entity_candidate_ids, entity_position_ids,
                entity_segment_ids, entity_attention_mask):
        def maybe_no_grad():
            if self.args.use_softmax_average and self.args.update_params_in_disambi:
                return contextlib.ExitStack()
            else:
                return torch.no_grad()

        with maybe_no_grad():
            ed_attention_mask = self._compute_extended_attention_mask(word_attention_mask, entity_attention_mask)
            ed_word_embeddings = self.ed_embeddings(word_ids, word_segment_ids)
            ed_mask_embeddings = self.ed_entity_embeddings(entity_attention_mask * self.entity_mask_id,
                                                           entity_position_ids, entity_segment_ids)
            ed_encoder_outputs = self.ed_encoder(torch.cat([ed_word_embeddings, ed_mask_embeddings], dim=1),
                                                 ed_attention_mask, [None] * self.config.num_hidden_layers)
            ed_mask_output = ed_encoder_outputs[0][:, word_ids.size(1):, :]
            ed_mask_output = self.entity_predictions.transform(ed_mask_output)
            ed_candidate_embeddings = self.ed_entity_embeddings.entity_embeddings(entity_candidate_ids)

            ed_logits = (ed_mask_output.unsqueeze(2) * ed_candidate_embeddings).sum(-1)
            ed_bias = self.entity_prediction_bias(entity_candidate_ids).squeeze(-1)
            ed_logits = ed_logits + ed_bias

        entity_candidate_embeddings = self.entity_embeddings(
            entity_candidate_ids, entity_position_ids.unsqueeze(-2), entity_segment_ids.unsqueeze(-1))

        if self.args.use_softmax_average:
            ed_logits = ed_logits / self.args.entity_softmax_temp
            ed_logits.masked_fill_(entity_candidate_ids == 0, -10000.0)
            ed_probs = F.softmax(ed_logits, dim=-1)

            entity_embeddings = (entity_candidate_embeddings * ed_probs.unsqueeze(-1)).sum(-2)
        else:
            index_tensor = ed_logits.argmax(2, keepdim=True).unsqueeze(-1).expand(-1, -1, 1, self.config.hidden_size)
            entity_embeddings = torch.gather(entity_candidate_embeddings, 2, index_tensor).squeeze(2)

            ed_probs = F.softmax(ed_logits, dim=-1)
            mask = (ed_probs.max(2)[0] >= self.args.min_context_entity_prob).long()
            entity_attention_mask = entity_attention_mask * mask

        word_embeddings = self.embeddings(word_ids, word_segment_ids)
        attention_mask = self._compute_extended_attention_mask(word_attention_mask, entity_attention_mask)

        return self.encoder(torch.cat([word_embeddings, entity_embeddings], dim=1), attention_mask,
                            [None] * self.config.num_hidden_layers)

    def load_state_dict(self, state_dict, *args, **kwargs):
        new_state_dict = state_dict.copy()
        for key, weights in state_dict.items():
            if key.startswith('encoder.') or key.startswith('embeddings.') or key.startswith('entity_embeddings.'):
                new_state_dict['ed_' + key] = weights
        super(LukeTwoStageBaseModel, self).load_state_dict(new_state_dict, *args, **kwargs)


class LukeExtendedTwoStageBaseModel(LukeModel):
    def __init__(self, args):
        super(LukeExtendedTwoStageBaseModel, self).__init__(args.model_config)

        self.args = args
        self.entity_mask_id = args.entity_vocab[MASK_TOKEN]
        self.entity_unk_id = args.entity_vocab[UNK_TOKEN]

        self.encoder = Encoder(self.config)

        if args.use_softmax_average and args.update_params_in_disambi:
            raise NotImplementedError()

        else:
            self.ed_embeddings = copy.deepcopy(self.embeddings)
            self.ed_entity_embeddings = copy.deepcopy(self.entity_embeddings)
            self.ed_encoder = BertEncoder(self.config)
            for param in itertools.chain(self.ed_embeddings.parameters(), self.ed_entity_embeddings.parameters(),
                                         self.ed_encoder.parameters()):
                param.requires_grad = False

        self.entity_predictions = EntityPredictionHead(self.config)
        self.entity_predictions.decoder.weight = self.entity_embeddings.entity_embeddings.weight
        self.entity_prediction_bias = nn.Embedding(args.model_config.entity_vocab_size, 1, padding_idx=0)
        self.entity_prediction_bias.weight.data = self.entity_predictions.bias.data.view(-1, 1)

    def forward(self, word_ids, word_segment_ids, word_attention_mask, entity_candidate_ids, entity_position_ids,
                entity_segment_ids, entity_attention_mask):
        def maybe_no_grad():
            if self.args.use_softmax_average and self.args.update_params_in_disambi:
                return contextlib.ExitStack()
            else:
                return torch.no_grad()

        with maybe_no_grad():
            ed_attention_mask = self._compute_extended_attention_mask(word_attention_mask, entity_attention_mask)
            ed_word_embeddings = self.ed_embeddings(word_ids, word_segment_ids)
            ed_mask_embeddings = self.ed_entity_embeddings(entity_attention_mask * self.entity_mask_id,
                                                           entity_position_ids, entity_segment_ids)
            ed_encoder_outputs = self.ed_encoder(torch.cat([ed_word_embeddings, ed_mask_embeddings], dim=1),
                                                 ed_attention_mask, [None] * self.config.num_hidden_layers)
            ed_mask_output = ed_encoder_outputs[0][:, word_ids.size(1):, :]
            ed_mask_output = self.entity_predictions.transform(ed_mask_output)
            ed_candidate_embeddings = self.ed_entity_embeddings.entity_embeddings(entity_candidate_ids)

            ed_logits = (ed_mask_output.unsqueeze(2) * ed_candidate_embeddings).sum(-1)
            ed_bias = self.entity_prediction_bias(entity_candidate_ids).squeeze(-1)
            ed_logits = ed_logits + ed_bias

        entity_candidate_embeddings = self.entity_embeddings(
            entity_candidate_ids, entity_position_ids.unsqueeze(-2), entity_segment_ids.unsqueeze(-1))

        if self.args.use_softmax_average:
            ed_logits = ed_logits / self.args.entity_softmax_temp
            ed_logits.masked_fill_(entity_candidate_ids == 0, -10000.0)
            ed_probs = F.softmax(ed_logits, dim=-1)

            entity_embeddings = (entity_candidate_embeddings * ed_probs.unsqueeze(-1)).sum(-2)
        else:
            index_tensor = ed_logits.argmax(2, keepdim=True).unsqueeze(-1).expand(-1, -1, 1, self.config.hidden_size)
            entity_embeddings = torch.gather(entity_candidate_embeddings, 2, index_tensor).squeeze(2)

            ed_probs = F.softmax(ed_logits, dim=-1)
            mask = (ed_probs.max(2)[0] >= self.args.min_context_entity_prob).long()
            entity_attention_mask = entity_attention_mask * mask

        word_embeddings = self.embeddings(word_ids, word_segment_ids)
        attention_mask = self._compute_extended_attention_mask(word_attention_mask, entity_attention_mask)

        return self.encoder(word_embeddings, entity_embeddings, attention_mask)

    def load_state_dict(self, state_dict, *args, **kwargs):
        new_state_dict = state_dict.copy()

        for num in range(self.config.num_hidden_layers):
            for name in ('weight', 'bias'):
                for name2 in ('query', 'key', 'value'):
                    new_state_dict[f'encoder.layer.{num}.attention.self.entity_{name2}.{name}'] = state_dict[f'encoder.layer.{num}.attention.self.{name2}.{name}']
                    new_state_dict[f'encoder.layer.{num}.attention.self.gate_{name2}.{name}'] = state_dict[f'encoder.layer.{num}.attention.self.{name2}.{name}']
                    new_state_dict[f'encoder.layer.{num}.attention.self.w2e_gate_{name2}.{name}'] = state_dict[f'encoder.layer.{num}.attention.self.{name2}.{name}']
                    new_state_dict[f'encoder.layer.{num}.attention.self.e2w_gate_{name2}.{name}'] = state_dict[f'encoder.layer.{num}.attention.self.{name2}.{name}']
                    new_state_dict[f'encoder.layer.{num}.attention.self.w2e_{name2}.{name}'] = state_dict[f'encoder.layer.{num}.attention.self.{name2}.{name}']
                    new_state_dict[f'encoder.layer.{num}.attention.self.e2w_{name2}.{name}'] = state_dict[f'encoder.layer.{num}.attention.self.{name2}.{name}']
                    new_state_dict[f'encoder.layer.{num}.attention.self.e2e_{name2}.{name}'] = state_dict[f'encoder.layer.{num}.attention.self.{name2}.{name}']

        for key, weights in state_dict.items():
            if key.startswith('encoder.') or key.startswith('embeddings.') or key.startswith('entity_embeddings.'):
                new_state_dict['ed_' + key] = weights
        super(LukeExtendedTwoStageBaseModel, self).load_state_dict(new_state_dict, *args, **kwargs)


class GatedSelfAttention(nn.Module):
    def __init__(self, config):
        super(GatedSelfAttention, self).__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.w2e_gate_query = nn.Linear(config.hidden_size, self.all_head_size)
        self.w2e_gate_key = nn.Linear(config.hidden_size, self.all_head_size)
        self.e2w_gate_query = nn.Linear(config.hidden_size, self.all_head_size)
        self.e2w_gate_key = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, word_hidden_states, entity_hidden_states, attention_mask, layer_num):
        hidden_states = torch.cat([word_hidden_states, entity_hidden_states], dim=1)

        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        gate_scores = attention_scores.new_zeros(attention_scores.size())

        w2e_gate_query_layer = self.transpose_for_scores(self.w2e_gate_query(word_hidden_states))
        w2e_gate_key_layer = self.transpose_for_scores(self.w2e_gate_key(entity_hidden_states))
        w2e_gate_scores = torch.matmul(w2e_gate_query_layer, w2e_gate_key_layer.transpose(-1, -2))
        w2e_mask = attention_scores.new_zeros(attention_scores.size(), dtype=torch.bool)
        w2e_mask[:, :, :word_hidden_states.size(1), word_hidden_states.size(1):] = 1
        gate_scores = gate_scores.masked_scatter(w2e_mask, w2e_gate_scores)

        e2w_gate_query_layer = self.transpose_for_scores(self.e2w_gate_query(entity_hidden_states))
        e2w_gate_key_layer = self.transpose_for_scores(self.e2w_gate_key(word_hidden_states))
        e2w_gate_scores = torch.matmul(e2w_gate_query_layer, e2w_gate_key_layer.transpose(-1, -2))
        e2w_mask = attention_scores.new_zeros(attention_scores.size(), dtype=torch.bool)
        e2w_mask[:, :, word_hidden_states.size(1):, :word_hidden_states.size(1)] = 1
        gate_scores = gate_scores.masked_scatter(e2w_mask, e2w_gate_scores)

        gate_scores = torch.sigmoid(gate_scores) * -10.0
        attention_scores = attention_scores + gate_scores

        # gate_scores = F.sigmoid(-gate_scores) * -100.0
        # gate_scores[:, :, :word_hidden_states.size(1), :word_hidden_states.size(1)] = 0
        # gate_scores[:, :, word_hidden_states.size(1):, word_hidden_states.size(1):] = 0
        # gate_scores = torch.sigmoid(gate_scores)


        # gate_query_layer = self.transpose_for_scores(self.gate_query(hidden_states))
        # gate_key_layer = self.transpose_for_scores(self.gate_key(hidden_states))
        # gate_scores = torch.matmul(gate_query_layer, gate_key_layer.transpose(-1, -2))
        # gate_scores = F.sigmoid(-gate_scores) * -100.0
        # gate_scores[:, :, :word_hidden_states.size(1), :word_hidden_states.size(1)] = 0
        # gate_scores[:, :, word_hidden_states.size(1):, word_hidden_states.size(1):] = 0

        # attention_scores = attention_scores + gate_scores

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores + attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        word_context_layer = context_layer[:, :word_hidden_states.size(1), :]
        entity_context_layer = context_layer[:, word_hidden_states.size(1):, :]

        return word_context_layer, entity_context_layer


class GatedAttention(nn.Module):
    def __init__(self, config):
        super(GatedAttention, self).__init__()
        self.self = GatedSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, word_hidden_states, entity_hidden_states, attention_mask, layer_num):
        word_self_output, entity_self_output = self.self(word_hidden_states, entity_hidden_states,
                                                         attention_mask, layer_num)
        hidden_states = torch.cat([word_hidden_states, entity_hidden_states], dim=1)
        self_output = torch.cat([word_self_output, entity_self_output], dim=1)
        output = self.output(self_output, hidden_states)
        return output[:, :word_hidden_states.size(1), :], output[:, word_hidden_states.size(1):, :]


class Layer(nn.Module):
    def __init__(self, config):
        super(Layer, self).__init__()
        self.attention = GatedAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, word_hidden_states, entity_hidden_states, attention_mask, layer_num):
        word_attention_output, entity_attention_output = self.attention(word_hidden_states, entity_hidden_states,
                                                                        attention_mask, layer_num)
        attention_output = torch.cat([word_attention_output, entity_attention_output], dim=1)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output[:, :word_hidden_states.size(1), :], layer_output[:, word_hidden_states.size(1):, :]


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.layer = nn.ModuleList([Layer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, word_hidden_states, entity_hidden_states, attention_mask):
        for n, layer_module in enumerate(self.layer):
            word_hidden_states, entity_hidden_states = layer_module(word_hidden_states, entity_hidden_states,
                                                                    attention_mask, n)
        return word_hidden_states, entity_hidden_states
