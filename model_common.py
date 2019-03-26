# -*- coding: utf-8 -*-

import copy
import json
import logging
import math
import torch
from torch import nn

logger = logging.getLogger(__name__)


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class BaseConfig(object):
    def __repr__(self):
        return json.dumps(self.__dict__, indent=2, sort_keys=True) + "\n"

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        return output


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class WordEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(WordEmbeddings, self).__init__()
        self.config = config

        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class SelfAttention(nn.Module):
    def __init__(self, config):
        super(SelfAttention, self).__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        # batch, num_attention_heads, seq_length, attention_head_size
        return x.permute(0, 2, 1, 3)

    def forward(self, word_hidden_states, entity_hidden_states, attention_mask):
        hidden_states = torch.cat([word_hidden_states, entity_hidden_states], dim=1)
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        word_seq_size = word_hidden_states.size(1)

        word_context_layer = context_layer[:, :word_seq_size, :]
        entity_context_layer = context_layer[:, word_seq_size:, :]

        return (word_context_layer, entity_context_layer)


class SelfOutput(nn.Module):
    def __init__(self, config):
        super(SelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        self.self = SelfAttention(config)
        self.output = SelfOutput(config)

    def forward(self, word_hidden_states, entity_hidden_states, attention_mask):
        (word_self_output, entity_self_output) = self.self(word_hidden_states, entity_hidden_states,
                                                           attention_mask)
        word_output = self.output(word_self_output, word_hidden_states)
        entity_output = self.output(entity_self_output, entity_hidden_states)
        return (word_output, entity_output)


class Intermediate(nn.Module):
    def __init__(self, config):
        super(Intermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = gelu(hidden_states)
        return hidden_states


class Output(nn.Module):
    def __init__(self, config):
        super(Output, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class Layer(nn.Module):
    def __init__(self, config):
        super(Layer, self).__init__()
        self.attention = Attention(config)
        self.intermediate = Intermediate(config)
        self.output = Output(config)

    def forward(self, word_hidden_states, entity_hidden_states, attention_mask):
        (word_attention_output, entity_attention_output) = self.attention(
            word_hidden_states, entity_hidden_states, attention_mask)
        word_intermediate_output = self.intermediate(word_attention_output)
        entity_intermediate_output = self.intermediate(entity_attention_output)
        word_layer_output = self.output(word_intermediate_output, word_attention_output)
        entity_layer_output = self.output(entity_intermediate_output, entity_attention_output)
        return (word_layer_output, entity_layer_output)


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        layer = Layer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])

    def forward(self, word_hidden_states, entity_hidden_states, attention_mask,
                output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            (word_hidden_states, entity_hidden_states) = layer_module(word_hidden_states,
                entity_hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append((word_hidden_states, entity_hidden_states))
        if not output_all_encoded_layers:
            all_encoder_layers.append((word_hidden_states, entity_hidden_states))
        return all_encoder_layers


class Pooler(nn.Module):
    def __init__(self, config):
        super(Pooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, word_hidden_states):
        pooled_output = word_hidden_states[:, 0]
        pooled_output = self.dense(pooled_output)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class PredictionHeadTransform(nn.Module):
    def __init__(self, config, out_hidden_size=None):
        super(PredictionHeadTransform, self).__init__()
        if out_hidden_size is None:
            out_hidden_size = config.hidden_size

        self.dense = nn.Linear(config.hidden_size, out_hidden_size)
        self.LayerNorm = LayerNorm(out_hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = gelu(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class LMPredictionHead(nn.Module):
    def __init__(self, config, word_embedding_weights):
        super(LMPredictionHead, self).__init__()
        self.transform = PredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(word_embedding_weights.size(1),
                                 word_embedding_weights.size(0),
                                 bias=False)
        self.decoder.weight = word_embedding_weights
        self.bias = nn.Parameter(torch.zeros(word_embedding_weights.size(0)))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class BertPreTrainingHeads(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertPreTrainingHeads, self).__init__()
        self.predictions = LMPredictionHead(config, bert_model_embedding_weights)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class BaseModel(nn.Module):
    def __init__(self, config):
        super(BaseModel, self).__init__()
        self.config = config

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def load_bert_weights(self, state_dict):
        missing_keys = []
        unexpected_keys = []
        error_msgs = []

        # copy state_dict so _load_from_state_dict can modify it
        state_dict = state_dict.copy()
        for key in list(state_dict.keys()):
            new_key = key.replace('gamma', 'weight').replace('beta', 'bias')
            if new_key.startswith('bert.'):
                new_key = new_key[5:]

            if key != new_key:
                state_dict[new_key] = state_dict[key]
                del state_dict[key]

        def load(module, prefix=''):
            module._load_from_state_dict(
                state_dict, prefix, {}, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        load(self, prefix='')
        if len(unexpected_keys) > 0:
            logger.info("Weights from pretrained model not used in {}: {}".format(
                self.__class__.__name__, sorted(unexpected_keys)))
