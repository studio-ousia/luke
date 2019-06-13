import copy
import json
import logging
import math
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss

logger = logging.getLogger(__name__)


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class LukeConfig(object):
    def __init__(self, vocab_size, entity_vocab_size, hidden_size, num_hidden_layers, num_attention_heads,
                 intermediate_size, hidden_dropout_prob, attention_probs_dropout_prob, max_position_embeddings,
                 type_vocab_size, initializer_range, **kwargs):
        self.vocab_size = vocab_size
        self.entity_vocab_size = entity_vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range

    def __repr__(self):
        return json.dumps(self.__dict__, indent=2, sort_keys=True) + "\n"

    def to_dict(self):
        return copy.deepcopy(self.__dict__)


class LukeE2EConfig(LukeConfig):
    def __init__(self, num_t_hidden_layers, entity_selector_softmax_temp, **kwargs):
        super(LukeE2EConfig, self).__init__(**kwargs)

        self.num_t_hidden_layers = num_t_hidden_layers
        self.entity_selector_softmax_temp = entity_selector_softmax_temp


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
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


class EntityEmbeddings(nn.Module):
    def __init__(self, config):
        super(EntityEmbeddings, self).__init__()
        self.config = config

        self.entity_embeddings = nn.Embedding(config.entity_vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.LayerNorm = LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, entity_ids, position_ids, token_type_ids=None):
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(entity_ids)

        entity_embeddings = self.entity_embeddings(entity_ids)

        # We first set all padding indices (i.e., -1) to 0, select embeddings using the indices, and then mask out the
        # embeddings for padding tokens.
        position_embeddings = self.position_embeddings(position_ids.clamp(min=0))
        position_embedding_mask = (position_ids != -1).type_as(position_embeddings).unsqueeze(-1)
        position_embeddings = position_embeddings * position_embedding_mask

        position_embeddings = torch.sum(position_embeddings, dim=-2)
        position_embeddings = position_embeddings / (position_embedding_mask.sum(dim=-2) + 1e-12)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = entity_embeddings + position_embeddings + token_type_embeddings
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
        attention_probs = F.softmax(attention_scores, dim=-1)

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
        (word_self_output, entity_self_output) = self.self(word_hidden_states, entity_hidden_states, attention_mask)
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
        (word_attention_output, entity_attention_output) = self.attention(word_hidden_states, entity_hidden_states,
                                                                          attention_mask)
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

    def forward(self, word_hidden_states, entity_hidden_states, attention_mask, output_all_encoded_layers=True,
                target_layers=None):
        all_encoder_layers = []
        for (n, layer_module) in enumerate(self.layer):
            if target_layers and n not in target_layers:
                continue
            (word_hidden_states, entity_hidden_states) = layer_module(word_hidden_states, entity_hidden_states,
                                                                      attention_mask)
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
    def __init__(self, config):
        super(PredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = LayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = gelu(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)

        return hidden_states


class LMPredictionHead(nn.Module):
    def __init__(self, config, word_embedding_weights):
        super(LMPredictionHead, self).__init__()
        self.transform = PredictionHeadTransform(config)
        self.decoder = nn.Linear(word_embedding_weights.size(1), word_embedding_weights.size(0), bias=False)
        self.decoder.weight = word_embedding_weights
        self.bias = nn.Parameter(torch.zeros(word_embedding_weights.size(0)))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias

        return hidden_states


class EntityPredictionHead(nn.Module):
    def __init__(self, config, entity_embedding_weights):
        super(EntityPredictionHead, self).__init__()
        self.transform = PredictionHeadTransform(config)
        self.decoder = nn.Linear(entity_embedding_weights.size(1), entity_embedding_weights.size(0), bias=False)
        self.decoder.weight = entity_embedding_weights
        self.bias = nn.Parameter(torch.zeros(entity_embedding_weights.size(0)))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias

        return hidden_states


class EntitySelector(nn.Module):
    def __init__(self, config, entity_embedding_weights):
        super(EntitySelector, self).__init__()
        self.transform = PredictionHeadTransform(config)
        self.embeddings = nn.Embedding(entity_embedding_weights.size(0), entity_embedding_weights.size(1),
                                       padding_idx=0)
        self.embeddings.weight = entity_embedding_weights
        # be careful for weight decay!!
        self.bias = nn.Embedding(config.entity_vocab_size, 1, padding_idx=0)

    def forward(self, hidden_states, entity_candidate_ids):
        # entity_embeddings: [batch_size, entity_length, entity_candidate_length, hidden_size]
        entity_embeddings = self.embeddings(entity_candidate_ids)
        # entity_bias: [batch_size, entity_length, entity_candidate_length]
        entity_bias = self.bias(entity_candidate_ids).squeeze(-1)

        hidden_states = self.transform(hidden_states)

        scores = (hidden_states.unsqueeze(-2) * entity_embeddings).sum(-1) + entity_bias
        scores += (entity_candidate_ids == 0).to(dtype=scores.dtype) * -10000.0

        return scores


class BertPreTrainingHeads(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertPreTrainingHeads, self).__init__()
        self.predictions = LMPredictionHead(config, bert_model_embedding_weights)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)

        return prediction_scores, seq_relationship_score


class LukeBaseModel(nn.Module):
    def __init__(self, config):
        super(LukeBaseModel, self).__init__()

        self.config = config

        self.encoder = Encoder(config)
        self.pooler = Pooler(config)
        self.embeddings = WordEmbeddings(config)
        self.entity_embeddings = EntityEmbeddings(config)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.Embedding):
            if module.embedding_dim == 1:  # embedding for bias parameters
                module.weight.data.zero_()
            else:
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
            module._load_from_state_dict(state_dict, prefix, {}, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        load(self, prefix='')
        if len(unexpected_keys) > 0:
            logger.info("Weights from pretrained model not used in {}: {}".format(self.__class__.__name__,
                                                                                  sorted(unexpected_keys)))

    def _compute_extended_attention_mask(self, word_attention_mask, entity_attention_mask):
        attention_mask = torch.cat([word_attention_mask, entity_attention_mask], dim=1)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        return extended_attention_mask


class LukeModel(LukeBaseModel):
    def forward(self, word_ids, word_segment_ids, word_attention_mask, entity_ids, entity_position_ids,
                entity_segment_ids, entity_attention_mask, output_all_encoded_layers=True):
        extended_attention_mask = self._compute_extended_attention_mask(word_attention_mask, entity_attention_mask)

        word_embedding_output = self.embeddings(word_ids, word_segment_ids)
        entity_embedding_output = self.entity_embeddings(entity_ids, entity_position_ids, entity_segment_ids)
        encoded_layers = self.encoder(word_embedding_output, entity_embedding_output, extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers)
        word_sequence_output = encoded_layers[-1][0]
        pooled_output = self.pooler(word_sequence_output)

        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]

        return (encoded_layers, pooled_output)


class LukePretrainingModel(LukeModel):
    def __init__(self, config):
        super(LukePretrainingModel, self).__init__(config)

        self.cls = BertPreTrainingHeads(config, self.embeddings.word_embeddings.weight)
        self.entity_predictions = EntityPredictionHead(config, self.entity_embeddings.entity_embeddings.weight)

        self.apply(self.init_weights)

    def forward(self, word_ids, word_segment_ids, word_attention_mask, entity_ids, entity_position_ids,
                entity_segment_ids, entity_attention_mask, masked_entity_labels=None, masked_lm_labels=None,
                is_random_next=None, **kwargs):
        (encoded_layers, pooled_output) = super(LukePretrainingModel, self).forward(
            word_ids, word_segment_ids, word_attention_mask, entity_ids, entity_position_ids, entity_segment_ids,
            entity_attention_mask, output_all_encoded_layers=False
        )
        word_sequence_output = encoded_layers[0]
        entity_sequence_output = encoded_layers[1]

        loss_fn = CrossEntropyLoss(ignore_index=-1)
        ret = dict(loss=0.0)

        if masked_entity_labels is not None:
            entity_mask = (masked_entity_labels != -1)
            masked_entity_sequence_output = torch.masked_select(entity_sequence_output, entity_mask.unsqueeze(-1))
            masked_entity_sequence_output = masked_entity_sequence_output.view(-1, self.config.hidden_size)

            entity_scores = self.entity_predictions(masked_entity_sequence_output)
            entity_scores = entity_scores.view(-1, self.config.entity_vocab_size)
            entity_labels = torch.masked_select(masked_entity_labels, entity_mask)

            ret['masked_entity_loss'] = loss_fn(entity_scores, entity_labels)
            ret['masked_entity_correct'] = (torch.argmax(entity_scores, 1).data == entity_labels.data).sum()
            ret['masked_entity_total'] = entity_labels.ne(-1).sum()
            ret['loss'] += ret['masked_entity_loss']

        if masked_lm_labels is not None:
            masked_lm_mask = (masked_lm_labels != -1)
            masked_word_sequence_output = torch.masked_select(word_sequence_output, masked_lm_mask.unsqueeze(-1))
            masked_word_sequence_output = masked_word_sequence_output.view(-1, self.config.hidden_size)

            masked_lm_scores = self.cls.predictions(masked_word_sequence_output)
            masked_lm_scores = masked_lm_scores.view(-1, self.config.vocab_size)
            masked_lm_labels = torch.masked_select(masked_lm_labels, masked_lm_mask)

            ret['masked_lm_loss'] = loss_fn(masked_lm_scores, masked_lm_labels)
            ret['masked_lm_correct'] = (torch.argmax(masked_lm_scores, 1).data == masked_lm_labels.data).sum()
            ret['masked_lm_total'] = masked_lm_labels.ne(-1).sum()
            ret['loss'] += ret['masked_lm_loss']

        if is_random_next is not None:
            nsp_score = self.cls.seq_relationship(pooled_output)
            ret['nsp_loss'] = loss_fn(nsp_score, is_random_next)
            ret['nsp_correct'] = (torch.argmax(nsp_score, 1).data == is_random_next.data).sum()
            ret['nsp_total'] = ret['nsp_correct'].new_tensor(word_ids.size(0))
            ret['loss'] += ret['nsp_loss']

        return ret


class LukeE2EModel(LukeBaseModel):
    def __init__(self, config):
        super(LukeE2EModel, self).__init__(config)

        self.entity_selector = EntitySelector(config, self.entity_embeddings.entity_embeddings.weight)
        self.t_target_layers = frozenset(range(0, config.num_t_hidden_layers))
        self.k_target_layers = frozenset(range(config.num_t_hidden_layers, config.num_hidden_layers))

    def forward(self, word_ids, word_segment_ids, word_attention_mask, entity_candidate_ids, entity_position_ids,
                entity_segment_ids, entity_attention_mask, masked_entity_labels=None, output_all_encoded_layers=True,
                output_entity_selector_scores=False):
        extended_attention_mask = self._compute_extended_attention_mask(word_attention_mask, entity_attention_mask)

        word_embedding_output = self.embeddings(word_ids, word_segment_ids)
        entity_ids = entity_attention_mask * 3  # index of [MASK2] token is 3
        mask_entity_embedding_output = self.entity_embeddings(entity_ids, entity_position_ids, entity_segment_ids)

        t_encoded_layers = self.encoder(word_embedding_output, mask_entity_embedding_output, extended_attention_mask,
                                        target_layers=self.t_target_layers,
                                        output_all_encoded_layers=output_all_encoded_layers)
        (t_word_sequence_output, t_entity_sequence_output) = t_encoded_layers[-1]
        # entity_selector_scores: [batch_size, entity_length, entity_candidate_length]
        entity_selector_scores = self.entity_selector(t_entity_sequence_output, entity_candidate_ids)
        entity_selector_scores = entity_selector_scores / self.config.entity_selector_softmax_temp
        entity_attention_probs = F.softmax(entity_selector_scores, dim=-1)

        # entity_candidate_ids: batch x entity_size x entity_candidate_size
        # entity_segment_ids: batch x entity_size
        # entity_position_ids: batch x entity_size x mention_size
        # entity_embedding_output: batch x entity_size x entity_candidate_size x hidden_size
        entity_embedding_output = self.entity_embeddings(entity_candidate_ids, entity_position_ids.unsqueeze(-2),
                                                         entity_segment_ids.unsqueeze(-1))
        entity_embedding_output = (entity_embedding_output * entity_attention_probs.unsqueeze(-1)).sum(-2)

        if masked_entity_labels is not None:
            mask_entity_ids = entity_segment_ids.new_full(entity_segment_ids.size(), 2)  # index of [MASK] token is 2
            mask_embedding_output = self.entity_embeddings(mask_entity_ids, entity_position_ids, entity_segment_ids)
            entity_embedding_output.masked_scatter_((masked_entity_labels != -1).unsqueeze(-1), mask_embedding_output)

        k_encoded_layers = self.encoder(t_word_sequence_output, entity_embedding_output, extended_attention_mask,
                                        target_layers=self.k_target_layers,
                                        output_all_encoded_layers=output_all_encoded_layers)

        word_sequence_output = k_encoded_layers[-1][0]
        pooled_output = self.pooler(word_sequence_output)

        if not output_all_encoded_layers:
            k_encoded_layers = k_encoded_layers[-1]

        if output_entity_selector_scores:
            return (k_encoded_layers, pooled_output, entity_selector_scores)
        else:
            return (k_encoded_layers, pooled_output)


class LukeE2EPretrainingModel(LukeE2EModel):
    def __init__(self, config):
        super(LukeE2EPretrainingModel, self).__init__(config)

        self.cls = BertPreTrainingHeads(config, self.embeddings.word_embeddings.weight)
        self.entity_predictions = EntityPredictionHead(config, self.entity_embeddings.entity_embeddings.weight)

        self.apply(self.init_weights)

    def forward(self, word_ids, word_segment_ids, word_attention_mask, entity_candidate_ids, entity_position_ids,
                entity_segment_ids, entity_attention_mask, entity_candidate_labels=None, masked_entity_labels=None,
                masked_lm_labels=None, is_random_next=None, **kwargs):
        (encoded_layers, pooled_output, entity_selector_scores) = super(LukeE2EPretrainingModel, self).forward(
            word_ids, word_segment_ids, word_attention_mask, entity_candidate_ids, entity_position_ids,
            entity_segment_ids, entity_attention_mask, masked_entity_labels=masked_entity_labels,
            output_all_encoded_layers=False, output_entity_selector_scores=True
        )
        word_sequence_output = encoded_layers[0]
        entity_sequence_output = encoded_layers[1]

        loss_fn = CrossEntropyLoss(ignore_index=-1)
        ret = dict(loss=0.0)

        if masked_entity_labels is not None:
            masked_entity_mask = (masked_entity_labels != -1)
            masked_entity_sequence_output = torch.masked_select(entity_sequence_output,
                                                                masked_entity_mask.unsqueeze(-1))
            masked_entity_sequence_output = masked_entity_sequence_output.view(-1, self.config.hidden_size)

            entity_scores = self.entity_predictions(masked_entity_sequence_output)
            entity_scores = entity_scores.view(-1, self.config.entity_vocab_size)
            entity_labels = torch.masked_select(masked_entity_labels, masked_entity_mask)

            ret['masked_entity_loss'] = loss_fn(entity_scores, entity_labels)
            ret['masked_entity_correct'] = (torch.argmax(entity_scores, 1).data == entity_labels.data).sum()
            ret['masked_entity_total'] = entity_labels.ne(-1).sum()
            ret['loss'] += ret['masked_entity_loss']

        if masked_lm_labels is not None:
            masked_lm_mask = (masked_lm_labels != -1)
            masked_word_sequence_output = torch.masked_select(word_sequence_output, masked_lm_mask.unsqueeze(-1))
            masked_word_sequence_output = masked_word_sequence_output.view(-1, self.config.hidden_size)

            masked_lm_scores = self.cls.predictions(masked_word_sequence_output)
            masked_lm_scores = masked_lm_scores.view(-1, self.config.vocab_size)
            masked_lm_labels = torch.masked_select(masked_lm_labels, masked_lm_mask)

            ret['masked_lm_loss'] = loss_fn(masked_lm_scores, masked_lm_labels)
            ret['masked_lm_correct'] = (torch.argmax(masked_lm_scores, 1).data == masked_lm_labels.data).sum()
            ret['masked_lm_total'] = masked_lm_labels.ne(-1).sum()
            ret['loss'] += ret['masked_lm_loss']

        if is_random_next is not None:
            nsp_score = self.cls.seq_relationship(pooled_output)

            ret['nsp_loss'] = loss_fn(nsp_score, is_random_next)
            ret['nsp_correct'] = (torch.argmax(nsp_score, 1).data == is_random_next.data).sum()
            ret['nsp_total'] = ret['nsp_correct'].new_tensor(word_ids.size(0))
            ret['loss'] += ret['nsp_loss']

        if entity_candidate_labels is not None:
            entity_candidate_labels = entity_candidate_labels.view(-1)
            entity_selector_scores = entity_selector_scores.view(entity_candidate_labels.size(0), -1)

            ret['entity_selector_loss'] = loss_fn(entity_selector_scores, entity_candidate_labels)
            ret['entity_selector_correct'] = (torch.argmax(entity_selector_scores, 1).data ==
                                              entity_candidate_labels.data).sum()
            ret['entity_selector_total'] = (entity_candidate_labels != -1).sum()
            ret['loss'] += ret['entity_selector_loss']

        return ret
