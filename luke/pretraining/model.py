import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from pytorch_transformers.modeling_bert import BertPreTrainingHeads, BertPredictionHeadTransform

from luke.model import LukeModel, LukeE2EModel


class EntityPredictionHead(nn.Module):
    def __init__(self, config):
        super(EntityPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, config.entity_vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.entity_vocab_size))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias

        return hidden_states


class LukePretrainingModel(LukeModel):
    def __init__(self, config):
        super(LukePretrainingModel, self).__init__(config)

        self.cls = BertPreTrainingHeads(config)
        self.cls.predictions.decoder.weight = self.embeddings.word_embeddings.weight
        self.entity_predictions = EntityPredictionHead(config)
        self.entity_predictions.decoder.weight = self.entity_embeddings.entity_embeddings.weight

        self.apply(self.init_weights)

    def forward(self, word_ids, word_segment_ids, word_attention_mask, entity_ids, entity_position_ids,
                entity_segment_ids, entity_attention_mask, masked_entity_labels=None, masked_lm_labels=None,
                is_random_next=None, **kwargs):
        output = super(LukePretrainingModel, self).forward(
            word_ids, word_segment_ids, word_attention_mask, entity_ids, entity_position_ids, entity_segment_ids,
            entity_attention_mask
        )
        word_sequence_output, entity_sequence_output, pooled_output = output[:3]

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


class LukeE2EPretrainingModel(LukeE2EModel):
    def __init__(self, config):
        super(LukeE2EPretrainingModel, self).__init__(config)

        self.cls = BertPreTrainingHeads(config)
        self.cls.predictions.decoder.weight = self.embeddings.word_embeddings.weight
        self.entity_predictions = EntityPredictionHead(config)
        self.entity_predictions.decoder.weight = self.entity_embeddings.entity_embeddings.weight

        self.apply(self.init_weights)

    def forward(self, word_ids, word_segment_ids, word_attention_mask, entity_candidate_ids, entity_position_ids,
                entity_segment_ids, entity_attention_mask, entity_candidate_labels=None, masked_entity_labels=None,
                masked_lm_labels=None, is_random_next=None, **kwargs):
        output = super(LukeE2EPretrainingModel, self).forward(
            word_ids, word_segment_ids, word_attention_mask, entity_candidate_ids, entity_position_ids,
            entity_segment_ids, entity_attention_mask, masked_entity_labels=masked_entity_labels,
            output_entity_selector_scores=True
        )
        word_sequence_output, entity_sequence_output, pooled_output, entity_selector_scores = output[:4]

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
