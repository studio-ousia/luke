import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from luke.model import LukeModel
from luke.pretraining.model import EntityPredictionHead


class LukeForQuestionAnswering(LukeModel):
    def __init__(self, config, entity_mask_id, entity_unk_id, use_softmax_average, entity_softmax_temp,
                 min_context_entity_prob, update_params_in_disambi):
        super(LukeForQuestionAnswering, self).__init__(config)

        self.config = config
        self._entity_mask_id = entity_mask_id
        self._entity_unk_id = entity_unk_id
        self._use_softmax_average = use_softmax_average
        self._entity_softmax_temp = entity_softmax_temp
        self._min_context_entity_prob = min_context_entity_prob
        self._update_params_in_disambi = update_params_in_disambi

        self.entity_predictions = EntityPredictionHead(config)
        self.entity_predictions.decoder.weight = self.entity_embeddings.entity_embeddings.weight
        self.entity_prediction_bias = nn.Embedding(config.entity_vocab_size, 1, padding_idx=0)
        # self.entity_prediction_bias.weight = self.entity_predictions.bias.view(-1, 1)
        self.entity_prediction_bias.weight.data = self.entity_predictions.bias.data.view(-1, 1)

        # self.transform = BertPredictionHeadTransform(config)
        # self.decoder = nn.Linear(config.hidden_size, config.entity_vocab_size, bias=False)
        # if config.entity_emb_size is not None and config.entity_emb_size != config.hidden_size:
        #     self.pre_decoder_dense = nn.Linear(config.hidden_size, config.entity_emb_size, bias=False)
        # self.bias = nn.Parameter(torch.zeros(config.entity_vocab_size))
        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        self.apply(self.init_weights)

    def forward(self, word_ids, word_segment_ids, word_attention_mask, entity_candidate_ids, entity_position_ids,
                entity_segment_ids, entity_attention_mask, start_positions=None, end_positions=None):
        if self._use_softmax_average:
            extended_attention_mask = self._compute_extended_attention_mask(word_attention_mask, entity_attention_mask)
            word_embeddings = self.embeddings(word_ids, word_segment_ids)
            mask_embeddings = self.entity_embeddings(entity_attention_mask * self._entity_mask_id,
                                                     entity_position_ids, entity_segment_ids)
            encoder_outputs = self.encoder(torch.cat([word_embeddings, mask_embeddings], dim=1),
                                           extended_attention_mask, [None] * self.config.num_hidden_layers)
            mask_sequence_output = encoder_outputs[0][:, word_ids.size(1):, :]
            mask_sequence_output = self.entity_predictions.transform(mask_sequence_output)
            if self.config.entity_emb_size != self.config.hidden_size:
                mask_sequence_output = self.entity_predictions.pre_decoder_dense(mask_sequence_output)
            candidate_embeddings = self.entity_embeddings.entity_embeddings(entity_candidate_ids)

            attention_logits = (mask_sequence_output.unsqueeze(2) * candidate_embeddings).sum(-1)
            attention_logits = attention_logits + self.entity_prediction_bias(entity_candidate_ids).squeeze(-1)
            attention_logits = attention_logits / self._entity_softmax_temp
            attention_logits.masked_fill_(entity_candidate_ids == 0, -10000.0)
            attention_probs = F.softmax(attention_logits, dim=-1)
            if not self._update_params_in_disambi:
                attention_probs = attention_probs.detach()

            entity_embeddings = self.entity_embeddings(entity_candidate_ids, entity_position_ids.unsqueeze(-2),
                                                       entity_segment_ids.unsqueeze(-1))
            entity_embeddings = (entity_embeddings * attention_probs.unsqueeze(-1)).sum(-2)

            # entity_prob = torch.sigmoid(self.entity_filter(attention_logits.max(2)[0].unsqueeze(-1)))
            # # entity_prob = torch.sigmoid(self.entity_filter(attention_logits.clamp(min=0.0) / 10.0))
            # entity_embeddings = entity_embeddings * entity_prob + self.entity_nil_embedding * (1.0 - entity_prob)
            # # print(entity_prob)
            if self._min_context_entity_prob != 0.0:
                mask = (attention_probs.max(2)[0] >= self._min_context_entity_prob).long()
                entity_attention_mask = entity_attention_mask * mask

            extended_attention_mask = self._compute_extended_attention_mask(word_attention_mask, entity_attention_mask)

            encoder_outputs = self.encoder(torch.cat([word_embeddings, entity_embeddings], dim=1),
                                           extended_attention_mask, [None] * self.config.num_hidden_layers)

        else:
            with torch.no_grad():
                mask_entity_ids = entity_attention_mask * self._entity_mask_id
                encoder_outputs = super(LukeForQuestionAnswering, self).forward(
                    word_ids, word_segment_ids, word_attention_mask, mask_entity_ids, entity_position_ids,
                    entity_segment_ids, entity_attention_mask
                )
                logits = self.entity_predictions(encoder_outputs[1]).view(-1, self.config.entity_vocab_size)
                entity_candidate_ids = entity_candidate_ids.reshape(-1, entity_candidate_ids.size(2))
                entity_candidate_mask = logits.new_zeros(logits.size(), dtype=torch.bool)
                entity_candidate_mask.scatter_(dim=1, index=entity_candidate_ids, src=(entity_candidate_ids != 0))
                logits = logits.masked_fill(~entity_candidate_mask, -10000.0).view(mask_entity_ids.size(0), -1,
                                                                                   self.config.entity_vocab_size)

                predicted_entity_ids = logits.argmax(2) * entity_attention_mask
                predicted_entity_ids = predicted_entity_ids * (predicted_entity_ids != self._entity_unk_id).long()
                entity_attention_mask = (predicted_entity_ids != 0).long()

                if self._min_context_entity_prob != 0.0:
                    entity_probs = F.softmax(logits, dim=2)
                    mask = (entity_probs.max(2)[0] >= self._min_context_entity_prob).long()
                    predicted_entity_ids = predicted_entity_ids * mask
                    entity_attention_mask = entity_attention_mask * mask

            encoder_outputs = super(LukeForQuestionAnswering, self).forward(
                word_ids, word_segment_ids, word_attention_mask, predicted_entity_ids, entity_position_ids,
                entity_segment_ids, entity_attention_mask)

        # extended_attention_mask = self._compute_extended_attention_mask(word_attention_mask, entity_attention_mask)
        # extended_attention_mask = extended_attention_mask[:, :, :, :word_ids.size(1)]
        # word_embeddings = self.embeddings(word_ids, word_segment_ids)
        # encoder_outputs = self.encoder(word_embeddings, extended_attention_mask,
        #                                [None] * self.config.num_hidden_layers)
        # word_hidden_states = encoder_outputs[0]
        word_hidden_states = encoder_outputs[0][:, :word_ids.size(1), :]
        logits = self.qa_outputs(word_hidden_states)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        if start_positions is not None and end_positions is not None:
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)

            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,)
        else:
            outputs = tuple()

        # start_logits = start_logits[:, :word_ids.size(1)]
        # end_logits = end_logits[:, :word_ids.size(1)]

        return outputs + (start_logits, end_logits,)
