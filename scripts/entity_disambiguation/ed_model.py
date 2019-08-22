import torch
from torch.nn import CrossEntropyLoss

from luke.model import LukeModel, EntityPredictionHead


class LukeForEntityDisambiguation(LukeModel):
    def __init__(self, config):
        super(LukeForEntityDisambiguation, self).__init__(config)

        self.entity_predictions = EntityPredictionHead(config,
            self.entity_embeddings.entity_embeddings.weight)

        self.apply(self.init_weights)

    def forward(self, word_ids, word_segment_ids, word_attention_mask, entity_ids,
                entity_position_ids, entity_segment_ids, entity_attention_mask,
                entity_candidate_ids, entity_label=None):
        (encoded_layers, _) = super(LukeForEntityDisambiguation, self).forward(
            word_ids, word_segment_ids, word_attention_mask, entity_ids, entity_position_ids,
            entity_segment_ids, entity_attention_mask, output_all_encoded_layers=False)

        entity_output = encoded_layers[1][:, 0]

        logits = self.entity_predictions(entity_output).view(-1, self.config.entity_vocab_size)

        entity_candidate_mask = logits.new_zeros(logits.size(), dtype=torch.uint8)
        entity_candidate_mask.scatter_(dim=1, index=entity_candidate_ids,
                                       src=(entity_candidate_ids != 0))
        logits = logits.masked_fill((1 - entity_candidate_mask), -1e32)

        if entity_label is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(logits, entity_label)
            return loss
        else:
            return logits
