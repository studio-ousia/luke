import torch

from luke.model import LukeModel
from luke.pretraining.model import EntityPredictionHead


class LukeForEntityDisambiguation(LukeModel):
    def __init__(self, config):
        super(LukeForEntityDisambiguation, self).__init__(config)

        self.entity_predictions = EntityPredictionHead(config)
        self.entity_predictions.decoder.weight = self.entity_embeddings.entity_embeddings.weight

        self.apply(self.init_weights)

    def forward(self, word_ids, word_segment_ids, word_attention_mask, entity_ids, entity_position_ids,
                entity_segment_ids, entity_attention_mask, candidate_ids):
        output = super(LukeForEntityDisambiguation, self).forward(
            word_ids, word_segment_ids, word_attention_mask, entity_ids, entity_position_ids, entity_segment_ids,
            entity_attention_mask)
        entity_sequence_output = output[1]
        logits = self.entity_predictions(entity_sequence_output).view(-1, self.config.entity_vocab_size)
        candidate_ids = candidate_ids.view(-1, candidate_ids.size()[-1])

        entity_candidate_mask = logits.new_zeros(logits.size(), dtype=torch.bool)
        entity_candidate_mask.scatter_(dim=1, index=candidate_ids, src=(candidate_ids != 0))
        logits = logits.masked_fill(~entity_candidate_mask, -10000.0)
        return logits.view(entity_ids.size(0), -1, self.config.entity_vocab_size)
