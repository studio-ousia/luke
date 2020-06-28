import torch.nn as nn
from torch.nn import CrossEntropyLoss

from luke.model import LukeEntityAwareAttentionModel


class LukeForReadingComprehension(LukeEntityAwareAttentionModel):
    def __init__(self, args):
        super(LukeForReadingComprehension, self).__init__(args.model_config)

        self.qa_outputs = nn.Linear(self.config.hidden_size, 2)
        self.apply(self.init_weights)

    def forward(
        self,
        word_ids,
        word_segment_ids,
        word_attention_mask,
        entity_ids,
        entity_position_ids,
        entity_segment_ids,
        entity_attention_mask,
        start_positions=None,
        end_positions=None,
    ):
        encoder_outputs = super(LukeForReadingComprehension, self).forward(
            word_ids,
            word_segment_ids,
            word_attention_mask,
            entity_ids,
            entity_position_ids,
            entity_segment_ids,
            entity_attention_mask,
        )

        word_hidden_states = encoder_outputs[0][:, : word_ids.size(1), :]
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

        return outputs + (start_logits, end_logits,)
