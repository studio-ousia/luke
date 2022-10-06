import dataclasses
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from transformers import LukeConfig
from transformers.file_utils import ModelOutput
from transformers.models.bert.modeling_bert import BertEmbeddings
from transformers.models.luke.modeling_luke import (
    LukeEntityEmbeddings,
    LukeModel,
    LukePreTrainedModel,
    EntityPredictionHead,
)


@dataclasses.dataclass
class EntityDisambiguationOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    entity_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class LukeEntityEmbeddingsForEntityDisambiguation(LukeEntityEmbeddings):
    def __init__(self, config: LukeConfig):
        super().__init__(config)
        self.mask_embedding = nn.Parameter(torch.zeros(config.entity_emb_size))
        self._mask_entity_id = 2

    def forward(
        self, entity_ids: torch.LongTensor, position_ids: torch.LongTensor, token_type_ids: torch.LongTensor = None
    ):
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(entity_ids)

        entity_embeddings = self.entity_embeddings(entity_ids)
        entity_embeddings.masked_scatter_(
            (entity_ids == self._mask_entity_id).unsqueeze(-1),
            self.mask_embedding.unsqueeze(0).expand_as(entity_embeddings),
        )
        if self.config.entity_emb_size != self.config.hidden_size:
            entity_embeddings = self.entity_embedding_dense(entity_embeddings)

        position_embeddings = self.position_embeddings(position_ids.clamp(min=0))
        position_embedding_mask = (position_ids != -1).type_as(position_embeddings).unsqueeze(-1)
        position_embeddings = position_embeddings * position_embedding_mask
        position_embeddings = torch.sum(position_embeddings, dim=-2)
        position_embeddings = position_embeddings / position_embedding_mask.sum(dim=-2).clamp(min=1e-7)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = entity_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class LukeForEntityDisambiguation(LukePreTrainedModel):
    def __init__(self, config: LukeConfig):
        super().__init__(config)
        self.luke = LukeModel(config)
        self.entity_predictions = EntityPredictionHead(config)
        self.luke.embeddings = BertEmbeddings(config)
        self.luke.entity_embeddings = LukeEntityEmbeddingsForEntityDisambiguation(config)

        self.post_init()

    def tie_weights(self):
        super().tie_weights()
        self._tie_or_clone_weights(self.entity_predictions.decoder, self.luke.entity_embeddings.entity_embeddings)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        entity_ids: Optional[torch.LongTensor] = None,
        entity_candidate_ids: Optional[torch.LongTensor] = None,
        entity_attention_mask: Optional[torch.LongTensor] = None,
        entity_token_type_ids: Optional[torch.LongTensor] = None,
        entity_position_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, EntityDisambiguationOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.luke(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            entity_ids=entity_ids,
            entity_attention_mask=entity_attention_mask,
            entity_token_type_ids=entity_token_type_ids,
            entity_position_ids=entity_position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        logits = self.entity_predictions(outputs.entity_last_hidden_state).view(-1, self.config.entity_vocab_size)

        if entity_candidate_ids is not None:
            entity_candidate_ids = entity_candidate_ids.view(-1, entity_candidate_ids.size(-1))
            entity_candidate_mask = logits.new_zeros(logits.size(), dtype=torch.bool)
            entity_candidate_mask.scatter_(dim=1, index=entity_candidate_ids, src=(entity_candidate_ids != 0))
            logits = logits.masked_fill(~entity_candidate_mask, -1e32)
        logits = logits.view(entity_ids.size(0), -1, self.config.entity_vocab_size)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(labels.view(-1).size(0), -1), labels.view(-1), ignore_index=-1)

        if not return_dict:
            output = (
                logits,
                outputs.hidden_states,
                outputs.entity_hidden_states,
                outputs.attentions,
            )
            return ((loss,) + output) if loss is not None else output

        return EntityDisambiguationOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            entity_hidden_states=outputs.entity_hidden_states,
            attentions=outputs.attentions,
        )
