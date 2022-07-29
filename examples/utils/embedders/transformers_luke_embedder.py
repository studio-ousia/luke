from typing import Optional

import torch
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder
from transformers.models.luke.modeling_luke import LukeModel


@TokenEmbedder.register("transformers-luke")
class TransformersLukeEmbedder(TokenEmbedder):
    def __init__(
        self,
        model_name: str,
        train_parameters: bool = True,
        output_embeddings: str = "token",
        use_entity_aware_attention: bool = False,
    ) -> None:
        """

        Parameters
        ----------
        model_name: str
            Model name registered in transformers
        train_parameters: `bool`
            Decide if tunening or freezing pre-trained weights.
        output_embeddings: `str`
            Choose output tokens or entities.
        """
        super().__init__()

        if output_embeddings not in {"token", "entity", "token+entity"}:
            raise ValueError(f"Invalid argument: {output_embeddings}")
        self.output_embeddings = output_embeddings

        self.luke_model = LukeModel.from_pretrained(model_name, use_entity_aware_attention=use_entity_aware_attention)
        if not train_parameters:
            for param in self.transformer_model.parameters():
                param.requires_grad = False

    def get_output_dim(self):
        return self.luke_model.config.hidden_size

    def forward(
        self,
        token_ids: torch.LongTensor,
        mask: torch.BoolTensor,
        type_ids: Optional[torch.LongTensor] = None,
        entity_ids: torch.LongTensor = None,
        entity_position_ids: torch.LongTensor = None,
        entity_segment_ids: torch.LongTensor = None,
        entity_attention_mask: torch.LongTensor = None,
    ) -> torch.Tensor:  # type: ignore

        if "entity" in self.output_embeddings and entity_ids is None:
            raise RuntimeError(
                "Entity embeddings are expected but the model cannot compute entity emebddings without entity_ids."
            )

        luke_outputs = self.luke_model(
            input_ids=token_ids,
            token_type_ids=type_ids,
            attention_mask=mask,
            entity_ids=entity_ids,
            entity_position_ids=entity_position_ids,
            entity_token_type_ids=entity_segment_ids,
            entity_attention_mask=entity_attention_mask,
        )

        if self.output_embeddings == "token":
            return luke_outputs.last_hidden_state
        elif self.output_embeddings == "entity":
            return luke_outputs.entity_last_hidden_state
        elif self.output_embeddings == "token+entity":
            return luke_outputs.last_hidden_state, luke_outputs.entity_last_hidden_state
        else:
            raise RuntimeError(f"Something is wrong with self.output_embeddings: {self.output_embeddings}.")
