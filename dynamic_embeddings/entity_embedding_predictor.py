import torch
import torch.nn as nn
from allennlp.data import Vocabulary
from allennlp.models import Model

from transformers import AutoTokenizer
from transformers.models.luke import LukeModel


@Model.register("entity_embedding_predictor")
class EntityEmbeddingPredictor(Model):
    def __init__(self, vocab: Vocabulary, luke_model_name: str, dropout: float = 0.1, freeze_encoder: bool = False):

        super().__init__(vocab=vocab)
        self.luke_model: LukeModel = LukeModel.from_pretrained(luke_model_name)
        self.luke_model.entity_embeddings.entity_embeddings.weight.requires_grad = False
        self.static_entity_embeddings = self.luke_model.entity_embeddings.entity_embeddings.weight.data

        if freeze_encoder:
            for param in self.luke_model.parameters():
                param.requires_grad = False

        self.pad_token_id = self.luke_model.config.pad_token_id
        self.entity_pad_token_id = AutoTokenizer.from_pretrained(luke_model_name).entity_vocab["[PAD]"]

        self.linear = nn.Linear(self.luke_model.config.hidden_size, self.luke_model.config.entity_emb_size)

        self.dropout = nn.Dropout(p=dropout)
        self.criterion = nn.MSELoss()

    def forward(
        self,
        word_ids: torch.LongTensor,
        entity_mask_tokens: torch.LongTensor,
        entity_position_ids: torch.LongTensor,
        gold_entity_ids: torch.LongTensor = None,
        **kwargs,
    ):

        model_output = self.luke_model.forward(
            word_ids,
            attention_mask=word_ids != self.pad_token_id,
            entity_ids=entity_mask_tokens,
            entity_attention_mask=entity_mask_tokens != self.entity_pad_token_id,
            entity_position_ids=entity_position_ids,
            return_dict=True,
        )
        entity_last_hidden_state = model_output.entity_last_hidden_state

        predicted_entity_embeddings = self.linear(entity_last_hidden_state)

        output_dict = {"predicted_entity_embeddings": predicted_entity_embeddings}

        if gold_entity_ids is not None:
            gold_entity_embeddings = self.static_entity_embeddings[gold_entity_ids]
            loss = self.criterion(gold_entity_embeddings, predicted_entity_embeddings)
            output_dict["loss"] = loss
        return output_dict
