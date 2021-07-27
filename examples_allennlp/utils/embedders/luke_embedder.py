from typing import Optional
import json
from overrides import overrides

import torch
from transformers import AutoConfig

from allennlp.modules.token_embedders.token_embedder import TokenEmbedder

from luke.utils.entity_vocab import EntityVocab, MASK_TOKEN, PAD_TOKEN
from luke.model import LukeModel, LukeConfig


@TokenEmbedder.register("luke")
class PretrainedLukeEmbedder(TokenEmbedder):
    def __init__(
        self,
        pretrained_weight_path: str,
        pretrained_metadata_path: str,
        entity_vocab_path: str = None,
        train_parameters: bool = True,
        gradient_checkpointing: bool = False,
        num_special_mask_embeddings: int = None,
        output_entity_embeddings: bool = False,
        num_additional_special_tokens: int = None,
        discard_entity_embeddings: bool = False,
    ) -> None:
        """

        Parameters
        ----------
        pretrained_weight_path: `str`
            Path to the luke pre-trained weight.

        pretrained_metadata_path: `str`
            Path to the luke pre-trained metadata, typically stored under the same directory as pretrained_weight_path.

        entity_vocab_path: `str`
            Path to the luke entity vocabulary.

        train_parameters: `bool`
            Decide if tunening or freezing pre-trained weights.

        gradient_checkpointing: `bool`
            Enable gradient checkpoinitng, which significantly reduce memory usage.

        num_special_mask_embeddings: `int`
            If specified, the model discard all the entity embeddings
            and only use the number of embeddings initialized with [MASK].
            This is used with the tasks such as named entity recognition (num_special_mask_embeddings=1),
            or relation classification (num_special_mask_embeddings=2).

        output_entity_embeddings: `bool`
            If specified, the model returns entity embeddings instead of token embeddings.
            If you need both, please use PretrainedLukeEmbedderWithEntity.

        num_additional_special_tokens: `int`
            Used when adding special tokens to the pre-trained vocabulary.
        discard_entity_embeddings: `bool`
            Replace entity embeddings with a dummy vector to save memory.
        """
        super().__init__()

        self.metadata = json.load(open(pretrained_metadata_path, "r"))["model_config"]
        if entity_vocab_path is not None:
            self.entity_vocab = EntityVocab(entity_vocab_path)
        else:
            self.entity_vocab = None

        model_weights = torch.load(pretrained_weight_path, map_location=torch.device("cpu"))
        self.num_special_mask_embeddings = num_special_mask_embeddings
        if num_special_mask_embeddings:
            assert self.entity_vocab is not None
            pad_id = self.entity_vocab.special_token_ids[PAD_TOKEN]
            mask_id = self.entity_vocab.special_token_ids[MASK_TOKEN]
            self.metadata["entity_vocab_size"] = 1 + num_special_mask_embeddings
            entity_emb = model_weights["entity_embeddings.entity_embeddings.weight"]
            mask_emb = entity_emb[mask_id].unsqueeze(0)
            pad_emb = entity_emb[pad_id].unsqueeze(0)
            model_weights["entity_embeddings.entity_embeddings.weight"] = torch.cat(
                [pad_emb] + [mask_emb for _ in range(num_special_mask_embeddings)]
            )

        if discard_entity_embeddings:
            self.metadata["entity_vocab_size"] = 1
            model_weights["entity_embeddings.entity_embeddings.weight"] = torch.zeros(
                1, self.metadata["entity_emb_size"]
            )

        config = LukeConfig(
            entity_vocab_size=self.metadata["entity_vocab_size"],
            bert_model_name=self.metadata["bert_model_name"],
            entity_emb_size=self.metadata["entity_emb_size"],
            **AutoConfig.from_pretrained(self.metadata["bert_model_name"]).to_dict(),
        )
        config.gradient_checkpointing = gradient_checkpointing

        self.output_entity_embeddings = output_entity_embeddings

        self.luke_model = LukeModel(config)
        self.luke_model.load_state_dict(model_weights, strict=False)

        if num_additional_special_tokens:
            word_emb = self.luke_model.embeddings.word_embeddings.weight
            embed_size = word_emb.size(1)
            additional_embs = [torch.rand(1, embed_size) for _ in range(num_additional_special_tokens)]
            augmented_weight = torch.nn.Parameter(torch.cat([word_emb] + additional_embs, dim=0))
            self.luke_model.embeddings.word_embeddings.weight = augmented_weight

        if not train_parameters:
            for param in self.transformer_model.parameters():
                param.requires_grad = False

    @overrides
    def get_output_dim(self):
        return self.metadata["hidden_size"]

    @overrides
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

        if self.output_entity_embeddings:
            assert entity_ids is not None

        luke_outputs = self.luke_model(
            token_ids,
            word_segment_ids=type_ids,
            word_attention_mask=mask,
            entity_ids=entity_ids,
            entity_position_ids=entity_position_ids,
            entity_segment_ids=entity_segment_ids,
            entity_attention_mask=entity_attention_mask,
        )

        if self.output_entity_embeddings:
            return luke_outputs[1]
        else:
            return luke_outputs[0]
