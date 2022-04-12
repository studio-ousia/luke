import torch
import torch.nn as nn

from transformers import LukePreTrainedModel, LukeModel
from transformers.models.luke.modeling_luke import EntityPairClassificationOutput

# temporary use this implementation until this gets merged into transformers
class LukeForEntityPairClassification(LukePreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.luke = LukeModel(config)

        self.num_labels = config.num_labels
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size * 2, config.num_labels, config.classifier_bias)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        entity_ids=None,
        entity_attention_mask=None,
        entity_token_type_ids=None,
        entity_position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)` or :obj:`(batch_size, num_labels)`, `optional`):
            Labels for computing the classification loss. If the shape is :obj:`(batch_size,)`, the cross entropy loss
            is used for the single-label classification. In this case, labels should contain the indices that should be
            in :obj:`[0, ..., config.num_labels - 1]`. If the shape is :obj:`(batch_size, num_labels)`, the binary
            cross entropy loss is used for the multi-label classification. In this case, labels should only contain
            ``[0, 1]``, where 0 and 1 indicate false and true, respectively.

        Returns:

        Examples::

            >>> from transformers import LukeTokenizer, LukeForEntityPairClassification

            >>> tokenizer = LukeTokenizer.from_pretrained("studio-ousia/luke-large-finetuned-tacred")
            >>> model = LukeForEntityPairClassification.from_pretrained("studio-ousia/luke-large-finetuned-tacred")

            >>> text = "Beyoncé lives in Los Angeles."
            >>> entity_spans = [(0, 7), (17, 28)]  # character-based entity spans corresponding to "Beyoncé" and "Los Angeles"
            >>> inputs = tokenizer(text, entity_spans=entity_spans, return_tensors="pt")
            >>> outputs = model(**inputs)
            >>> logits = outputs.logits
            >>> predicted_class_idx = logits.argmax(-1).item()
            >>> print("Predicted class:", model.config.id2label[predicted_class_idx])
            Predicted class: per:cities_of_residence
        """
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

        feature_vector = torch.cat(
            [outputs.entity_last_hidden_state[:, 0, :], outputs.entity_last_hidden_state[:, 1, :]], dim=1
        )
        feature_vector = self.dropout(feature_vector)
        logits = self.classifier(feature_vector)

        loss = None
        if labels is not None:
            # When the number of dimension of `labels` is 1, cross entropy is used as the loss function. The binary
            # cross entropy is used otherwise.
            if labels.ndim == 1:
                loss = nn.functional.cross_entropy(logits, labels)
            else:
                loss = nn.functional.binary_cross_entropy_with_logits(logits.view(-1), labels.view(-1).type_as(logits))

        if not return_dict:
            output = (
                logits,
                outputs.hidden_states,
                outputs.entity_hidden_states,
                outputs.attentions,
            )
            return ((loss,) + output) if loss is not None else output

        return EntityPairClassificationOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            entity_hidden_states=outputs.entity_hidden_states,
            attentions=outputs.attentions,
        )
