from transformers import BertTokenizer, RobertaTokenizer, PreTrainedTokenizer

from transformers.modeling_bert import (
    BertConfig,
    BertForPreTraining,
    BertPreTrainedModel,
)
from transformers.modeling_roberta import RobertaConfig, RobertaForMaskedLM
from transformers.modeling_xlm_roberta import XLMRobertaConfig, XLMRobertaForMaskedLM

from .tokenization_xlm_roberta_debugged import XLMRobertaTokenizerDebugged


def get_pretrained_model(mode_name: str) -> BertPreTrainedModel:
    if "xlm-roberta" in mode_name:
        return XLMRobertaForMaskedLM.from_pretrained(mode_name)
    elif "roberta" in mode_name:
        return RobertaForMaskedLM.from_pretrained(mode_name)
    elif "bert" in mode_name:
        return BertForPreTraining.from_pretrained(mode_name)
    else:
        raise NotImplementedError


def get_tokenizer(tokenizer_name: str) -> PreTrainedTokenizer:
    if "xlm-roberta" in tokenizer_name:
        return XLMRobertaTokenizerDebugged.from_pretrained(tokenizer_name)
    elif "roberta" in tokenizer_name:
        return RobertaTokenizer.from_pretrained(tokenizer_name)
    elif "bert" in tokenizer_name:
        return BertTokenizer.from_pretrained(tokenizer_name)
    else:
        raise NotImplementedError


def get_config(config_name: str):
    if "xlm-roberta" in config_name:
        return XLMRobertaConfig.from_pretrained(config_name)
    elif "roberta" in config_name:
        return RobertaConfig.from_pretrained(config_name)
    elif "bert" in config_name:
        return BertConfig.from_pretrained(config_name)
    else:
        raise NotImplementedError
