from transformers import XLMRobertaTokenizer as OriginalXLMRobertaTokenizer


class XLMRobertaTokenizer(OriginalXLMRobertaTokenizer):
    """
        The original XLMRobertaTokenizer is broken, so fix that ourselves.
        (https://github.com/huggingface/transformers/issues/2976)
    """

    def __init__(self, vocab_file, **kwargs):
        super().__init__(vocab_file, **kwargs)
        self.fairseq_tokens_to_ids["<mask>"] = len(self.sp_model) + self.fairseq_offset
        self.fairseq_ids_to_tokens = {v: k for k, v in self.fairseq_tokens_to_ids.items()}

    @property
    def vocab_size(self):
        return len(self.sp_model) + self.fairseq_offset
