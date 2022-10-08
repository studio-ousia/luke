import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from luke.utils.entity_vocab import EntityVocab, PAD_TOKEN, UNK_TOKEN

from dataset import Document, Mention


def create_dataloader(
    documents: List[Document],
    tokenizer: PreTrainedTokenizerBase,
    entity_vocab: EntityVocab,
    batch_size: int,
    fold: str,
    document_split_mode: str,
    max_seq_length: int,
    max_entity_length: int,
    max_mention_length: int,
    max_candidate_length: int,
) -> DataLoader:
    max_num_tokens = max_seq_length - 2

    input_dicts = []
    for document in documents:
        tokens = []
        mentions = []
        mention_spans = []
        cur = 0
        for mention in document.mentions:
            if fold == "train":
                # during training, we ignore a mention if its referent entity is not contained in
                # the candidate set or the vocabulary
                if all(mention.title != c.title for c in mention.candidates[:max_candidate_length]):
                    continue
                if mention.title not in entity_vocab:
                    continue

            tokens += tokenizer.tokenize(" ".join(document.words[cur : mention.start]))
            mention_tokens = tokenizer.tokenize(" ".join(document.words[mention.start : mention.end]))
            mentions.append(mention)
            mention_spans.append((len(tokens), len(tokens) + len(mention_tokens)))
            tokens += mention_tokens
            cur = mention.end
        tokens += tokenizer.tokenize(" ".join(document.words[cur:]))

        if len(tokens) <= max_num_tokens:
            input_dict = _create_input_dict(
                tokens=tokens,
                mentions=mentions,
                mention_spans=mention_spans,
                tokenizer=tokenizer,
                entity_vocab=entity_vocab,
                max_seq_length=max_seq_length,
                max_entity_length=max_entity_length,
                max_mention_length=max_mention_length,
                max_candidate_length=max_candidate_length,
            )
            input_dicts.append(input_dict)

        elif document_split_mode == "simple":
            token_mention_index_mapping = [None] * len(tokens)
            for n, span in enumerate(mention_spans):
                token_mention_index_mapping[span[0] : span[1]] = [n] * (span[1] - span[0])

            num_splits = math.ceil(len(tokens) / max_num_tokens)
            tokens_per_batch = math.ceil(len(tokens) / num_splits)
            doc_start = 0
            while True:
                doc_end = min(len(tokens), doc_start + tokens_per_batch)
                if fold != "train":
                    while True:
                        # a document should not be split inside a mention
                        if (
                            doc_end == len(tokens)
                            or not token_mention_index_mapping[doc_end - 1]
                            or (token_mention_index_mapping[doc_end - 1] != token_mention_index_mapping[doc_end])
                        ):
                            break
                        doc_end -= 1

                final_mentions, final_mention_spans = _get_mentions_and_spans_in_split_document(
                    mentions, mention_spans, doc_start, doc_end
                )
                input_dict = _create_input_dict(
                    tokens=tokens[doc_start:doc_end],
                    mentions=final_mentions,
                    mention_spans=final_mention_spans,
                    tokenizer=tokenizer,
                    entity_vocab=entity_vocab,
                    max_seq_length=max_seq_length,
                    max_entity_length=max_entity_length,
                    max_mention_length=max_mention_length,
                    max_candidate_length=max_candidate_length,
                )
                input_dicts.append(input_dict)

                if doc_end == len(tokens):
                    break
                doc_start = doc_end

        elif document_split_mode == "per_mention":
            for mention_index, (mention_start, mention_end) in enumerate(mention_spans):
                left_token_length = mention_start
                right_token_length = len(tokens) - mention_end
                mention_length = mention_end - mention_start
                half_context_size = int((max_num_tokens - mention_length) / 2)
                if left_token_length < right_token_length:
                    left_context_length = min(left_token_length, half_context_size)
                    right_context_length = min(
                        right_token_length, max_num_tokens - left_context_length - mention_length
                    )
                else:
                    right_context_length = min(right_token_length, half_context_size)
                    left_context_length = min(left_token_length, max_num_tokens - right_context_length - mention_length)
                doc_start = mention_start - left_context_length
                doc_end = mention_end + right_context_length

                ordered_mentions = [mentions[mention_index]] + mentions[:mention_index] + mentions[mention_index + 1 :]
                ordered_mention_spans = (
                    [mention_spans[mention_index]] + mention_spans[:mention_index] + mention_spans[mention_index + 1 :]
                )
                final_mentions, final_mention_spans = _get_mentions_and_spans_in_split_document(
                    ordered_mentions, ordered_mention_spans, doc_start, doc_end
                )
                input_dict = _create_input_dict(
                    tokens=tokens[doc_start:doc_end],
                    mentions=final_mentions,
                    mention_spans=final_mention_spans,
                    tokenizer=tokenizer,
                    entity_vocab=entity_vocab,
                    max_seq_length=max_seq_length,
                    max_entity_length=max_entity_length,
                    max_mention_length=max_mention_length,
                    max_candidate_length=max_candidate_length,
                    eval_mention_indices=[0],
                )
                input_dicts.append(input_dict)

        else:
            raise RuntimeError(f"Invalid document split mode: {document_split_mode}")

    if fold == "train":
        data_loader = DataLoader(input_dicts, batch_size=batch_size, collate_fn=_data_collator, shuffle=True)
    else:
        data_loader = DataLoader(input_dicts, batch_size=batch_size, collate_fn=_data_collator, shuffle=False)

    return data_loader


def _get_mentions_and_spans_in_split_document(
    mentions: List[Mention], mention_spans: List[Tuple[int, int]], doc_start: int, doc_end: int
) -> Tuple[List[Mention], List[Tuple[int, int]]]:
    new_mentions = []
    new_mention_spans = []
    for mention, (mention_start, mention_end) in zip(mentions, mention_spans):
        if mention_start >= doc_start and mention_end <= doc_end:
            new_mentions.append(mention)
            new_mention_spans.append((mention_start - doc_start, mention_end - doc_start))
    return (new_mentions, new_mention_spans)


def _create_input_dict(
    tokens: List[str],
    mentions: List[Mention],
    mention_spans: List[Tuple[int, int]],
    tokenizer: PreTrainedTokenizerBase,
    entity_vocab: EntityVocab,
    max_seq_length: int,
    max_entity_length: int,
    max_mention_length: int,
    max_candidate_length: int,
    eval_mention_indices: Optional[List[int]] = None,
) -> Dict[str, np.ndarray]:
    input_tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]
    input_ids = np.full(max_seq_length, tokenizer.pad_token_id, dtype=np.int)
    input_ids[: len(input_tokens)] = tokenizer.convert_tokens_to_ids(input_tokens)
    attention_mask = np.zeros(max_seq_length, dtype=np.int)
    attention_mask[: len(input_tokens)] = 1

    pad_entity_id = entity_vocab[PAD_TOKEN]
    unk_entity_id = entity_vocab[UNK_TOKEN]
    entity_ids = np.full(max_entity_length, pad_entity_id, dtype=np.int)
    entity_attention_mask = np.zeros(max_entity_length, dtype=np.int)
    entity_position_ids = np.full((max_entity_length, max_mention_length), -1, dtype=np.int)
    entity_candidate_ids = np.zeros((max_entity_length, max_candidate_length), dtype=np.int)
    eval_entity_mask = np.zeros(max_entity_length, dtype=np.int)

    for index, (mention, (mention_start, mention_end)) in enumerate(zip(mentions, mention_spans)):
        entity_ids[index] = entity_vocab.get_id(mention.title, default=unk_entity_id)
        entity_attention_mask[index] = 1
        if eval_mention_indices is None or index in eval_mention_indices:
            eval_entity_mask[index] = 1
        entity_position_ids[index][: mention_end - mention_start] = range(
            mention_start + 1, mention_end + 1
        )  # +1 for [CLS]
        candidate_ids = [
            entity_vocab.get_id(candidate.title, default=unk_entity_id)
            for candidate in mention.candidates[:max_candidate_length]
        ]
        entity_candidate_ids[index, : len(candidate_ids)] = candidate_ids

    return dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        entity_ids=entity_ids,
        entity_attention_mask=entity_attention_mask,
        entity_position_ids=entity_position_ids,
        entity_candidate_ids=entity_candidate_ids,
        eval_entity_mask=eval_entity_mask,
    )


def _data_collator(input_dicts: List[Dict[str, np.ndarray]]) -> Dict[str, torch.Tensor]:
    max_subword_length = max(input_dict["attention_mask"].sum() for input_dict in input_dicts)
    max_entity_length = max(input_dict["entity_attention_mask"].sum() for input_dict in input_dicts)
    batch = {}
    for feature_name in input_dicts[0].keys():
        if "entity" in feature_name:
            batch[feature_name] = torch.tensor(np.stack(input_dict[feature_name] for input_dict in input_dicts))[
                :, :max_entity_length
            ]
        else:
            batch[feature_name] = torch.tensor(np.stack(input_dict[feature_name] for input_dict in input_dicts))[
                :, :max_subword_length
            ]
    return batch
