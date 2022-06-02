from typing import List, Tuple
import os

import click
from transformers import AutoTokenizer, PreTrainedTokenizer
from wikipedia2vec.dump_db import DumpDB

from luke.utils.entity_vocab import EntityVocab
from luke.utils.model_utils import ENTITY_VOCAB_FILE
from luke.utils.sentence_splitter import SentenceSplitter
from luke.pretraining.dataset import (
    get_paragraph_links,
    links_to_link_ids,
    get_sentence_words_and_links,
)


def extract_span_and_context(
    word_ids: List[int], span_start: int, span_end: int, max_segment_length: int
) -> Tuple[List[int], Tuple[int, int]]:
    entity_length = span_end - span_start + 1
    sentence_length = len(word_ids)
    context_segments_length = max_segment_length - entity_length

    max_left_context_length = span_start
    max_right_context_length = sentence_length - span_end - 1

    if max_left_context_length < max_right_context_length:
        left_context_length = min(max_left_context_length, context_segments_length // 2)
        right_context_length = min(context_segments_length - left_context_length, max_right_context_length)
    else:
        right_context_length = min(max_right_context_length, context_segments_length // 2)
        left_context_length = min(context_segments_length - right_context_length, max_left_context_length)

    span_and_context = word_ids[span_start - left_context_length : span_end + right_context_length + 1]
    new_span = (left_context_length, len(span_and_context) - right_context_length - 1)
    return span_and_context, new_span


def process_page(
    page_title: str,
    dump_db: DumpDB,
    entity_vocab: EntityVocab,
    tokenizer: PreTrainedTokenizer,
    sentence_splitter: SentenceSplitter,
    min_segment_length: int,
    max_segment_length: int,
    max_mention_length: int,
):
    sentence_words_and_links = []
    for paragraph in dump_db.get_paragraphs(page_title):

        # First, get paragraph links.
        # Paragraph links are represented its form (link_title) and the start/end positions of strings
        # (link_start, link_end).
        paragraph_text, paragraph_links = get_paragraph_links(dump_db, paragraph)

        sentence_words_and_links += get_sentence_words_and_links(
            paragraph_text=paragraph_text,
            paragraph_links=paragraph_links,
            sentence_splitter=sentence_splitter,
            tokenizer=tokenizer,
            min_sentence_length=min_segment_length,
            max_num_tokens=512,
        )

    for words, links in sentence_words_and_links:
        links_ids = links_to_link_ids(
            links, entity_vocab=entity_vocab, include_unk_entities=False, language=dump_db.language
        )

        if not links_ids:
            continue

        sentence = tokenizer.convert_tokens_to_ids(words)
        assert min_segment_length <= len(sentence)

        for entity_id, link_start, link_end in links_ids:
            if (link_end - link_start + 1) > max_mention_length:
                continue

            word_ids = sentence
            if len(word_ids) > max_segment_length:
                word_ids, (link_start, link_end) = extract_span_and_context(sentence, link_start, link_end)
            assert len(word_ids) <= max_segment_length

            entity_position_ids = list(range(link_start, link_end))
            yield {"word_ids": word_ids, "entity_id": entity_id, "entity_position_ids": entity_position_ids}


@click.command()
@click.argument("dump_db_file", type=click.Path(exists=True))
@click.argument("tokenizer_name")
@click.argument("entity_vocab_file", type=str)
@click.argument("output_dir", type=click.Path(file_okay=False))
@click.option("--sentence-splitter", default="en")
@click.option("--max-segment-length", default=50)
@click.option("--max-mention-length", default=16)
@click.option("--min-segment-length", default=10)
def build_wikipedia_pretraining_dataset(
    dump_db_file: str,
    tokenizer_name: str,
    entity_vocab_file: str,
    output_dir: str,
    sentence_splitter: str,
    max_segment_length: int,
    max_mention_length: int,
    min_segment_length: int,
):
    dump_db = DumpDB(dump_db_file)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)
    sentence_splitter = SentenceSplitter.from_name(sentence_splitter)
    entity_vocab = EntityVocab(entity_vocab_file)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    tokenizer.save_pretrained(output_dir)
    entity_vocab.save(os.path.join(output_dir, ENTITY_VOCAB_FILE))

    target_titles = [
        title
        for title in dump_db.titles()
        if not (":" in title and title.lower().split(":")[0] in ("image", "file", "category"))
    ]

    for title in target_titles:
        for item in process_page(
            title,
            dump_db,
            entity_vocab=entity_vocab,
            tokenizer=tokenizer,
            sentence_splitter=sentence_splitter,
            min_segment_length=min_segment_length,
            max_segment_length=max_segment_length,
            max_mention_length=max_mention_length,
        ):
            breakpoint()
