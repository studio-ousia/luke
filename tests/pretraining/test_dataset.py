import tempfile

from transformers import AutoTokenizer

from luke.pretraining.dataset import WikipediaPretrainingDataset
from luke.utils.entity_vocab import EntityVocab
from luke.utils.sentence_splitter import SentenceSplitter

from .dummy_dump_db import DummyDumpDB


def test_build_and_read_dataset():
    dummy_dump_db = DummyDumpDB()

    tokenizer_name = "roberta-base"
    sentence_tokenizer = "icu"
    entity_vocab_file = "tests/fixtures/dummy_entity_vocab.tsv"

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)
    sentence_tokenizer = SentenceSplitter.from_name(sentence_tokenizer)

    entity_vocab = EntityVocab(entity_vocab_file)

    with tempfile.TemporaryDirectory() as temp_directory_path:

        WikipediaPretrainingDataset.build(
            dummy_dump_db,
            tokenizer,
            sentence_tokenizer,
            entity_vocab,
            temp_directory_path,
            language=None,
            max_seq_length=512,
            max_entity_length=128,
            max_mention_length=30,
            min_sentence_length=5,
            abstract_only=False,
            include_sentences_without_entities=True,
            include_unk_entities=True,
            pool_size=1,
            chunk_size=1,
            max_num_documents=None,
            predefined_entities_only=False,
        )

        dataset = WikipediaPretrainingDataset(temp_directory_path)
        entity_vocab = dataset.entity_vocab
        tokenizer = dataset.tokenizer
        items = [item for item in dataset.create_iterator(shuffle=False, repeat=False)]
        # the order of the items seems to be stochastic due to multiprocessing in build
        # so we sort the items here
        items = sorted(items, key=lambda x: x["page_id"])

        item = items[0]
        assert entity_vocab.get_title_by_id(item["page_id"]) == "Japan"
        assert (
            tokenizer.decode(item["word_ids"]).strip()
            == "Japan is an island country in East Asia. It is situated in the northwest Pacific Ocean."
        )
        for entity_id, entity_position_ids, expected_title, expected_mention in zip(
            item["entity_ids"],
            item["entity_position_ids"],
            ["Japan", "East Asia", "Pacific Ocean"],
            ["Japan", "East Asia", "Pacific Ocean"],
        ):
            assert entity_vocab.get_title_by_id(entity_id) == expected_title
            assert (
                tokenizer.decode(item["word_ids"][[i for i in entity_position_ids if i > -1]]).strip()
                == expected_mention
            )

        item = items[1]
        assert entity_vocab.get_title_by_id(item["page_id"]) == "Studio Ousia"
        assert (
            tokenizer.decode(item["word_ids"]).strip()
            == "Studio Ousia develops advanced multilingual natural language AI. Our award-winning AI will accelerate your business."
        )
        for entity_id, entity_position_ids, expected_title, expected_mention in zip(
            item["entity_ids"],
            item["entity_position_ids"],
            ["Studio Ousia", "Artificial Intelligence", "Artificial Intelligence"],
            ["Studio Ousia", "AI", "AI"],
        ):
            assert entity_vocab.get_title_by_id(entity_id) == expected_title
            assert (
                tokenizer.decode(item["word_ids"][[i for i in entity_position_ids if i > -1]]).strip()
                == expected_mention
            )
