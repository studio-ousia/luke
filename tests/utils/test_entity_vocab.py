import os
import pytest
import tempfile

from luke.utils.entity_vocab import EntityVocab

ENTITY_VOCAB_FIXTURE_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../fixtures/enwiki_20181220_entvocab_100.tsv"
)

MULTILINGUAL_ENTITY_VOCAB_FIXTURE_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../fixtures/en_ja_multilingual_vocab_test.jsonl"
)


@pytest.fixture
def entity_vocab():
    return EntityVocab(ENTITY_VOCAB_FIXTURE_FILE)


@pytest.fixture
def multilingual_entity_vocab():
    return EntityVocab(MULTILINGUAL_ENTITY_VOCAB_FIXTURE_FILE)


def test_entity_vocab(entity_vocab):
    assert len(entity_vocab) == 103
    assert len(list(entity_vocab)) == 103
    assert "United States" in entity_vocab
    assert entity_vocab["[PAD]"] == 0
    assert entity_vocab["United States"] == 4
    assert entity_vocab.get_id("United States") == 4
    assert entity_vocab.get_title_by_id(4) == "United States"
    assert entity_vocab.get_count_by_title("United States") == 261500


def test_multilingual_entity_vocab(multilingual_entity_vocab):
    assert len(multilingual_entity_vocab) == 6
    assert len(list(multilingual_entity_vocab)) == 9
    assert multilingual_entity_vocab.contains("フジテレビジョン", "ja")
    assert multilingual_entity_vocab.get_id("[MASK]", "ja") == 2
    assert multilingual_entity_vocab.get_id("[MASK]", "en") == 2
    assert multilingual_entity_vocab.get_id("フジテレビジョン", "ja") == 3
    assert multilingual_entity_vocab.get_title_by_id(3, "ja") == "フジテレビジョン"
    assert multilingual_entity_vocab.get_count_by_title("フジテレビジョン", "ja") == 142


def test_save_and_load(multilingual_entity_vocab):
    with tempfile.NamedTemporaryFile() as f:
        multilingual_entity_vocab.save(f.name)
        entity_vocab2 = EntityVocab(f.name)

        assert len(multilingual_entity_vocab) == len(entity_vocab2)

        # check if the two vocabs are identical after save and load
        for ent_id in range(len(multilingual_entity_vocab)):
            entities1 = multilingual_entity_vocab.inv_vocab[ent_id]
            entities2 = entity_vocab2.inv_vocab[ent_id]
            assert set(entities1) == set(entities2)
            assert multilingual_entity_vocab.counter[entities1[0]] == entity_vocab2.counter[entities2[0]]
            assert multilingual_entity_vocab.vocab[entities1[0]] == entity_vocab2.vocab[entities2[0]]
