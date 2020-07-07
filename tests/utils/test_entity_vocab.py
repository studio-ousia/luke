import os
import pytest

from luke.utils.entity_vocab import EntityVocab

ENTITY_VOCAB_FIXTURE_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../fixtures/enwiki_20181220_entvocab_100.tsv"
)

MULTILINGUAL_ENTITY_VOCAB_FIXTURE_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../fixtures/en_ja_multilingual_vocab_test.json"
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
