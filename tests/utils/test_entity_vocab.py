
import bz2
import os
import pytest

from luke.utils.entity_vocab import EntityVocab

ENTITY_VOCAB_FIXTURE_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../fixtures/enwiki_20181220_entvocab_100.tsv"
)


@pytest.fixture
def entity_vocab():
    return EntityVocab(ENTITY_VOCAB_FIXTURE_FILE)


def test_entity_vocab(entity_vocab):
    assert len(entity_vocab) == 103
    assert len(list(entity_vocab)) == 103
    assert 'United States' in entity_vocab
    assert entity_vocab['[PAD]'] == 0
    assert entity_vocab['United States'] == 4
    assert entity_vocab.get_id('United States') == 4
    assert entity_vocab.get_title_by_id(4) == 'United States'
    assert entity_vocab.get_count_by_title('United States') == 261500
