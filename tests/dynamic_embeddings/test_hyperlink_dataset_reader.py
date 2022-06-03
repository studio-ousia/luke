from dynamic_embeddings.hyperlink_dataset_reader import HyperlinkDatasetReader
import numpy as np


def decode_word_ids(word_ids: np.ndarray, tokenizer) -> str:
    return tokenizer.decode(word_ids[word_ids > -1])


def extract_entity(word_ids: np.ndarray, entity_position_ids: np.ndarray, tokenizer) -> str:
    return tokenizer.decode(word_ids[entity_position_ids[entity_position_ids > -1]])


def test_text_to_instance():
    reader = HyperlinkDatasetReader("studio-ousia/luke-base")
    tokenizer = reader.transformers_tokenizer

    # check inputs
    word_ids = np.array(
        [
            [494, 706, 126, 15400, 975, 6905, 8255, 49, 2453, 2642, 15400, 975, 6905, -1, -1, -1, -1, -1, -1, -1, -1],
            [85, 21, 703, 25, 5, 371, 881, 31, 49, 1403, 12, 90, 22764, 2453, 2642, 11, 5, 121, 4, 104, 4],
        ]
    )
    assert decode_word_ids(word_ids[0], tokenizer) == " March 24 – NSYNC releases their debut album NSYNC"
    assert (
        decode_word_ids(word_ids[1], tokenizer)
        == " It was released as the third single from their self-titled debut album in the U.S."
    )

    entity_position_ids = np.array(
        [
            [10, 11, 12, -1, -1, -1, -1, -1],
            [8, 9, 10, 11, 12, 13, 14, -1],
        ]
    )

    assert extract_entity(word_ids[0], entity_position_ids[0], tokenizer) == " NSYNC"
    assert extract_entity(word_ids[1], entity_position_ids[1], tokenizer) == " their self-titled debut album"

    # test the dataset reader
    instance = reader.text_to_instance("'N Sync (album)", word_ids, entity_position_ids)
    word_ids = instance["word_ids"].tensor
    assert extract_entity(word_ids, instance["entity_position_ids"].tensor[0], tokenizer) == " NSYNC"
    assert (
        extract_entity(word_ids, instance["entity_position_ids"].tensor[1], tokenizer)
        == " their self-titled debut album"
    )
