from transformers import AutoTokenizer

from luke.pretraining.tokenization import tokenize, tokenize_segments


def test_tokenize_with_roberta():

    tokenizer = AutoTokenizer.from_pretrained("roberta-base", use_fast=False)

    assert tokenize("Tokyo", tokenizer, add_prefix_space=True) == ["ĠTokyo"]
    assert tokenize("Tokyo", tokenizer, add_prefix_space=False) == ["Tok", "yo"]

    assert tokenize_segments(["Studio Ousia", " developed "], tokenizer, add_prefix_space=True) == [
        ["ĠStudio", "ĠO", "us", "ia"],
        ["Ġdeveloped"],
    ]
    assert tokenize_segments(["LUKE", "."], tokenizer, add_prefix_space=True) == [["ĠLU", "KE"], ["."]]


def test_tokenize_with_xlm_roberta():
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base", use_fast=False)

    assert tokenize("Tokyo", tokenizer, add_prefix_space=True) == ["▁Tokyo"]
    assert tokenize("Tokyo", tokenizer, add_prefix_space=False) == ["To", "ky", "o"]

    assert tokenize_segments(["私は", "寿司"], tokenizer) == [["▁私は"], ["寿", "司"]]
    assert tokenize_segments(["を食べた。"], tokenizer, add_prefix_space=False) == [["を", "食べた", "。"]]


def test_tokenize_with_bert():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased", use_fast=False)

    assert tokenize("Tokyo", tokenizer, add_prefix_space=False) == ["Tokyo"]
    assert tokenize("Tokyo", tokenizer, add_prefix_space=True) == ["Tokyo"]
    assert tokenize_segments(["Studio Ousia", " is cool."], tokenizer) == [
        ["Studio", "O", "##us", "##ia"],
        ["is", "cool", "."],
    ]
