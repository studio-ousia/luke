from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.tokenizers.pretrained_transformer_tokenizer import PretrainedTransformerTokenizer

from examples.entity_typing.reader import EntityTypingReader
from examples.utils.util import ENT

extra_tokens = [ENT]
# transformer_model_name = "roberta-base"
transformer_model_name = "bert-base-multilingual-cased"
tokenizer = PretrainedTransformerTokenizer(
    transformer_model_name, tokenizer_kwargs={"additional_special_tokens": extra_tokens}
)
token_indexers = {
    "tokens": PretrainedTransformerIndexer(
        transformer_model_name, tokenizer_kwargs={"additional_special_tokens": extra_tokens}
    )
}

test_data_path = "examples/entity_typing/tests/fixtures/test.json"


def test_read():

    reader = EntityTypingReader(tokenizer, token_indexers)

    instances = [i for i in reader.read(test_data_path)]
    assert len(instances) == 1

    instance = instances[0]
    expected = [
        "[CLS]",
        "On",
        "<ent>",
        "late",
        "Monday",
        "night",
        "<ent>",
        ",",
        "30th",
        "Nov",
        "2009",
        ",",
        "Bangladesh",
        "Police",
        "arrested",
        "Raj",
        "##kh",
        "##owa",
        "some",
        "##where",
        "near",
        "Dhaka",
        ".",
        "[SEP]",
    ]
    assert [t.text for t in instance["word_ids"]] == expected
    assert instance["labels"].labels == ["time"]
    assert instance["entity_span"].span_start == 2
    assert instance["entity_span"].span_end == 6


def test_read_entity():

    reader = EntityTypingReader(tokenizer, token_indexers, use_entity_feature=True)

    instances = [i for i in reader.read(test_data_path)]
    assert len(instances) == 1

    instance = instances[0]

    expected = [
        "[CLS]",
        "On",
        "late",
        "Monday",
        "night",
        ",",
        "30th",
        "Nov",
        "2009",
        ",",
        "Bangladesh",
        "Police",
        "arrested",
        "Raj",
        "##kh",
        "##owa",
        "some",
        "##where",
        "near",
        "Dhaka",
        ".",
        "[SEP]",
    ]
    assert [t.text for t in instance["word_ids"]] == expected
    assert instance["labels"].labels == ["time"]
    assert instance["entity_span"].span_start == 2
    assert instance["entity_span"].span_end == 4
