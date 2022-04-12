from luke.utils.sentence_splitter import ICUSentenceSplitter, OpenNLPSentenceSplitter

sent1 = "This is an English sentence. "
sent2 = "The non-BMP characters 🤩 should be handled properly."


def test_icu_span_tokenize():
    tokenizer = ICUSentenceSplitter("en")
    assert tokenizer.get_sentence_spans("".join((sent1, sent2))) == [(0, 29), (29, 81)]


def test_opennlp_span_tokenize():
    tokenizer = OpenNLPSentenceSplitter()
    assert tokenizer.get_sentence_spans("".join((sent1, sent2))) == [(0, 28), (29, 81)]
