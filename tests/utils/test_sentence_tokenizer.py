from luke.utils.sentence_tokenizer import OpenNLPSentenceTokenizer, ICUSentenceTokenizer

sent1 = "This is an English sentence. "
sent2 = "The non-BMP characters 🤩 should be handled properly."


def test_icu_span_tokenize():
    tokenizer = ICUSentenceTokenizer("en")
    assert tokenizer.span_tokenize("".join((sent1, sent2))) == [(0, 29), (29, 81)]


def test_opennlp_span_tokenize():
    tokenizer = OpenNLPSentenceTokenizer()
    assert tokenizer.span_tokenize("".join((sent1, sent2))) == [(0, 28), (29, 81)]
