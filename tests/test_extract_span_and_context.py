from create_hyper_links_dataset import extract_span_and_context


def test_left_truncated_sequence():
    word_ids = list(range(10))
    span_start = 5
    span_end = 8
    max_segment_length = 7
    assert ([3, 4, 5, 6, 7, 8, 9], (2, 5)) == extract_span_and_context(
        word_ids, span_start, span_end, max_segment_length
    )


def test_right_truncated_sequence():
    word_ids = list(range(10))
    span_start = 1
    span_end = 3
    max_segment_length = 7
    assert ([0, 1, 2, 3, 4, 5, 6], (1, 3)) == extract_span_and_context(
        word_ids, span_start, span_end, max_segment_length
    )


def test_truncated_sequence():
    word_ids = list(range(10))
    span_start = 3
    span_end = 5
    max_segment_length = 5
    assert ([2, 3, 4, 5, 6], (1, 3)) == extract_span_and_context(word_ids, span_start, span_end, max_segment_length)


def test_short_sequence():
    word_ids = list(range(5))
    span_start = 1
    span_end = 3
    max_segment_length = 7
    assert ([0, 1, 2, 3, 4], (1, 3)) == extract_span_and_context(word_ids, span_start, span_end, max_segment_length)
