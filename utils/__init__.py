# -*- coding: utf-8 -*-

# The following code was obtained from https://github.com/google-research/bert/blob/f39e881b169b9d53bea03d2d341b31707a6c052b/tokenization.py

import unicodedata


def clean_text(text, strip_accents=False):
    """Performs invalid character removal and whitespace cleanup on text."""
    output = []
    for char in text:
        cp = ord(char)
        if cp == 0 or cp == 0xfffd or _is_control(char):
            continue
        if _is_whitespace(char):
            output.append(" ")
        else:
            output.append(char)

    text = "".join(output)

    if strip_accents:
        text = _strip_accents(text)

    return text


def _strip_accents(text):
    """Strips accents from a piece of text."""
    text = unicodedata.normalize("NFD", text)
    output = []

    for char in text:
        cat = unicodedata.category(char)
        if cat == "Mn":
            continue
        output.append(char)

    return "".join(output)


def _is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False
