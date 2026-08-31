"""Offline wiring tests — no live API/network calls. Run: pytest tests/"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from humanizers import baseline
from humanizers._shared import cap_words, split_sentences, word_count


def test_baseline_passthrough():
    assert baseline.humanize("hello world") == "hello world"


def test_cap_words_under_limit():
    text = "one two three"
    assert cap_words(text, max_words=10) == text


def test_cap_words_over_limit():
    text = " ".join(f"w{i}" for i in range(250))
    capped = cap_words(text, max_words=200)
    assert word_count(capped) == 200


def test_split_sentences_basic():
    text = "This is one. This is two! Is this three?"
    sents = split_sentences(text)
    assert sents == ["This is one.", "This is two!", "Is this three?"]


def test_word_count():
    assert word_count("a b c") == 3
    assert word_count("") == 0
