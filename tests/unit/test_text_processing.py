"""Unit tests for text processing utilities."""

from __future__ import annotations

import pytest

from localtalk.utils.text_processing import (
    chunk_text_for_streaming,
    clean_text_for_tts,
    get_first_sentence,
)

pytestmark = pytest.mark.unit


# ────────────────────────── clean_text_for_tts ──────────────────────────


class TestCleanTextForTts:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("**bold text**", "bold text"),
            ("*italic text*", "italic text"),
            # Triple asterisks: regex handles 1-2, leaving one asterisk on each side
            ("***both***", "*both*"),
            ("`code snippet`", "code snippet"),
            ("# Header", "Header"),
            ("### Sub header", "Sub header"),
            ("Check https://example.com out", "Check link out"),
            ("Visit http://test.org", "Visit link"),
            ("  extra   whitespace  ", "extra whitespace"),
            ("Hello **world**. Visit https://foo.com", "Hello world. Visit link"),
        ],
    )
    def test_clean_text_various_inputs(self, raw, expected):
        assert clean_text_for_tts(raw) == expected

    def test_clean_text_empty(self):
        assert clean_text_for_tts("") == ""

    def test_clean_text_no_markdown(self):
        assert clean_text_for_tts("Just plain text.") == "Just plain text."

    def test_clean_text_preserves_sentence_punctuation(self):
        assert clean_text_for_tts("Hello! How are you?") == "Hello! How are you?"


# ────────────────────────── get_first_sentence ──────────────────────────


class TestGetFirstSentence:
    def test_simple_period(self):
        first, rest = get_first_sentence("Hello world. How are you?")
        assert first == "Hello world."
        assert rest == "How are you?"

    def test_exclamation(self):
        # "Wow!" is < 10 chars, so get_first_sentence joins it with the next sentence
        first, rest = get_first_sentence("Wow! That is great.")
        assert first == "Wow! That is great."
        assert rest == ""

    def test_question_mark(self):
        first, rest = get_first_sentence("Are you sure? I think so.")
        assert first == "Are you sure?"
        assert rest == "I think so."

    def test_no_punctuation(self):
        first, rest = get_first_sentence("No ending here")
        assert first == "No ending here"
        assert rest == ""

    def test_short_first_sentence_joined(self):
        """A first sentence < 10 chars should be joined with the next one."""
        first, rest = get_first_sentence("Hi. How are you today? I am fine.")
        assert "Hi." in first
        assert "How are you today?" in first
        assert rest == "I am fine."

    def test_empty_string(self):
        first, rest = get_first_sentence("")
        assert first == ""
        assert rest == ""

    def test_single_sentence(self):
        first, rest = get_first_sentence("Just one sentence.")
        assert first == "Just one sentence."
        assert rest == ""

    def test_multiline_text(self):
        first, rest = get_first_sentence("First line.\nSecond line.")
        assert first == "First line."
        assert "Second line." in rest


# ─────────────────────── chunk_text_for_streaming ───────────────────────


class TestChunkTextForStreaming:
    def test_single_short_sentence(self):
        assert chunk_text_for_streaming("Hello world.") == ["Hello world."]

    def test_multiple_sentences_under_limit(self):
        text = "First sentence. Second sentence. Third one."
        assert chunk_text_for_streaming(text) == [text]

    def test_split_when_exceeding_max_chunk_size(self):
        # Each sentence has 5 words, max_chunk_size=5 → first fits, 5+5=10 > 5 so second splits
        text = "First sentence here right now. Second sentence here right now."
        chunks = chunk_text_for_streaming(text, max_chunk_size=5)
        assert len(chunks) == 2
        assert chunks[0] == "First sentence here right now."
        assert chunks[1] == "Second sentence here right now."

    def test_default_max_chunk_size(self):
        # Build text exceeding 50 words
        sentence = "One two three four five six."  # 6 words
        text = " ".join([sentence] * 12)  # 72 words total
        chunks = chunk_text_for_streaming(text)
        for chunk in chunks:
            assert len(chunk.split()) <= 50

    def test_empty_string(self):
        assert chunk_text_for_streaming("") == []

    def test_custom_max_chunk_size(self):
        text = "A B C. D E F. G H I."
        chunks = chunk_text_for_streaming(text, max_chunk_size=4)
        # Each sentence has 3 words; 3+3=6 > 4 → separate chunks
        assert len(chunks) == 3

    def test_sentence_longer_than_max_chunk_size(self):
        """A single sentence exceeding max_chunk_size is kept as one chunk (hard split not performed)."""
        long_sentence = " ".join(["word"] * 20)
        text = long_sentence + "."
        chunks = chunk_text_for_streaming(text, max_chunk_size=10)
        # The sentence is 21 words (including the period-attached last word), which exceeds 10
        # The function does not hard-split, so it becomes a single oversized chunk
        assert len(chunks) == 1
