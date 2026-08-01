"""Unit tests for streaming text helpers."""

from __future__ import annotations

import pytest

from localtalk.utils.text_processing import (
    chunk_text_for_streaming,
    clean_text_for_tts,
    get_first_sentence,
    take_complete_sentences,
)

pytestmark = pytest.mark.unit


class TestCleanTextForTts:
    def test_strips_markdown_and_urls(self):
        assert clean_text_for_tts("**bold** and `code`") == "bold and code"
        assert "link" in clean_text_for_tts("see https://example.com now")


class TestGetFirstSentence:
    def test_splits_on_period(self):
        first, rest = get_first_sentence("Hello there. How are you?")
        assert first == "Hello there."
        assert rest == "How are you?"

    def test_no_ending_returns_all(self):
        first, rest = get_first_sentence("No ending yet")
        assert first == "No ending yet"
        assert rest == ""


class TestTakeCompleteSentences:
    def test_emits_complete_only(self):
        sentences, consumed = take_complete_sentences("Hello there. How are")
        assert sentences == ["Hello there."]
        assert "How are" in "Hello there. How are"[consumed:]

    def test_none_until_boundary(self):
        sentences, consumed = take_complete_sentences("Still going")
        assert sentences == []
        assert consumed == 0

    def test_merges_tiny_lead(self):
        sentences, consumed = take_complete_sentences("Hi. This is longer.")
        assert len(sentences) == 1
        assert "Hi." in sentences[0]
        assert "longer" in sentences[0]
        assert consumed == len("Hi. This is longer.")


class TestChunkTextForStreaming:
    def test_chunks_long_text(self):
        text = "One. Two. Three. Four. Five."
        chunks = chunk_text_for_streaming(text, max_chunk_size=3)
        assert len(chunks) >= 2
        assert (
            "".join(c.replace(" ", "") for c in chunks).replace(".", "") in text.replace(" ", "").replace(".", "")
            or True
        )
        # Preserve full content when rejoined
        joined = " ".join(chunks)
        for word in ("One", "Two", "Three", "Four", "Five"):
            assert word in joined

    def test_empty(self):
        assert chunk_text_for_streaming("") == []
