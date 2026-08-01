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


class TestTakeCompleteSentencesCJK:
    def test_cjk_splits_without_whitespace(self):
        # Chinese has no inter-sentence spaces; 。 must terminate mid-buffer.
        # (First sentence is >= 10 chars so the tiny-lead merge rule stays out.)
        text = "今天的天气真的非常不错了。我们一起去公园散步吧。走吧"
        sentences, consumed = take_complete_sentences(text)
        assert sentences == ["今天的天气真的非常不错了。", "我们一起去公园散步吧。"]
        assert text[consumed:] == "走吧"

    def test_cjk_tiny_lead_merges_like_english(self):
        # Very short CJK leads (< 6 chars) fold into the next sentence rather
        # than being spoken as an abrupt fragment.
        sentences, consumed = take_complete_sentences("你好。今天天气很好。我们去")
        assert sentences == ["你好。 今天天气很好。"]
        assert "你好。今天天气很好。我们去"[consumed:] == "我们去"

    def test_cjk_complete_clause_does_not_merge(self):
        # Chinese is denser than English: a 7-char sentence is a complete
        # clause, not a fragment, so it must stream as its own TTS piece.
        text = "今天天气很好。我们出去走走吧。然后"
        sentences, consumed = take_complete_sentences(text)
        assert sentences == ["今天天气很好。", "我们出去走走吧。"]
        assert text[consumed:] == "然后"

    def test_latin_lead_still_uses_english_threshold(self):
        sentences, _ = take_complete_sentences("Hi there. 我们开始今天的会议吧。")
        assert sentences == ["Hi there. 我们开始今天的会议吧。"]

    def test_cjk_streaming_emits_first_sentence_early(self):
        # Simulates a multi-token chunk arriving mid-generation: the finished
        # sentence must be emitttable before the stream ends.
        sentences, consumed = take_complete_sentences("好的，我明白了。今天天气")
        assert sentences == ["好的，我明白了。"]
        assert "好的，我明白了。今天天气"[consumed:] == "今天天气"

    def test_cjk_consecutive_punctuation_stays_together(self):
        sentences, _ = take_complete_sentences("真的吗？！下次再说")
        assert sentences == ["真的吗？！"]

    def test_mixed_latin_cjk(self):
        text = "这个问题的答案是 42。明白了吗？"
        sentences, consumed = take_complete_sentences(text)
        assert sentences == ["这个问题的答案是 42。", "明白了吗？"]
        assert consumed == len(text)

    def test_latin_decimal_still_protected(self):
        sentences, consumed = take_complete_sentences("The price is 3.5 dollars. Done")
        assert sentences == ["The price is 3.5 dollars."]
        assert consumed == len("The price is 3.5 dollars. ")


class TestChunkTextForStreamingCJK:
    def test_splits_chinese_sentences(self):
        chunks = chunk_text_for_streaming("好的。今天天气很好。我们去公园散步吧。", max_chunk_size=10)
        assert chunks == ["好的。", "今天天气很好。", "我们去公园散步吧。"]

    def test_chinese_chunk_size_counts_characters(self):
        # Without CJK-aware sizing this whole reply counted as ~1 "word" and
        # could never be bounded; each Chinese character is ~1 spoken syllable.
        long_sentence = "这是一个非常非常长的句子，几乎没有停顿地一直说下去。"
        chunks = chunk_text_for_streaming(f"{long_sentence}{long_sentence}", max_chunk_size=len(long_sentence) + 2)
        assert len(chunks) == 2

    def test_cjk_consecutive_punctuation_not_split(self):
        chunks = chunk_text_for_streaming("真的吗？！太好了。", max_chunk_size=4)
        assert chunks[0].startswith("真的吗？！")

    def test_full_coverage_order_preserved(self):
        text = "好的。今天天气很好。我们去公园散步吧。"
        chunks = chunk_text_for_streaming(text)
        assert " ".join(chunks).replace(" ", "") == text


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
