"""Unit tests for emotion analysis utilities."""

from __future__ import annotations

import pytest

from localtalk.utils.emotion import adjust_tts_parameters, analyze_emotion

pytestmark = pytest.mark.unit


# ────────────────────────── analyze_emotion ──────────────────────────


class TestAnalyzeEmotion:
    def test_neutral_text_returns_default(self):
        """Plain text with no emotional keywords or exclamations returns 0.5."""
        assert analyze_emotion("The table is brown.") == 0.5

    def test_keyword_matching_case_insensitive(self):
        score = analyze_emotion("I am HAPPY today")
        assert score == pytest.approx(0.6)  # 0.5 base + 0.1 for "happy"

    def test_multiple_keywords_accumulate(self):
        score = analyze_emotion("This is amazing and wonderful and fantastic")
        assert score == pytest.approx(0.8)  # 0.5 + 0.1 * 3

    def test_exclamation_contribution(self):
        score_no_excl = analyze_emotion("Wow")
        score_with_excl = analyze_emotion("Wow!")
        assert score_with_excl > score_no_excl
        assert score_with_excl == pytest.approx(0.65)  # 0.5 + 0.1 ("!") + 0.05 (one "!")

    def test_multiple_exclamations(self):
        score = analyze_emotion("Yes!!!")
        # 0.5 + 0.1 for "!" keyword + 0.05 * 3 for three "!" chars = 0.75
        assert score == pytest.approx(0.75)

    def test_score_capped_at_0_9(self):
        text = "amazing terrible love hate excited sad happy angry wonderful awful fantastic horrible great bad excellent poor !!!!!!"
        score = analyze_emotion(text)
        assert score == 0.9

    def test_score_floored_at_0_3(self):
        # The floor is 0.3, but with a base of 0.5 and only positive contributions,
        # the floor is not reachable through normal inputs.
        # Verify that the minimum is at least 0.3 (the clamp exists).
        score = analyze_emotion("")
        assert score >= 0.3
        assert score == pytest.approx(0.5)  # empty text → no keywords, base unchanged

    def test_qmark_exclam_keyword(self):
        """The '?!' keyword is checked as a substring."""
        score = analyze_emotion("What?!")
        # 0.5 + 0.1 for "?" substring match + 0.1 for "!" substring match + 0.1 for "?!" match + 0.05 for one "!"
        # Wait — "?" is not in the keyword list. Let me check.
        # Keywords: "!" and "?!" are in the list. "?" alone is not.
        # "what?!" contains "!" → +0.1, contains "?!" → +0.1, plus count("!") = 1 → +0.05
        assert score == pytest.approx(0.75)  # 0.5 + 0.1 + 0.1 + 0.05

    def test_ellipsis_keyword(self):
        score = analyze_emotion("I was thinking...")
        assert score == pytest.approx(0.6)  # 0.5 + 0.1 for "..." keyword


# ────────────────────── adjust_tts_parameters ──────────────────────


class TestAdjustTtsParameters:
    def test_neutral_text_keeps_base_exaggeration_blended(self):
        exag, cfg = adjust_tts_parameters("The table is brown.", base_exaggeration=0.5, base_cfg=0.5)
        # emotion_score = 0.5, adjusted_exag = 0.5*0.5 + 0.5*0.5 = 0.5
        assert exag == pytest.approx(0.5)
        # emotion_score <= 0.6 → cfg unchanged
        assert cfg == pytest.approx(0.5)

    def test_emotional_text_reduces_cfg(self):
        exag, cfg = adjust_tts_parameters("This is amazing!", base_exaggeration=0.5, base_cfg=0.5)
        # emotion_score = 0.5 + 0.1 (amazing) + 0.1 ("!") + 0.05 (one "!") = 0.75
        # adjusted_exag = 0.5*0.5 + 0.75*0.5 = 0.625
        assert exag == pytest.approx(0.625)
        # emotion_score > 0.6 → cfg = 0.5 * 0.8 = 0.4
        assert cfg == pytest.approx(0.4)

    def test_low_emotion_keeps_cfg(self):
        exag, cfg = adjust_tts_parameters("hello", base_exaggeration=0.5, base_cfg=0.5)
        # emotion_score = 0.5, not > 0.6
        assert cfg == pytest.approx(0.5)

    def test_high_emotion_caps_exaggeration(self):
        text = "amazing wonderful fantastic!!!"
        exag, cfg = adjust_tts_parameters(text, base_exaggeration=1.0, base_cfg=1.0)
        # emotion_score: 0.5 + 0.1*3 (keywords) + 0.1 ("!") + 0.05*3 ("!!!") = 1.0 → capped to 0.9
        # adjusted_exag = 1.0*0.5 + 0.9*0.5 = 0.95
        assert exag == pytest.approx(0.95)
        # cfg = 1.0 * 0.8 = 0.8
        assert cfg == pytest.approx(0.8)
