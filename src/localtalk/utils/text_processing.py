"""Text processing utilities for streaming TTS and speakable output."""

from __future__ import annotations

import re

import mistune
from mistune.renderers.html import HTMLRenderer

# Sentence boundary: Latin punctuation only terminates a sentence when followed
# by whitespace or end-of-string (protects decimals like "3.5"); CJK punctuation
# (。！？) terminates on its own because Chinese has no inter-sentence spaces —
# requiring whitespace would prevent streamed Chinese from ever splitting
# mid-response.
_SENTENCE_END = re.compile(r"[.!?]+(?:\s+|$)|[。！？]+\s*")

# CJK ideographs plus common full-width punctuation. Chinese text has no
# spaces, so ``str.split`` undercounts its spoken length; each CJK character
# is roughly one spoken syllable.
_CJK_CHARS = re.compile(r"[㐀-䶿一-鿿豈-﫿。！？，、；：]")


def _spoken_units(text: str) -> int:
    """Approximate spoken length: whitespace-separated words plus CJK characters."""
    return len(text.split()) + len(_CJK_CHARS.findall(text))


# A tiny leading sentence is folded into the next one so TTS doesn't speak an
# abrupt fragment. Chinese is denser than English — a few characters already
# form a complete clause — so the bar is lower for CJK-dominant sentences.
_TINY_LEAD_CHARS = 10
_CJK_TINY_LEAD_CHARS = 6


def _tiny_lead_threshold(sentence: str) -> int:
    """Merge threshold for a leading fragment; lower for CJK-dominant text."""
    cjk = len(_CJK_CHARS.findall(sentence))
    return _CJK_TINY_LEAD_CHARS if cjk * 2 >= len(sentence) else _TINY_LEAD_CHARS


def clean_text_for_tts(text: str) -> str:
    """Clean text for better TTS output (lightweight, no mistune).

    Prefer :func:`strip_markdown` when the model may emit full markdown;
    this handles the common inline cases quickly.
    """
    text = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"#{1,6}\s+", "", text)
    text = re.sub(r"https?://\S+", "link", text)
    return " ".join(text.split())


def get_first_sentence(text: str) -> tuple[str, str]:
    """Extract the first sentence for immediate TTS playback.

    Uses the same boundary rules as :func:`take_complete_sentences` (Latin
    punctuation only terminates when followed by whitespace/end; CJK marks
    terminate alone; tiny leads merge). Returns ``(text, "")`` when no
    complete sentence boundary is found.
    """
    text = text.strip()
    if not text:
        return "", ""

    sentences, consumed = take_complete_sentences(text)
    if not sentences:
        return text, ""

    first_sentence = sentences[0]
    remaining = text[consumed:].strip()
    # When tiny-lead merge consumed two sentences, remaining still starts after
    # the full consumed span; if more complete sentences were in the buffer,
    # rejoin them into remaining so callers still see a single first + rest.
    if len(sentences) > 1:
        remaining = " ".join([*sentences[1:], remaining]).strip() if remaining else " ".join(sentences[1:])
    return first_sentence, remaining


def take_complete_sentences(buffer: str) -> tuple[list[str], int]:
    """Split *buffer* into complete sentences and the consume index.

    A sentence is complete when it ends with Latin ``.``, ``!``, or ``?``
    followed by whitespace/end-of-buffer, or with CJK ``。``, ``！``, or ``？``
    (no trailing whitespace required, so streamed Chinese splits mid-response).
    Very short leading sentences (< 10 chars, or < 6 for CJK-dominant text)
    are merged with the next one when possible (same rule as
    :func:`get_first_sentence`).

    Returns:
        ``(sentences, consumed)`` where ``consumed`` is the index into *buffer*
        after the last complete sentence (0 if none). Remainder is
        ``buffer[consumed:]``.
    """
    if not buffer:
        return [], 0

    sentences: list[str] = []
    last = 0
    for match in _SENTENCE_END.finditer(buffer):
        end = match.end()
        piece = buffer[last:end].strip()
        if piece:
            sentences.append(piece)
        last = end

    if not sentences:
        return [], 0

    # Merge a tiny first sentence into the next when both exist.
    if len(sentences) >= 2 and len(sentences[0]) < _tiny_lead_threshold(sentences[0]):
        merged = f"{sentences[0]} {sentences[1]}"
        sentences = [merged, *sentences[2:]]

    return sentences, last


def chunk_text_for_streaming(text: str, max_chunk_size: int = 40) -> list[str]:
    """Split text into sentence-aware chunks for streaming TTS.

    Args:
        text: Text to chunk.
        max_chunk_size: Soft maximum spoken units per chunk (sentences may
            exceed). English counts whitespace-separated words; CJK characters
            count one unit each since Chinese has no inter-word spaces.

    Returns:
        Non-empty chunks covering the full text order.
    """
    text = text.strip()
    if not text:
        return []

    # Latin punctuation splits only at a following space (protects "3.5");
    # CJK punctuation splits anywhere (no inter-sentence spaces in Chinese),
    # keeping consecutive CJK marks (？！) attached to their sentence.
    sentences = re.split(r"(?<=[.!?])\s+|(?<=[。！？])(?![。！？])", text)
    chunks: list[str] = []
    current_chunk = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        units_in_sentence = _spoken_units(sentence)
        units_in_chunk = _spoken_units(current_chunk) if current_chunk else 0

        if current_chunk and units_in_chunk + units_in_sentence > max_chunk_size:
            chunks.append(current_chunk)
            current_chunk = sentence
        else:
            current_chunk = f"{current_chunk} {sentence}".strip() if current_chunk else sentence

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


class _PlainTextRenderer(HTMLRenderer):
    """Renderer that strips markdown formatting, outputting plain text."""

    def text(self, text):
        return text

    def emphasis(self, text):
        return text

    def strong(self, text):
        return text

    def link(self, text, **attrs):
        return text

    def image(self, text, **attrs):
        return text or ""

    def codespan(self, text):
        return text

    def linebreak(self):
        return "\n"

    def softbreak(self):
        return " "

    def paragraph(self, text):
        return text + "\n\n"

    def heading(self, text, level, **attrs):
        return text + "\n"

    def block_code(self, code, **attrs):
        return code + "\n"

    def block_quote(self, text):
        return text

    def list(self, text, ordered, **attrs):
        return text

    def list_item(self, text, **attrs):
        return "• " + text + "\n" if text else ""

    def thematic_break(self):
        return "\n"


def strip_markdown(text: str) -> str:
    """Strip markdown formatting from text, returning plain text."""
    md = mistune.create_markdown(renderer=_PlainTextRenderer())
    return md(text).strip()
