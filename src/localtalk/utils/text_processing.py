"""Text processing utilities for streaming TTS and speakable output."""

from __future__ import annotations

import re

# Sentence boundary: punctuation followed by whitespace or end-of-string.
_SENTENCE_END = re.compile(r"[.!?。！？]+(?:\s+|$)")


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

    Returns:
        ``(first_sentence, remaining_text)``. If no sentence ending is found,
        returns ``(text, "")``.
    """
    text = text.strip()
    if not text:
        return "", ""

    match = re.search(r"(.+?[.!?。！？])\s*(.*)", text, re.DOTALL)
    if not match:
        return text, ""

    first_sentence = match.group(1).strip()
    remaining = match.group(2).strip()

    # Too short alone — fold in the next sentence when available.
    if len(first_sentence) < 10 and remaining:
        next_match = re.search(r"(.+?[.!?。！？])\s*(.*)", remaining, re.DOTALL)
        if next_match:
            first_sentence = f"{first_sentence} {next_match.group(1).strip()}"
            remaining = next_match.group(2).strip()

    return first_sentence, remaining


def take_complete_sentences(buffer: str) -> tuple[list[str], int]:
    """Split *buffer* into complete sentences and the consume index.

    A sentence is complete when it ends with ``.``, ``!``, or ``?``.
    Very short leading sentences (< 10 chars) are merged with the next one
    when possible (same rule as :func:`get_first_sentence`).

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
    if len(sentences) >= 2 and len(sentences[0]) < 10:
        merged = f"{sentences[0]} {sentences[1]}"
        sentences = [merged, *sentences[2:]]

    return sentences, last


def chunk_text_for_streaming(text: str, max_chunk_size: int = 40) -> list[str]:
    """Split text into sentence-aware chunks for streaming TTS.

    Args:
        text: Text to chunk.
        max_chunk_size: Soft maximum words per chunk (sentences may exceed).

    Returns:
        Non-empty chunks covering the full text order.
    """
    text = text.strip()
    if not text:
        return []

    sentences = re.split(r"(?<=[.!?。！？])\s+", text)
    chunks: list[str] = []
    current_chunk = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        words_in_sentence = len(sentence.split())
        words_in_chunk = len(current_chunk.split()) if current_chunk else 0

        if current_chunk and words_in_chunk + words_in_sentence > max_chunk_size:
            chunks.append(current_chunk)
            current_chunk = sentence
        else:
            current_chunk = f"{current_chunk} {sentence}".strip() if current_chunk else sentence

    if current_chunk:
        chunks.append(current_chunk)

    return chunks
