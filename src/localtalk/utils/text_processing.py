"""Text processing utilities for the Local Talk App."""

import re


def clean_text_for_tts(text: str) -> str:
    """Clean text for better TTS output.

    Args:
        text: Raw text to clean

    Returns:
        Cleaned text suitable for TTS
    """
    # Remove markdown formatting
    text = re.sub(r"\*{1,2}([^\*]+)\*{1,2}", r"\1", text)  # Remove bold/italic
    text = re.sub(r"`([^`]+)`", r"\1", text)  # Remove code blocks
    text = re.sub(r"#{1,6}\s+", "", text)  # Remove headers

    # Remove URLs
    text = re.sub(r"https?://\S+", "link", text)

    # Remove extra whitespace
    text = " ".join(text.split())

    return text


def get_first_sentence(text: str) -> tuple[str, str]:
    """Extract the first sentence from text for immediate TTS playback.

    Args:
        text: Full text response

    Returns:
        Tuple of (first_sentence, remaining_text)
    """
    # Common sentence endings
    sentence_endings = r"[.!?]"

    # Find the first sentence ending
    match = re.search(f"(.+?{sentence_endings})\\s*(.*)", text, re.DOTALL)

    if match:
        first_sentence = match.group(1).strip()
        remaining = match.group(2).strip()

        # Make sure the first sentence is substantial enough
        if len(first_sentence) < 10 and remaining:
            # Too short, try to get the next sentence too
            next_match = re.search(f"(.+?{sentence_endings})\\s*(.*)", remaining, re.DOTALL)
            if next_match:
                first_sentence += " " + next_match.group(1).strip()
                remaining = next_match.group(2).strip()

        return first_sentence, remaining
    else:
        # No sentence ending found, return the whole text
        return text, ""


def chunk_text_for_streaming(text: str, max_chunk_size: int = 50) -> list[str]:
    """Split text into chunks suitable for streaming TTS.

    Args:
        text: Text to chunk
        max_chunk_size: Maximum words per chunk

    Returns:
        List of text chunks
    """
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        words_in_sentence = len(sentence.split())
        words_in_chunk = len(current_chunk.split())

        if words_in_chunk + words_in_sentence <= max_chunk_size:
            current_chunk = (current_chunk + " " + sentence).strip()
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk)

    return chunks
