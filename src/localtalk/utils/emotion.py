"""Emotion analysis utilities."""


def analyze_emotion(text: str) -> float:
    """Analyze text for emotional content.

    Args:
        text: Text to analyze

    Returns:
        Emotion score between 0.3 and 0.9
    """
    # Keywords that suggest more emotion
    emotional_keywords = [
        "amazing", "terrible", "love", "hate", "excited", "sad", "happy", "angry",
        "wonderful", "awful", "fantastic", "horrible", "great", "bad", "excellent",
        "poor", "!", "?!", "..."
    ]

    emotion_score = 0.5  # Default neutral

    text_lower = text.lower()
    for keyword in emotional_keywords:
        if keyword in text_lower:
            emotion_score += 0.1

    # Count exclamation marks
    emotion_score += text.count("!") * 0.05

    # Cap between 0.3 and 0.9
    return min(0.9, max(0.3, emotion_score))


def adjust_tts_parameters(text: str, base_exaggeration: float, base_cfg: float) -> tuple[float, float]:
    """Adjust TTS parameters based on text emotion.

    Args:
        text: Text to analyze
        base_exaggeration: Base exaggeration value
        base_cfg: Base CFG weight value

    Returns:
        Tuple of (adjusted_exaggeration, adjusted_cfg)
    """
    emotion_score = analyze_emotion(text)

    # Blend with base values
    adjusted_exaggeration = base_exaggeration * 0.5 + emotion_score * 0.5

    # Lower cfg_weight for more expressive responses
    adjusted_cfg = base_cfg * 0.8 if emotion_score > 0.6 else base_cfg

    return adjusted_exaggeration, adjusted_cfg
