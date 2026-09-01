"""Shared helpers for humanizer tiers."""
import re


def split_sentences(text: str) -> list[str]:
    """Naive sentence splitter (good enough for source_texts register).

    Kept dependency-free so it can run inside the Modal image without extra
    downloads. Splits on ., !, ? followed by whitespace + capital/EOF.
    """
    text = text.strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def cap_words(text: str, max_words: int = 200) -> str:
    """Deterministically truncate to the first `max_words` words (whitespace split)."""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def has_overlong_token(text: str, max_token_length: int = 10) -> bool:
    """Return True if any word token exceeds the configured max length."""
    return any(len(token) > max_token_length for token in re.findall(r"\w+", text))


def word_count(text: str) -> int:
    return len(text.split())
