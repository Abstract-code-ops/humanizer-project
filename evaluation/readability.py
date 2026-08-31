"""Readability scoring — Flesch Reading Ease. Higher = easier to read.

Pure function, no network calls. Uses `textstat`.
"""
import textstat


def readability(text: str) -> float:
    if not text.strip():
        return 0.0
    return float(textstat.flesch_reading_ease(text))
