"""Semantic similarity between original and humanized text.

Pure function, no I/O beyond the network call for the "embeddings" method.
method="embeddings" (default, config.yaml) uses MiniLM via the Modal /embed
endpoint, cosine similarity, 0.0-1.0, higher = less meaning lost.
method="lexical" is a dependency-free fallback (token-overlap Jaccard) for
offline/dev use — not what results.csv is built from.
"""
import os
import math
import requests

from config import load_config


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _embed(text: str, timeout: int = 60) -> list[float]:
    url = os.environ["MODAL_EMBED_URL"]
    resp = requests.post(url, json={"text": text}, timeout=timeout)
    resp.raise_for_status()
    return resp.json()["embedding"]


def _lexical(a: str, b: str) -> float:
    set_a, set_b = set(a.lower().split()), set(b.lower().split())
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def similarity(a: str, b: str, method: str | None = None) -> float:
    method = method or load_config()["similarity"]["method"]
    if method == "embeddings":
        return _cosine(_embed(a), _embed(b))
    elif method == "lexical":
        return _lexical(a, b)
    raise ValueError(f"Unknown similarity method: {method}")
