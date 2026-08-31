"""Proxy AI-detector — stand-in for Turnitin (no public Turnitin API).

Wraps the Modal-hosted /detect endpoint behind a plain `score(text) -> float`
surface, higher = more AI-like. Defaults to desklib (RAID-tuned; the model
used for all proxy_score values in results.csv). RADAR is also available on
the same endpoint by passing model=... explicitly.
"""
import os
import requests

from config import load_config

DEFAULT_MODEL = "desklib/ai-text-detector-v1.01"


def score(text: str, model: str | None = None, timeout: int = 120) -> float:
    cfg = load_config()
    model = model or cfg["detector"]["model"] or DEFAULT_MODEL
    url = os.environ["MODAL_DETECT_URL"]
    resp = requests.post(url, json={"text": text, "model": model}, timeout=timeout)
    resp.raise_for_status()
    return float(resp.json()["ai_probability"])
