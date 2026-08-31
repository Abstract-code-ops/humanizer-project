"""Source-text generator — Gemini, model gemini-3.1-flash-lite.

One-directional dependency: this module never imports from humanizers/ or
detectors/. Source text is generated once and reused across all tiers.

`dry_run=True` returns a stub without hitting the API, so tests / pipeline
wiring checks never spend real API budget.
"""
import os
import requests

_GEMINI_URL_TMPL = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)
DEFAULT_MODEL = "gemini-3.1-flash-lite"


def generate(prompt: str, model: str = DEFAULT_MODEL, dry_run: bool = False,
             timeout: int = 60) -> str:
    if dry_run:
        return f"[DRY RUN generated text for prompt: {prompt[:60]}...]"

    api_key = os.environ["GEMINI_API_KEY"]
    url = _GEMINI_URL_TMPL.format(model=model)
    resp = requests.post(
        url,
        params={"key": api_key},
        json={"contents": [{"parts": [{"text": prompt}]}]},
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["candidates"][0]["content"]["parts"][0]["text"].strip()
