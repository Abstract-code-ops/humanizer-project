"""paraphrase_llm tier: Gemini prompted to 'sound human'.

STATUS: counterproductive — measured to *raise* the AI-detector score (see
docs/report-facts.md, finding 4). Kept only as a benchmark data point; the UI
must not surface this tier as a user-facing option.
"""
import os
import requests

from config import load_config

_GEMINI_URL_TMPL = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)


def humanize(text: str, timeout: int = 60) -> str:
    cfg = load_config()["paraphrase_llm"]
    api_key = os.environ["GEMINI_API_KEY"]
    model = cfg["model"]
    prompt = cfg["prompt"].format(text=text)

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
