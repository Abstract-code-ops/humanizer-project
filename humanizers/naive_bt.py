"""naive_bt tier: back-translation English -> French -> English.

Calls the Modal-hosted MarianMT (Helsinki-NLP/opus-mt-en-fr + fr-en) endpoint.
Single French pivot (chosen for reproducibility over robustness — see
docs/decision-log.md Iteration 8).

IMPORTANT (Iteration 12 fix baked into the contract): the Modal endpoint
translates one sentence at a time internally with plain beam decoding — do not
re-introduce chunking or repetition-penalty knobs client-side; that caused the
runaway-repetition bug documented in the decision log.
"""
import os
import requests


def humanize(text: str, timeout: int = 180) -> str:
    url = os.environ["MODAL_BACKTRANSLATE_URL"]
    resp = requests.post(url, json={"text": text}, timeout=timeout)
    resp.raise_for_status()
    return resp.json()["backtranslated"]
