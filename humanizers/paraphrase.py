"""paraphrase tier: dedicated HF paraphraser (Vamsi/T5_Paraphrase_Paws) via Modal.

History (see docs/decision-log.md): google/flan-t5-base summarized instead of
paraphrasing; tuner007/pegasus_paraphrase had a broken spiece.model tokenizer.
T5_Paraphrase_Paws (fine-tuned on the PAWS paraphrase corpus) is the tier that
actually preserves length/meaning.

`strength` is 1-5 (default 3), forwarded to the Modal endpoint, which scales
temperature/top_k/top_p accordingly.
"""
import os
import time
import requests


class ParaphraseError(RuntimeError):
    pass


def humanize(text: str, strength: int = 3, timeout: int = 180, retries: int = 1) -> str:
    strength = max(1, min(5, int(strength)))  # clamp, mirrors server-side validation
    url = os.environ["MODAL_PARAPHRASE_URL"]

    last_err = None
    for attempt in range(retries + 1):
        try:
            resp = requests.post(url, json={"text": text, "strength": strength}, timeout=timeout)
            resp.raise_for_status()
            return resp.json()["paraphrased"]
        except requests.RequestException as e:
            last_err = e
            if attempt < retries:
                time.sleep(2)  # brief backoff, e.g. for cold-start flakiness
    raise ParaphraseError(
        f"Paraphrase endpoint failed after {retries + 1} attempt(s). "
        f"If this is the first call in a while, the Modal container may still be "
        f"cold-starting (can take 10-60s) — try again shortly. Original error: {last_err}"
    )
