"""adversarial tier: single call to the Modal /adversarial endpoint, which
now runs the whole rewrite -> detect -> repeat loop server-side inside one
warm container (see deploy/modal_app.py, class Adversarial).

Previously this class ran that loop client-side — one HTTP round trip to
/paraphrase and one to /detect per iteration, sequentially, from wherever
Flask happens to be hosted. That was the actual source of "adversarial is
out of the question with this performance": network latency and Modal
cold-starts were being paid once per iteration instead of once per request.
Now it's exactly one HTTP call total, no matter how many internal iterations
the server-side loop uses to converge.

NOTE: I haven't seen humanizers/_shared.py, so this calls `requests`
directly rather than any existing shared HTTP helper (retries / error
wrapping / timeouts) your other humanizer modules might already centralize.
If paraphrase.py or detectors/proxy.py already route through a shared
`_post_to_modal()`-style helper, swap that in here instead of the inline
`requests.post` call below, for consistency.
"""
import logging
import os

import requests

from config import load_config

logger = logging.getLogger(__name__)

MODAL_ADVERSARIAL_URL = os.environ.get("MODAL_ADVERSARIAL_URL", "")


class AdversarialLoop:
    def __init__(self, detector_model: str | None = None,
                 target_proxy_score: float | None = None,
                 strength_sweep: list[int] | None = None,
                 max_iters: int | None = None):
        cfg = load_config()
        adv_cfg = cfg["adversarial"]

        self.detector_model = detector_model or cfg["detector"]["model"]
        self.target_proxy_score = (
            target_proxy_score if target_proxy_score is not None
            else adv_cfg["target_proxy_score"]
        )
        self.max_iters = max_iters or adv_cfg["max_iters"]

        # Accepted for backward compatibility with existing callers/config
        # (e.g. ui/app.py just does `AdversarialLoop()`), but no longer used
        # directly — the server-side loop in Modal now ramps paraphrase
        # strength itself as it alternates between paraphrase and
        # backtranslate. Kept as a constructor arg so nothing else needs to
        # change.
        self.strength_sweep = strength_sweep or adv_cfg.get("strength_sweep")

        # The current Modal-side Adversarial class always scores against its
        # own default (desklib) detector — it doesn't yet accept a `model`
        # override the way /detect does. If a non-default detector_model is
        # requested here, it silently won't be honored server-side; surface
        # that instead of failing quietly.
        self._detector_model_unsupported = (
            self.detector_model != cfg["detector"]["model"]
        )
        if self._detector_model_unsupported:
            logger.warning(
                "AdversarialLoop was given detector_model=%r, but the Modal "
                "/adversarial endpoint currently always scores against %r "
                "internally — the override is not applied server-side.",
                self.detector_model, cfg["detector"]["model"],
            )

    def humanize(self, text: str) -> str:
        if not MODAL_ADVERSARIAL_URL:
            raise RuntimeError(
                "MODAL_ADVERSARIAL_URL is not set. Add it to your .env — "
                "it's printed by `modal deploy deploy/modal_app.py` as the "
                "URL for Adversarial.adversarial."
            )

        payload = {
            "text": text,
            "target_ai_probability": self.target_proxy_score,
            "max_iterations": self.max_iters,
        }

        # Connect timeout short; read timeout generous but a little under
        # the Modal function's own timeout=300. If this fires first, the
        # Modal container keeps running orphaned regardless — that's a
        # client-side request timeout, not a cancellation (see the
        # cancellation discussion in docs/decision-log.md for the actual
        # fix: Modal .spawn()/.cancel() rather than a synchronous POST).
        try:
            resp = requests.post(MODAL_ADVERSARIAL_URL, json=payload, timeout=(10, 240))
            resp.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"adversarial endpoint request failed: {e}") from e

        data = resp.json()
        if "error" in data:
            raise RuntimeError(f"adversarial endpoint error: {data['error']}")

        logger.info(
            "adversarial: %s -> %s in %s iteration(s) (target_reached=%s)",
            round(data.get("original_ai_probability", -1), 4),
            round(data.get("ai_probability", -1), 4),
            data.get("iterations_used"),
            data.get("target_reached"),
        )

        return data["humanized"]
