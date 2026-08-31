"""adversarial tier: paraphrase-strength sweep, scored against the proxy
detector each iteration, keeps the lowest-scoring variant.

Realizes the "detector-as-reward" idea as search (not RL): sweeps
strength 1..5 through the `paraphrase` tier, stops early once a candidate's
proxy_score <= target_proxy_score (default 0.30, config.yaml), max 5 iters.
On already-low-scoring (human) text it will correctly just return the best
candidate found, which may be close to the original.
"""
from humanizers import paraphrase as paraphrase_tier
from detectors import proxy as proxy_detector
from config import load_config


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
        self.strength_sweep = strength_sweep or adv_cfg["strength_sweep"]
        self.max_iters = max_iters or adv_cfg["max_iters"]

    def humanize(self, text: str) -> str:
        best_text = text
        best_score = proxy_detector.score(text, model=self.detector_model)

        for i, strength in enumerate(self.strength_sweep[: self.max_iters]):
            candidate = paraphrase_tier.humanize(text, strength=strength)
            score = proxy_detector.score(candidate, model=self.detector_model)
            if score < best_score:
                best_text, best_score = candidate, score
            if best_score <= self.target_proxy_score:
                break

        return best_text
