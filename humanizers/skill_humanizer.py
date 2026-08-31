"""skill_gemini / skill_deepseek tiers: LLM + vendored 'Humanizer' skill prompt.

Prompts an LLM with the vendored blader/humanizer skill (skills/humanizer/SKILL.md,
MIT license, 35 "signs of AI writing" patterns + rewrite process) to remove
those patterns while preserving meaning.

STATUS: measured counterproductive against a RAID-trained detector (see
docs/report-facts.md finding 7) — improves meaning retention over
paraphrase_llm but does not evade detection. Kept as a benchmark data point;
do not surface in the UI.

`provider` is a constructor arg so the same class drives both `skill_gemini`
and `skill_deepseek` tiers, contrasting backends with prompt held constant.
"""
import os
import requests

from config import load_config

_GEMINI_URL_TMPL = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)
_DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"


class SkillHumanizer:
    def __init__(self, provider: str, skill_path: str | None = None, model: str | None = None):
        cfg = load_config()["skill_humanizer"]
        self.skill_path = skill_path or cfg["skill_path"]
        provider_cfg = cfg["providers"].get(provider)
        if provider_cfg is None:
            raise ValueError(f"Unknown skill_humanizer provider: {provider!r}")
        self.provider = provider_cfg["provider"]
        self.model = model or provider_cfg["model"]
        self._skill_text = self._load_skill()

    def _load_skill(self) -> str:
        try:
            with open(self.skill_path, "r") as f:
                return f.read()
        except FileNotFoundError:
            # Vendored skill file not present in this checkout — degrade to a
            # short built-in instruction rather than crashing the tier.
            return (
                "Rewrite the text to remove common signs of AI writing "
                "(generic transitions, hedging, repetitive sentence rhythm, "
                "overused vocabulary) while preserving meaning and length."
            )

    def _prompt(self, text: str) -> str:
        return (
            f"{self._skill_text}\n\n"
            f"Apply the above to rewrite this text. Output only the rewritten "
            f"text, no commentary:\n\n{text}"
        )

    def humanize(self, text: str, timeout: int = 60) -> str:
        prompt = self._prompt(text)
        if self.provider == "gemini":
            return self._call_gemini(prompt, timeout)
        elif self.provider == "deepseek":
            return self._call_deepseek(prompt, timeout)
        raise ValueError(f"Unsupported provider: {self.provider}")

    def _call_gemini(self, prompt: str, timeout: int) -> str:
        api_key = os.environ["GEMINI_API_KEY"]
        url = _GEMINI_URL_TMPL.format(model=self.model)
        resp = requests.post(
            url,
            params={"key": api_key},
            json={"contents": [{"parts": [{"text": prompt}]}]},
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()

    def _call_deepseek(self, prompt: str, timeout: int) -> str:
        api_key = os.environ["DEEPSEEK_API_KEY"]
        resp = requests.post(
            _DEEPSEEK_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
