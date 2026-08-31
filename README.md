# Humanizer

AI-text humanizer + benchmark harness. Takes AI-generated text, rewrites it
through several tiers, scores the result against a proxy AI detector, and
records everything so the techniques can be compared. Ships a CLI experiment
runner and a thin Flask UI over the same backend.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in GEMINI_API_KEY, DEEPSEEK_API_KEY (optional), MODAL_*_URL
```

## Deploy the ML backend (Modal)

```bash
pip install modal
modal setup
.venv/bin/modal deploy deploy/modal_app.py
# copy the five printed URLs into .env as MODAL_DETECT_URL / MODAL_PARAPHRASE_URL /
# MODAL_EMBED_URL / MODAL_BACKTRANSLATE_URL / MODAL_HEALTH_URL
```

## Run the CLI experiment

```bash
python run_experiment.py                 # all tiers, all source texts -> data/results.csv
python run_experiment.py --tier paraphrase --sample pigeon
python scripts/make_charts.py             # regenerate data/charts/*.png
```

## Run the web UI

```bash
python ui/app.py
# open http://localhost:5000  (Humanize / Detect / Compare)
```

The **Humanize** action runs `paraphrase` + `naive_bt` + `adversarial` and
returns whichever scores lowest on the AI detector, after capping input to
200 words. `paraphrase_llm`, `skill_gemini`, `skill_deepseek` are measured
counterproductive and are not surfaced in the UI (see `docs/report-facts.md`).

## Tests (offline, no API calls)

```bash
pytest tests/
```

## Docs

- `docs/architecture.md` — layout, layer boundaries, model/hosting decisions
- `docs/BACKEND_SUMMARY.md` — UI-developer-facing API contract
- `docs/report-facts.md` — full experiment results + findings for the report
- `docs/decision-log.md` — running history of decisions and bug fixes

## Notes

- No separate database — `data/results.csv` is the single source of truth for
  both the CLI and the UI.
- Turnitin has no public API; `turnitin_score` is filled in by hand.
- Endpoints are currently unauthenticated — fine for a research deploy on a
  private network, add Modal auth + `concurrency_limit` before public exposure.
