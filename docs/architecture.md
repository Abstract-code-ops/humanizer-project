# Architecture

## Layout
```
humanizer_project/
├── generators/          # produces original AI source text (Gemini API)
├── humanizers/           # one module per tier, shared interface
│   ├── baseline.py
│   ├── naive.py           # synonym substitution
│   ├── naive_bt.py        # back-translation (en→fr→en)
│   ├── paraphrase.py      # HF paraphraser (T5_Paraphrase_Paws via Modal)
│   ├── paraphrase_llm.py  # LLM-prompted paraphrase (Gemini)
│   ├── adversarial_loop.py
│   └── _shared.py         # shared helpers
├── detectors/             # wraps Turnitin (manual) + proxy detector API
├── evaluation/            # semantic similarity, readability, scoring
├── ui/                    # Flask web frontend (Humanize/Detect/Compare)
├── deploy/                # modal_app.py (hosts detect/paraphrase/embed/backtranslate)
├── data/
│   ├── source_texts/
│   └── results.csv
├── docs/                   # this file, progress tracker, etc.
└── run_experiment.py       # orchestrates generate → humanize → detect → score
```
(Note: `commercial.py` was dropped by user decision — no commercial-tool tier.)

## Layer boundaries
- **generators/** never calls anything in `humanizers/` or `detectors/` — one-directional dependency, source text is generated once and reused across all tiers.
- **humanizers/** modules only depend on `evaluation/` for internal checks (e.g. adversarial loop calling the proxy detector). No humanizer module calls Turnitin directly — that's `detectors/`'s job.
- **detectors/** is the only layer that talks to Turnitin or the proxy API. If Turnitin scoring is manual, `detectors/turnitin.py` exposes a function that reads a manually-filled CSV column rather than special-casing manual entry throughout the codebase.
- **evaluation/** is pure functions — no I/O beyond reading text, no API calls. Semantic similarity and readability only.
- **run_experiment.py** is the only place that orchestrates across layers.

## Shared interface (critical convention)
Every humanizer tier implements the same function signature so tiers are swappable:
```python
def humanize(text: str) -> str:
    ...
```
Adversarial-loop and other tiers needing extra config (detector, thresholds) take them as constructor args to a class, but still expose `.humanize(text) -> str` as the call surface `run_experiment.py` uses.

## Scoring workflow decision
Turnitin has no public API → two-track scoring:
1. **Proxy detector** drives the adversarial loop automatically. Because no free-tier commercial API (GPTZero etc.) is available, the proxy is a **self-hosted Hugging Face model** (see "AI-detector model + hosting" below) wrapped in `detectors/proxy.py` behind the same `score(text) -> float` surface.
2. **Turnitin** is scored manually at the end — final humanized outputs are submitted once each and the score is recorded in `data/results.csv`.
Report the gap between proxy-score and Turnitin-score as its own finding — don't treat the proxy as equivalent to Turnitin.

## Generator model
Source text is produced by **Gemini** via the Google Generative Language API. Model string: `gemini-3.1-flash-lite`. The API key is read from `os.environ["GEMINI_API_KEY"]` (in `.env`, gitignored). The generator keeps a `dry_run` flag so tests and pipeline wiring checks never spend real API budget.

## AI-detector model + hosting (self-hosted HF, no paid API)
The proxy detector is an open HF model, not a commercial API. Candidates (ranked by fit for *this* project, which must detect **humanized/paraphrased** text, not just raw AI text):

1. **RADAR** — `TrustSafeAI/RADAR-Vicuna-7B` (RoBERTa-large backbone). Adversarially trained *jointly against a paraphraser*, so it is explicitly robust to paraphrasing — the single most relevant property for a humanizer benchmark. NeurIPS 2023. Non-commercial license (fine here). **Recommended.**
2. **ParaDetect** — `srikanthgali/paradetect-deberta-v3-lora` (DeBERTa-v3-large + LoRA). ~99% accuracy, but 435M-param backbone → heavier to host. Fallback if RADAR underperforms on our text.
3. **followsci/bert-ai-text-detector** — BERT-base (110M), tuned on academic text, 99.5% accuracy. Lightest option; strong if our source texts are academic in register.

Hosting decision — **Modal** (serverless, stable HTTPS URL):
- HF's free **CPU Basic** Space now requires a paid PRO plan for new compute
  Spaces, and HF **ZeroGPU** is Gradio-only with ~5 min/day GPU quota (too tight
  for an adversarial loop). So the three models are hosted on **Modal**
  (`deploy/modal_app.py`) instead.
- One Modal app exposes five endpoints, each with its **own stable URL** (Modal
  `fastapi_endpoint` gives one URL per function, not a shared base). Record them
  in `.env` as `MODAL_DETECT_URL`, `MODAL_PARAPHRASE_URL`, `MODAL_EMBED_URL`,
  `MODAL_BACKTRANSLATE_URL`, `MODAL_HEALTH_URL`. ~$30/month free serverless credits,
  scale-to-zero. See `docs/hosting-modal.md` for the full walkthrough.
- **Colab is now only a dev fallback**, not the hosting target (its ngrok URL
  rotates and it disconnects on idle).

Wherever the models run, `detectors/proxy.py` / `humanizers/paraphrase.py` call
them over HTTP like any external API.

## Embedding / semantic similarity model
Replace the lexical cosine similarity with a real embedding model (per code-standards this stays behind `evaluation/similarity.similarity(method=...)`):
- **`sentence-transformers/all-MiniLM-L6-v2`** — 384-dim, 80M params, maps paragraphs to dense vectors, cosine similarity. Small/cheap, runs on CPU, standard for paragraph similarity. **Recommended.**
- Alternative if more multilingual coverage needed: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.

## HF paraphrase model hosting (in the shared Modal app)
The paraphraser is **`Vamsi/T5_Paraphrase_Paws`** (a T5 model fine-tuned on the
PAWS paraphrase corpus). It preserves length/meaning far better than a generic
instruction model (e.g. `google/flan-t5-base`, which summarized instead of
paraphrasing). Served from the **same Modal app** as the detector and embedder —
see `docs/hosting-modal.md`.

- The app exposes `POST /paraphrase` with body `{"text": "...", "strength": 1-5}`, returning `{"paraphrased": "..."}`.
- **`humanizers/paraphrase.py`** calls this endpoint over HTTP like any other external API — it does not know or care that the model is running in Modal rather than Colab:
  ```python
  def humanize(text: str) -> str:
      resp = requests.post(
          os.environ["MODAL_PARAPHRASE_URL"],
          json={"text": text, "strength": 2},
          timeout=60,
      )
      return resp.json()["paraphrased"]
  ```
- **Model choice**: `Vamsi/T5_Paraphrase_Paws` is a dedicated paraphraser. Note
  the sequence of models tried: `google/flan-t5-base` summarized instead of
  paraphrasing; `tuner007/pegasus_paraphrase` had a broken `spiece.model`
  tokenizer incompatible with modern transformers/tiktoken. Add a retry/clear
  error message in `paraphrase.py` if the endpoint is cold-starting.
- The Modal URLs are **stable** (unlike ngrok), so the four `MODAL_*_URL` variables in `.env` are fixed once deployed, never hardcoded into the repo.

## Back-translation model hosting (in the shared Modal app)
The naive back-translation tier (`humanizers/naive_bt.py`) round-trips text
**English → French → English** using two **Helsinki-NLP MarianMT Opus-MT**
models (`Helsinki-NLP/opus-mt-en-fr` + `Helsinki-NLP/opus-mt-fr-en`, Apache-2.0,
~300 MB each, CPU). Chosen over NLLB-200 (CC-BY-NC, heavier) and mBART-50
(heavier, no en↔fr quality edge) — see `docs/decision-log.md` Iteration 8.

- The app exposes `POST /backtranslate` with body `{"text": "..."}`, returning `{"backtranslated": "..."}`.
- **`humanizers/naive_bt.py`** calls `MODAL_BACKTRANSLATE_URL` over HTTP, exactly like `paraphrase.py`.

## Data flow
```
source_texts/*.txt
   → generators (if not already AI-authored)
   → humanizers/<tier>.humanize(text)
   → detectors/turnitin.score(text)  [manual]
   → detectors/proxy.score(text)     [automated]
   → evaluation/similarity.score(original, humanized)
   → evaluation/readability.score(humanized)
   → append row to data/results.csv
```

## Config
Single `config.yaml` at project root for: API keys (loaded from env, never committed), model names per tier, adversarial loop thresholds/max_iters, list of source text files. No hardcoded model names inside tier modules.
