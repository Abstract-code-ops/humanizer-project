# Humanizer — Backend Summary (for the UI developer)

This document is a plain-English overview of the backend so you can build/maintain
the web UI without reading every Python file. It covers **what the system does**,
**the tiers**, **the HTTP endpoints you call**, **environment variables / keys**,
**how scoring works**, and — most importantly — the **new behavior change** for
the "Humanize" action (3-way best-of pipeline + a 200-word cap).

Repo to clone (all code lives here):

```
https://github.com/Abstract-code-ops/humanizer.git
```

---

## 1. What this project is

An **AI-text humanizer + benchmark harness**. It takes AI-generated text, rewrites
it to sound human through several different "tiers" (techniques), scores the
result against an AI detector, and records everything so the techniques can be
compared.

Two parts share one codebase:

1. **CLI pipeline** (`run_experiment.py`) — runs a fixed set of source texts
   through every tier and appends scores to `data/results.csv`. This is the
   canonical experiment/record.
2. **Web UI** (`ui/`) — a thin Flask client over the *same* Modal endpoints and
   the *same* `results.csv`. There is **no separate database**; the UI is just a
   nicer front door onto the existing backend.

The deliverable is both the tool and the comparative report; the instructor's
goal is to *measure* how well humanization evades Turnitin, not to ship a
consumer product.

---

## 2. Key concepts / terminology

- **Tier** — a single humanization technique (e.g. synonym swap, back-translation,
  paraphrase). Each has a `humanize(text) -> str` interface.
- **Proxy detector / `proxy_score`** — the stand-in for Turnitin. It is a
  self-hosted Hugging Face model (`desklib/ai-text-detector-v1.01`) served on
  Modal. Output is a probability `0.0 – 1.0`, **higher = more AI-like**. Lower is
  "more human", which is the goal.
- **Turnitin score** — manual, NOT automated. The `turnitin_score` column in
  `results.csv` is filled by hand later; there is no public Turnitin API.
- **Similarity** — semantic cosine similarity (MiniLM embeddings) between the
  original and the humanized text. `0.0 – 1.0`, **higher = less meaning lost**.
- **Readability** — Flesch Reading Ease. Higher = easier to read.

---

## 3. The tiers (humanization techniques)

Ordered roughly by sophistication. The `mode` string is what the UI sends.

| mode / tier | Technique | Backend | Status for UI |
|---|---|---|---|
| `baseline` | passthrough (no change) | `humanizers/baseline.py` | (reference only) |
| `naive` | WordNet synonym swap (~15% content words) | `humanizers/naive.py` | local, no network |
| `naive_bt` | back-translation English → French → English (MarianMT) | Modal `/backtranslate` | **top performer** |
| `paraphrase` | dedicated HF paraphraser (`Vamsi/T5_Paraphrase_Paws`) | Modal `/paraphrase` | **top performer** |
| `paraphrase_llm` | Gemini prompt "sound human" (short, hand-written prompt) | Gemini API | counterproductive — do not surface |
| `adversarial` | paraphrase + score with detector, keep lowest | Modal `/paraphrase` + `/detect` | **top performer** |
| `skill_gemini` | Gemini + vendored "Humanizer" skill prompt | Gemini API | counterproductive — do not surface |
| `skill_deepseek` | DeepSeek + vendored "Humanizer" skill prompt | DeepSeek API | counterproductive — do not surface |
| (commercial) | dropped | — | removed — do not surface |

### The three tiers that actually work (measured on the benchmark)

These are the only ones that consistently move the detector score down:

1. **`paraphrase`** — the strongest overall on AI text (e.g. `age_of_exploration`
   0.9763 → 0.0057), though it is inconsistent on some samples.
2. **`naive_bt`** (back-translation) — strongest on the two AI nature essays
   (`pigeon` 0.9991 → 0.1214).
3. **`adversarial`** — best when there is headroom (`beaver` 0.9999 → 0.0303).

The LLM-prompt tiers (`paraphrase_llm`, `skill_gemini`, `skill_deepseek`)
**backfire**: they often *raise* the AI score or stamp human text with AI
phrasing. They exist only as benchmark data points; the UI should **not** offer
them.

---

## 4. ⭐ NEW: "Humanize" now runs a 3-way best-of pipeline (behavior change)

Instead of the user picking one tier, the primary **Humanize** action now:

1. Runs the input through the **three top-performing tiers**:
   - `paraphrase` (via Modal `/paraphrase`)
   - `naive_bt` (via Modal `/backtranslate`)
   - `adversarial` (paraphrase + detector loop)
   - (internally, `adversarial` is itself a loop that also calls `/paraphrase` +
     `/detect`)
2. Scores **each** candidate with the proxy detector (`proxy_score`, lower = better).
3. Returns the candidate with the **lowest (most human) score**.

### Word-count limit: 200

Before any processing, the input is capped/truncated to **200 words**. Rationale:
the paraphrase and back-translation models are sentence/paragraph-level and
degrade (repetition, gibberish, truncation) on very long inputs; a 200-word cap
keeps outputs clean and keeps API/latency reasonable.

- Truncate to the first 200 words (by whitespace) deterministically.
- The UI should show a subtle note that input is capped at 200 words (e.g. a
  counter that turns amber past 200, or a "200-word limit" hint).
- The similarity metric is still computed against the **truncated** input, so the
  "meaning lost" number is meaningful.

This is the single biggest thing to get right in the new UI.

---

## 5. HTTP endpoints the UI can call

All scoring/rewriting models run on **Modal** (serverless). The Flask UI posts to
them over HTTPS. Modal gives **one stable URL per function** (not a shared base).

### Modal model endpoints (the real ML backend)

| Purpose | Method | URL (env var) | Request body | Response |
|---|---|---|---|---|
| Detect (AI probability) | POST | `MODAL_DETECT_URL` | `{"text": "...", "model": "desklib/ai-text-detector-v1.01"}` | `{"ai_probability": 0.87}` |
| Paraphrase (T5) | POST | `MODAL_PARAPHRASE_URL` | `{"text": "...", "strength": 1–5}` | `{"paraphrased": "..."}` |
| Back-translate | POST | `MODAL_BACKTRANSLATE_URL` | `{"text": "..."}` | `{"backtranslated": "..."}` |
| Embed (similarity) | POST | `MODAL_EMBED_URL` | `{"text": "..."}` | `{"embedding": [0.12, -0.03, ...]}` (384 floats) |
| Health | GET | `MODAL_HEALTH_URL` | — | `{"status":"ok", ...}` |

Notes for the UI:

- **Cold start** — the first request after idle can take 10–60s (loads ~3 GB of
  models). The UI must show a "Working…" state and a generous timeout (use ≥120s,
  up to 180s for backtranslate/paraphrase). Once warm (~5 min window) it's fast.
- **`strength`** for paraphrase is an integer 1–5; 3 is the default; the
  adversarial loop internally sweeps 1..5.
- **Detector direction** — a *high* `ai_probability` = AI, *low* = human. The
  circular gauge in the UI should map **low → green (human), high → red (AI)**.

### Flask UI endpoints (already implemented in `ui/app.py`)

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Humanize page |
| GET | `/detect` | Detect page |
| GET | `/compare` | Compare/results page |
| POST | `/api/humanize` | Humanize a text (see §6) |
| POST | `/api/detect` | Proxy a detector score |
| POST | `/api/similarity` | Compute MiniLM similarity (`{a, b}`) |
| GET | `/api/results` | All `results.csv` rows (JSON) |
| GET | `/api/summary` | Per-tier averages for the Compare charts |

---

## 6. `/api/humanize` contract (the core action)

Current request/response shape (from `ui/app.py`):

**Request** (`POST`, JSON):

```json
{
  "text": "the input text (will be capped at 200 words)",
  "mode": "paraphrase",
  "strength": 3
}
```

**Response**:

```json
{
  "humanized": "the rewritten text",
  "mode": "paraphrase",
  "proxy_score": 0.0438,
  "similarity": 0.9753,
  "readability": 35.17
}
```

**⚠️ For the new best-of behavior**, the backend should change so that instead of a
single `mode`, the request runs all three top tiers and returns the best candidate.
Decide the exact contract with whoever owns the backend; the minimal compatible
change is to keep the same response shape but make `mode` return
`"best-of-three"` (or add a `mode: "auto"` / `mode: "best"` value) and have
`humanized` be the lowest-`proxy_score` output among `paraphrase` + `naive_bt` +
`adversarial`.

The rest of the response stays the same so the existing gauge/stats rendering is
unchanged.

---

## 7. Environment variables / secrets

The backend reads these from `.env` (gitignored — **never commit them**).

```
DEEPSEEK_API_KEY=sk-...                  # unused by the UI (skill_deepseek tier only)
GEMINI_API_KEY=AIzaSy...                 # generator + paraphrase_llm / skill_gemini tiers
MODAL_DETECT_URL=https://abstract-code-ops--humanizer-models-detect.modal.run
MODAL_PARAPHRASE_URL=https://abstract-code-ops--humanizer-models-paraphrase.modal.run
MODAL_EMBED_URL=https://abstract-code-ops--humanizer-models-embed.modal.run
MODAL_HEALTH_URL=https://abstract-code-ops--humanizer-models-health.modal.run
MODAL_BACKTRANSLATE_URL=https://abstract-code-ops--humanizer-models-backtranslate.modal.run
PROXY_PROVIDER=hf
```

The full Modal URLs (these are the stable, deploy-time values — do not guess them):

```
MODAL_DETECT_URL       = https://abstract-code-ops--humanizer-models-detect.modal.run
MODAL_PARAPHRASE_URL   = https://abstract-code-ops--humanizer-models-paraphrase.modal.run
MODAL_EMBED_URL        = https://abstract-code-ops--humanizer-models-embed.modal.run
MODAL_HEALTH_URL       = https://abstract-code-ops--humanizer-models-health.modal.run
MODAL_BACKTRANSLATE_URL= https://abstract-code-ops--humanizer-models-backtranslate.modal.run
```

Important for the UI developer:

- The UI must **not** hardcode these URLs — read them from env/config, because
  they can change on a redeploy (`modal deploy`).
- The detector `model` id used everywhere is `desklib/ai-text-detector-v1.01`
  (the RAID-tuned detector chosen for `proxy_score`). The other detector,
  `TrustSafeAI/RADAR-Vicuna-7B`, also exists on the endpoint but is **not** the
  one the benchmark scores against.
- `modal` CLI lives at `.venv/bin/modal` (not on PATH).

---

## 8. Data files the UI reads/writes

| Path | Role |
|---|---|
| `data/results.csv` | **Single source of truth** for scores. Schema: `sample_id,tier,proxy_score,turnitin_score,similarity,readability`. Append-only during runs. The UI appends to it via `/api/humanize` → `_score_and_record`. |
| `data/outputs/<tier>/<sample_id>.txt` | Raw humanized text, saved alongside CSV rows for spot-checking. |
| `data/source_texts/*.txt` | The 6 fixed input texts (do not change; changing them breaks comparability). |
| `config.yaml` | Experimental parameters (source list, detector model, eval method, adversarial thresholds). |

The Compare page reads `/api/results` (table) and `/api/summary` (per-tier
averages for the Chart.js bar charts), recomputed live from `results.csv`.

---

## 9. What the UI developer actually needs to build (summary)

1. **Humanize** action = **3-way best-of** (`paraphrase` + `naive_bt` +
   `adversarial`), return the lowest `proxy_score` output.
2. **Cap input at 200 words** before processing; show a visible limit indicator.
3. Render a **circular gauge** for `proxy_score` (low → green/human, high →
   red/AI) plus plain numbers for `similarity` (%) and `readability` (grade).
4. Keep the existing **Compare/Results** page reading `/api/results` +
   `/api/summary` — no change needed there beyond possibly adding the new
   "best-of" row.
5. **Do not surface** `paraphrase_llm`, `skill_gemini`, `skill_deepseek`, or
   `commercial` — they underperform or were dropped.
6. Handle **cold start** (long first request) and plain-HTTP clipboard fallback
   (`document.execCommand`) if served without TLS.

The measured results and all methodology caveats (proxy ≠ Turnitin, naive tier
non-determinism, 200-word rationale, etc.) are consolidated in
`docs/report-facts.md` for when you write the report.
