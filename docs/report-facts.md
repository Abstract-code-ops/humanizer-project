# Report Facts & Data (consolidated)

This file is a single source of truth for everything the final report needs:
scope, methods, model stack, the humanizer tiers, the experiment results, and the
caveats an instructor is likely to probe. It was assembled from the actual
`data/results.csv`, `config.yaml`, and the project docs / decision log.

---

## 1. Project scope and goal

- **What**: a benchmark harness that measures how well different AI-text
  "humanization" techniques evade an AI-text detector, while quantifying the
  quality/meaning tradeoff.
- **Goal (from `project-overview.md`)**: implement multiple humanization tiers of
  increasing sophistication, run a fixed set of source texts through each, and
  score every output on (a) AI-detection probability, (b) semantic similarity to
  the original, and (c) readability. Then explain *why* each technique moves the
  score the way it does.
- **Non-goal**: not a consumer product; not reverse-engineering Turnitin; not
  training a custom paraphrase model from scratch.

## 2. Deliverable status

| Component | Status |
|---|---|
| Repo scaffold + config | done |
| Source texts (6 samples) | done |
| Generator (Gemini) | done |
| Humanizer tiers (5 real + baseline) | done |
| Detectors (Turnitin manual + proxy) | done |
| Evaluation (similarity + readability) | done |
| Orchestration (`run_experiment.py`) | done |
| Charts (score-by-tier, tradeoff scatter) | done |
| Web UI (Humanize/Detect/Compare) | done |
| Writeup | not started (user writes externally) |

## 3. Models / stack

- **Generator (source text)**: Google Gemini, model `gemini-3.1-flash-lite`,
  key `GEMINI_API_KEY`.
- **Proxy AI detector (primary, used for all `proxy_score` values)**:
  `desklib/ai-text-detector-v1.01` — a **DeBERTa-v3-large** fine-tune (mean-pooled
  + linear head → single logit → sigmoid = P(AI-generated)). Topped the **RAID
  benchmark** (adversarial leaderboard) at submission. Served on Modal as
  `/detect`. Label direction verified: *high value = AI-generated*.
- **Second detector (available, not used for proxy_score)**: `TrustSafeAI/RADAR-Vicuna-7B`
  (RoBERTa-large, adversarially trained against paraphrasing). Softmax, class 0 = AI.
- **Paraphraser (HF)**: `Vamsi/T5_Paraphrase_Paws` (T5 fine-tuned on PAWS).
  Served on Modal as `/paraphrase` (body `{"text", "strength" 1-5}`).
- **Back-translation**: two **Helsinki-NLP MarianMT Opus-MT** models,
  `Helsinki-NLP/opus-mt-en-fr` + `Helsinki-NLP/opus-mt-fr-en` (Apache-2.0,
  ~300 MB each, CPU). Served on Modal as `/backtranslate` (en → fr → en).
- **Embedder (semantic similarity)**: `sentence-transformers/all-MiniLM-L6-v2`
  (384-dim cosine). Served on Modal as `/embed`.
- **LLM-prompted paraphraser**: Google Gemini `gemini-3.1-flash-lite` via
  `PARAPHRASE_PROVIDER=gemini`.
- **Hosting**: all ML models on **Modal** (app `humanizer-models`, workspace
  `abstract-code-ops`), stable HTTPS URLs in `.env` (`MODAL_DETECT_URL`,
  `MODAL_PARAPHRASE_URL`, `MODAL_EMBED_URL`, `MODAL_BACKTRANSLATE_URL`,
  `MODAL_HEALTH_URL`). ~$30/mo free credits, scale-to-zero, T4 GPU for
  detect/paraphrase, CPU for embed/backtranslate.
- **Environment**: Python 3.12, `.venv`; `nltk` (WordNet etc.), `matplotlib`,
  `transformers<4.49`, `sentencepiece`, `protobuf`, `tiktoken` (deploy image).

## 4. The humanizer tiers (shared interface `humanize(text) -> str`)

Ordered by sophistication:

1. **baseline** — passthrough (`baseline.py`). Reference point.
2. **naive** — synonym substitution (`naive.py`). WordNet same-POS synonym swap on
   content words (noun/verb/adj/adv), skipping stopwords, **~15% substitution
   rate** (configurable `substitution_rate=0.15`). No word-sense disambiguation
   (intentional — keeps the tier "naive").
3. **naive_bt** — back-translation (`naive_bt.py`). MarianMT round-trip
   **English → French → English**, single pivot.
4. **paraphrase** — dedicated HF paraphraser (`paraphrase.py`).
   `Vamsi/T5_Paraphrase_Paws` via Modal `/paraphrase` (strength 3).
5. **paraphrase_llm** — instruction-tuned LLM rewrite (`paraphrase_llm.py`).
   Gemini prompted to "rewrite so it sounds human, preserve meaning/length".
6. **adversarial** — `AdversarialLoop` class. Paraphrase-strength sweep (1..5),
   each candidate scored with **desklib**, keeps the lowest-scoring variant, stops
   early once `target_proxy_score <= 0.3` (max 5 iters). Realizes AuthorMist's
   "detector-as-reward" idea as search, not RL.
7. **skill_gemini** / **skill_deepseek** — external-skill rewrites. Prompt an LLM
   (Gemini `gemini-3.1-flash-lite` / DeepSeek `deepseek-chat`) with the vendored
   **blader/humanizer** skill (`skills/humanizer/SKILL.md`, MIT, 35 "signs of AI
   writing" patterns) to remove those patterns while preserving meaning.
   `SkillHumanizer` provider is a constructor arg (`humanizers/skill_humanizer.py`).

(Dropped: `commercial.py` manual-logging tier — out of scope by user decision.)

## 5. Source texts (6 samples)

| sample_id | words | origin |
|---|---|---|
| writer_voice | 139 | human ("Reddit voice" post) |
| china_expansion | 338 | human (history writeup) |
| age_of_exploration | 421 | **AI-generated** (LLM answer) |
| pigeon | 387 | **AI-generated** (Gemini) |
| mycelium | 287 | **AI-generated** (Gemini) |
| beaver | 272 | **AI-generated** (Gemini) |

Note: the three *former* initial texts (`climate_change`, `artificial_intelligence`,
`renaissance_art`, all AI-generated) were **removed** this session and replaced
with three Gemini-authored nature essays (`pigeon`, `mycelium`, `beaver`). The
final set is 3 AI-generated + 2 human texts + 1 AI-generated LLM answer, for
uneven lengths (139–421 words). Length affects per-sample readability/similarity
but not the detector comparison.

## 6. Metrics / scoring

- **proxy_score** = desklib AI-probability (0.0–1.0, higher = more AI-like).
  This is the *stand-in* for Turnitin (no public Turnitin API).
- **turnitin_score** = manual, currently **empty** column (to be backfilled).
- **similarity** = semantic cosine similarity (MiniLM embeddings) between
  original and humanized text (0.0–1.0, higher = less meaning lost).
- **readability** = Flesch Reading Ease (higher = easier to read).

## 7. Full experiment results (`data/results.csv`)

`proxy_score` (desklib) per sample × tier:

| tier | writer_voice | china_expansion | age_of_exploration | pigeon | mycelium | beaver |
|---|---|---|---|---|---|---|
| baseline | 0.046 | 0.0006 | 0.9763 | 0.9991 | 1.0 | 0.9999 |
| naive | 0.0105 | 0.0005 | 0.0287 | 0.9111 | 0.9997 | 0.9632 |
| naive_bt | 0.0629 | 0.001 | 0.1429 | 0.1214 | 0.9985 | 0.9963 |
| paraphrase | 0.0442 | 0.0017 | 0.0057 | 0.0438 | 0.9974 | 0.6204 |
| paraphrase_llm | 0.9756 | 0.9991 | 0.9839 | 0.998 | 1.0 | 0.9995 |
| adversarial | 0.046 | 0.0006 | 0.1897 | 0.1742 | 0.3412 | 0.0303 |

Mean `similarity` (semantic, higher = less meaning lost) and `readability`
(Flesch, higher = easier) per tier:

| tier | similarity | readability |
|---|---|---|
| baseline | 1.0000 | 35.30 |
| naive | 0.9863 | 34.41 |
| naive_bt | 0.9742 | 38.96 |
| paraphrase | 0.9625 | 36.74 |
| paraphrase_llm | 0.9109 | 34.13 |
| adversarial | 0.9599 | 36.74 |

Note: these are the **post-fix** rerun values (back-translation and paraphrase
decode bugs fixed — see decision-log Iterations 12–13), so they differ materially
from the earlier gibberish-affected run. The old pre-restructure snapshot is
preserved at `data/results.pre-restructure-backup.csv`.

## 8. Key findings (the report's headline conclusions)

1. **The detector cleanly separates the human vs AI texts at baseline.**
   - Human: writer_voice **0.046**, china_expansion **0.0006**.
   - AI: age_of_exploration **0.9763**, pigeon **0.9991**, mycelium **1.0**, beaver **0.9999**.
2. **`paraphrase` (T5) is the strongest humanizer on AI text** (after the decode
   fix): on `age_of_exploration` it drops **0.9763 → 0.0057** and on `pigeon`
   **0.9991 → 0.0438**, at ~0.96 mean similarity. But it is noteably **inconsistent** —
   `mycelium` stays at 0.9974 and `beaver` at 0.6204.
3. **`naive_bt` (back-translation) is the strongest on the two AI nature essays**:
   `pigeon` **0.9991 → 0.1214** and (with `adversarial`) `beaver` — with high mean
   similarity (0.9742).
4. **`paraphrase_llm` (Gemini "sound human" prompt) is counterproductive and
   harmful**: it *raises* the human texts to ~0.98–0.999 (stamps them with generic
   AI phrasing) and does nothing for the AI texts. Clear evidence that naive
   prompt-based "humanize" can *backfire* against a RAID-trained detector.
5. **The adversarial loop helps most when there is headroom**: it drove `beaver`
   **0.9999 → 0.0303** and `pigeon` **0.9991 → 0.1742**, but on `mycelium` it only
   reached 0.3412 (still worse than paraphrase/naive_bt). On already-low human
   text it correctly returns the original unchanged.
6. **The three AI nature essays are near detector saturation (~1.0)**, so most
   tiers can only nudge some of them; `age_of_exploration` and `pigeon`/`beaver`
   are the more informative AI cells for comparing tiers.
7. **The external-skill tiers (skill_gemini / skill_deepseek) also backfire.**
   On the AI nature essays they hold near saturation (`pigeon`/`mycelium`/`beaver`
   ~0.99–1.0) and they *raise* the human texts (`writer_voice` 0.046→~0.94–0.98,
   `china_expansion` 0.0006→0.94). The curated 35-pattern skill improves meaning
   retention (similarity ~0.90–0.98 vs `paraphrase_llm`'s 0.88–0.95) but does not
   evade a RAID-trained detector — strong evidence that "remove AI tells" rewriting
   alone, even with a good prompt, does not fool a detector adversarially trained
   on paraphrased/humanized text.

## 9. Caveats / methodology notes (things an instructor may probe)

- **Proxy ≠ Turnitin**: the `proxy_score` is from `desklib/ai-text-detector-v1.01`
  (a self-hosted HF model), NOT Turnitin. Turnitin scores are manual and still
  empty — the gap between proxy and Turnitin is itself a finding, not presumed zero.
- **Naive tier is non-deterministic**: `naive.py` uses `seed=None` by default, so
  each experiment run produces *different* synonym substitutions. The CSV holds
  the last run's values. Re-running the naive tier would change its scores unless
  a fixed seed is passed. (Flagging this for reproducibility.)
- **Substitution rate (15%) and no word-sense disambiguation** are deliberate,
  minimally-conservative choices; neither was fixed by the project docs.
- **Back-translation pivot = French only** (single pivot, reproducibility over
  robustness). MarianMT chosen over NLLB-200 (non-commercial, heavier) and
  mBART-50 (heavier, no en↔fr quality edge).
- **Adversarial loop optimizes against desklib only**, using a paraphrase-strength
  sweep (not best-of-N, not RL). Best-of-N and worst-case-dual-detector were
  considered and deferred.
- **Source-text provenance**: `writer_voice` and `china_expansion` are human;
  `age_of_exploration` is AI-generated (LLM answer); `pigeon`, `mycelium`, and
  `beaver` are AI-generated (Gemini-authored nature essays, confirmed by user).
  The earlier `climate_change`/`artificial_intelligence`/`renaissance_art` texts
  were removed and replaced. Lengths are uneven (139–421 words).

## 10. Charts

Two chart mechanisms exist:

- **Static report PNGs** (`data/charts/`, via `scripts/make_charts.py`):
  - `score_by_tier.png` — mean desklib AI-probability per tier (bar chart).
  - `tradeoff_scatter.png` — detection-score-drop vs. semantic similarity per
    (tier, sample); the quality/evasion tradeoff.
  - Regenerate: `python scripts/make_charts.py --results data/results.csv --outdir data/charts`.
- **Live web charts** (in the UI "Compare" page): three interactive Chart.js bar
  charts (AI probability, similarity, readability) with hover tooltips, driven by
  `/api/summary` and polling every 5s. These recompute from `data/results.csv`
  and include UI-triggered runs.

## 11. Where each artifact lives

- Results: `data/results.csv` (36 clean rows; schema `sample_id,tier,proxy_score,turnitin_score,similarity,readability`).
- Source texts: `data/source_texts/*.txt`.
- Humanizer outputs: `data/outputs/<tier>/<sample_id>.txt`.
- Config: `config.yaml` (source list, detector model, eval method, adversarial loop params).
- Web UI: `ui/app.py` + `ui/templates/` + `ui/static/` (Humanize/Detect/Compare).
- Charts script: `scripts/make_charts.py`; charts in `data/charts/`.
- Endpoint CLI: `scripts/model.sh`.
- Decision history: `docs/decision-log.md`; open assumptions: `docs/open-questions.md`.
