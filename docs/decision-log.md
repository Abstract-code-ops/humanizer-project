# Decision Log (this session)

A running record of decisions made during the conversational build-out, so the
"why" behind the current state is preserved even though the chat transcript is
not stored on disk. Pointer to the artifacts that encode each decision.

## Iteration 1 — scaffolding (initial build)

- Repo was docs-only (no Python source, no `pip`, no git, no `package-lock.json`).
- Installed `python3-pip`, `python3-venv`; created `.venv`.
- Scaffolded `generators/`, `humanizers/`, `detectors/`, `evaluation/`,
  `data/source_texts/`, `data/outputs/`, `docs/`, `tests/`.
- Wrote `config.yaml`, fixed source texts (3 topics), `baseline.py`,
  `detectors/turnitin.py` + `detectors/proxy.py`, `evaluation/similarity.py` +
  `evaluation/readability.py`, `run_experiment.py`, tests, `pyproject.toml`,
  `.gitignore`; `git init`.
- `package-lock.json` was accidentally deleted by the user → added to `.gitignore`
  (it belongs to the Reasonix CLI wrapper, not the research project).

## Iteration 2 — stack decisions (user-driven)

- **Generator**: switched to Gemini, model `gemini-3.1-flash-lite` (user first
  said "3.1 flash" then corrected to "flash lite"). Key in `GEMINI_API_KEY`.
- **Detector**: no free commercial API → self-hosted HF model. Chose
  `TrustSafeAI/RADAR-Vicuna-7B` (RoBERTa-large, adversarially trained against
  paraphrasing — best fit for detecting *humanized* text).
- **Paraphraser**: `Vamsi/T5_Paraphrase_Paws` (T5 fine-tuned on PAWS paraphrase
  corpus). History: tried `google/flan-t5-base` (summarized, not paraphrased),
  then `tuner007/pegasus_paraphrase` (broken `spiece.model` tokenizer), landed on
  T5_Paraphrase_Paws.
- **Embedder**: `sentence-transformers/all-MiniLM-L6-v2` (384-dim, cosine).
- **Similarity**: upgrade from lexical cosine to MiniLM embeddings (behind
  `similarity.similarity(method=...)`).

## Iteration 3 — hosting decision (user-driven)

- Initially considered HF Spaces CPU (one Space, all 3 models, ~3-4 GB RAM).
- **Corrected after research**: HF's free CPU Basic Spaces now require paid PRO,
  and free ZeroGPU is Gradio-only with ~5 min/day GPUX quota.
- **Final: Modal** — serverless, stable HTTPS URL, ~$30/month free credits,
  scale-to-zero. Colab is now only a dev fallback (no longer the hosting target).
- Created `deploy/modal_app.py` (endpoints `/detect`, `/paraphrase`, `/embed`,
  `/health`) and `docs/hosting-modal.md` (walkthrough).

## Where each decision lives

- Hosting / model choices: `architecture.md`
- Open assumptions & unresolved items: `docs/open-questions.md`
- Module status: `progress-tracker.md`
- Hosting guide: `docs/hosting-modal.md`
- Deployment code: `deploy/modal_app.py`

## Iteration 4 — wiring + deployment (done)

- `detectors/proxy.py` → `hf` provider calls `MODAL_DETECT_URL` (RADAR). ✅
- `evaluation/similarity.py` → `embeddings` method calls `MODAL_EMBED_URL` (MiniLM). ✅
- `generators/__init__.py` → Gemini `gemini-3.1-flash-lite` via
  `generativelanguage.googleapis.com/v1beta/models/{model}:generateContent`. ✅
- `humanizers/paraphrase.py` → new tier calling `MODAL_PARAPHRASE_URL`. ✅
- `config.yaml` → activated live providers (`gemini` / `hf` / `embeddings`). ✅
- `scripts/model.sh` → CLI to call detect/paraphrase/embed/health. ✅
- `tests/test_wiring.py` → 5 offline tests (no live calls). ✅
- Deployed to Modal (app `humanizer-models`, workspace `abstract-code-ops`); four
  URLs live. ✅ `detect`, `embed`, `health` verified working; `paraphrase` BLOCKED.

## Iteration 5 — second detector model (desklib)

- Added `desklib/ai-text-detector-v1.01` as a second proxy detector, selectable
  at runtime. It's a **DeBERTa-v3-large** fine-tune (mean-pooled + linear head,
  single logit -> sigmoid = P(AI)), and it tops the RAID benchmark's adversarial
  leaderboard — a more representative detector for *humanized/paraphrased* text
  than RADAR, which was the original choice.
- `deploy/modal_app.py`: `/detect` now accepts `{"text", "model"}` and dispatches
  between RADAR (softmax, class 0) and Desklib (sigmoid logit) via
  `_DETECTOR_LOADERS`. `proxy.score(...)` gains a `model=` kwarg (or
  `DETECTOR_MODEL` env), defaulting to RADAR so existing wiring is unchanged.
- The Desklib model needs a **custom module** (`DesklibAIDetectionModel`) because
  it is a raw logit head, not a `*ForSequenceClassification` checkpoint — so it
  cannot be loaded with `AutoModelForSequenceClassification` like RADAR.
- Tests added (`test_proxy_sends_desklib_model_to_endpoint`,
  `test_proxy_defaults_to_radar`). **Needs a Modal redeploy** before Desklib is
  callable live (the `paraphrase` 500 issue is still separate and unresolved).

## Iteration 6 — deployment bugs resolved (paraphrase + detector)

- **Root cause of the long-standing `/paraphrase` 500 was NOT a stale container
  or a broken `tuner007/pegasus_paraphrase` model** (that earlier diagnosis was
  wrong). The real cause was a **missing dependency**: the Modal image installed
  only `torch`, `transformers`, `sentence-transformers`, `fastapi`, so T5's fast
  tokenizer had no `sentencepiece` to build from `spiece.model`. Transformers then
  fell through to a tiktoken conversion path, and `tiktoken` was also absent —
  hence the misleading "tiktoken is required / Tiktoken failed" traceback.
  - **Fix**: added `sentencepiece`, `protobuf` (sentencepiece's dep), and `tiktoken`
    to the image.
- **Detector also had a real bug** (independent of the tokenizer): the desklib
  `DesklibAIDetectionModel.forward()` didn't accept the `token_type_ids` that the
  DeBERTa-v3 tokenizer emits, so `model(**inputs)` raised
  `TypeError ... unexpected keyword argument 'token_type_ids'`.
  - **Fix**: `forward(..., **kwargs)` absorbs-and-drops `token_type_ids` (DeBERTa
    doesn't need it at inference).
- Pinned `transformers<4.49` (the desklib custom `PreTrainedModel` subclass hits
  `all_tied_weights_keys` on >=4.49).
- **Result**: all four endpoints (`detect`/RADAR + `detect`/desklib, `paraphrase`,
  `embed`, `health`) now work. **Desklib label direction confirmed**: a high
  sigmoid value = AI-generated (no flip needed). RADAR label direction also
  confirmed correct.

## Iteration 7 — naive tier (synonym substitution)

- Implemented `humanizers/naive.py`: WordNet same-POS synonym substitution on
  content words (noun/verb/adj/adv), skipping stopwords, preserving casing, with
  `substitution_rate` (default **15%**), `seed` (reproducibility), and a
  `dry_run` arg for interface uniformity.
- Uses `nltk` (punkt/punkt_tab/averaged_perceptron_tagger_eng/wordnet/stopwords)
  with lazy, idempotent download in `_ensure_resources()`. `_synonyms_for()`
  factored out so tests can inject a deterministic map without the WordNet corpus.
- Wired into `run_experiment.py` `_resolve_tier` as `"naive"`.
- **Noted limitation** (for the report): naive WordNet substitution can produce
  context-inappropriate synonyms (e.g. "artificial → hokey", "approach → near")
  because it ignores sense disambiguation — evidence the tier is semantically
  shallow even though it lowers lexical/turn-of-phrase overlap. This is the
  exact tradeoff the benchmark exists to quantify.

## Iteration 8 — naive back-translation tier

- **User decisions** (explicit, not assumed): back-translation is a **separate
  tier** (`naive_bt`) rather than a technique inside `naive` — so the two show as
  distinct rows in `results.csv` (diverges from `architecture.md`'s "synonym +
  back-translation" single-tier wording). Single **French** pivot. Model is
  **MarianMT Opus-MT** (`Helsinki-NLP/opus-mt-en-fr` + `Helsinki-NLP/opus-mt-fr-en`,
  Apache-2.0, ~300 MB each, CPU) rather than NLLB-200 (non-commercial, heavier) or
  mBART-50 (heavier, no quality edge for en↔fr).
- Added `/backtranslate` endpoint to `deploy/modal_app.py` (CPU, round-trip
  en→fr→en) + `MODAL_BACKTRANSLATE_URL` env var. Created `humanizers/naive_bt.py`
  (`humanize(text, dry_run)`). Wired `"naive_bt"` into `run_experiment.py`.
- **Orchestration fix**: `run_experiment.py` now inspects the tier's `humanize`
  signature and passes `dry_run` only when the tier accepts it, so `--dry-run`
  no longer crashes tiers with a `dry_run` kwarg (a latent gap that also affects
  `naive`).

## Iteration 9 — adversarial-loop tier

- Implemented `humanizers/adversarial_loop.py` as an `AdversarialLoop` class
  (constructor args per architecture.md's "extra config" rule) exposing
  `.humanize(text) -> str`.
- **User decisions** (explicit): perturbation is **paraphrase-only** (sweep
  `strength` 1..5); detector is **Desklib only**
  (`desklib/ai-text-detector-v1.01`).
- **Mechanism**: score the original, then for each strength 1..min(max_iters,5)
  paraphrase and rescore; keep the lowest-scoring variant; stop early once
  `target_score` is met. Realizes AuthorMist's "detector-as-reward" as search,
  not RL (no training).
- **Rule compliance**: does not import `paraphrase.py` (calls `/paraphrase`
  directly) per "no tier imports another tier"; `dry_run` stub avoids endpoint
  calls in tests/wiring.
- Wired `"adversarial"` into `run_experiment.py` (reads `adversarial_loop` from
  `config.yaml` for `target_proxy_score` / `max_iters`).

## Iteration 10 — LLM-prompted paraphrase tier (+ drop commercial)

- Added `humanizers/paraphrase_llm.py` (tier `paraphrase_llm`): prompts **Gemini**
  (`gemini-3.1-flash-lite`, `PARAPHRASE_MODEL` env) to "rewrite the text so it
  sounds naturally human while preserving meaning/length". Contrasts with the
  dedicated TF paraphraser (`paraphrase.py` → `Vamsi/T5_Paraphrase_Paws`): one is
  instruction-tuned generative rewriting, the other a purpose-built paraphraser.
- Provider/env knobs: `PARAPHRASE_PROVIDER` (`gemini` | `mock`), `PARAPHRASE_MODEL`.
  `dry_run` stub avoids API spend in tests/wiring.
- **Dropped `commercial.py` (manual logging tier)** per user decision — the
  commercial-tool comparison is out of scope now (was Goal 1's optional tier #4).

## Iteration 11 — paraphrase content-loss + back-translation gibberish fixes

- **Paraphraser content-loss**: T5_Paraphrase_Paws (T5-base) was summarising away
  content on long inputs. **Fix**: add `min_length ≈ 0.9× chunk length`,
  `length_penalty=1.0`, `no_repeat_ngram_size=3`, and switch from `do_sample` to
  greedy beam search — forces output closer to input length.
- **Back-translation gibberish** (`"the ses, the ses…"` repetition): MarianMT
  greedy (`num_beams=1`) ran away into repetition. **Fix**: `num_beams=5`,
  `no_repeat_ngram_size=3`, `repetition_penalty=1.2`.
- **Report note (GPU constraint)**: the ideal fix is a ~7B instruct LLM
  (Qwen2.5-7B / Llama-3.1-8B) for paraphrase, which preserves length/facts far
  better than any T5/pegasus-base model. **Excluded because the Modal GPU budget
  / T4 cannot host a 7B model in fp16.** The report should state this explicitly.

## Iteration 12 — back-translation gibberish + word-count collapse (root cause)

- **Symptom**: `naive_bt` output contained runaway repetition (`the ses, the ses…`,
  `right-right-right…`) and on long inputs collapsed to ~170 words (e.g.
  `china_expansion` 338 → 167).
- **Root cause** (two compounding mistakes in `deploy/modal_app.py`'s `/backtranslate`):
  1. `_translate` packed many sentences back into ~400-token chunks. MarianMT
     Opus-MT is a **sentence-level** translator; long multi-sentence chunks are
     out of distribution and degrade.
  2. The decode call passed `no_repeat_ngram_size=3` + `repetition_penalty=1.2`
     + `num_beams=5`. On MarianMT those penalties are the well-known trigger for
     exactly this degenerate repetition loop — they were added in Iteration 11 as
     a "fix" but made it worse.
- **Fix**: translate **one sentence at a time** (`_split_sentences` output used
  directly, no `_chunk_sentences`), and use plain beam decoding
  (`num_beams=4`, `do_sample=False`, `early_stopping=True`) with **no**
  `no_repeat_ngram_size` / `repetition_penalty`. Output length now tracks input
  length and the repetition loop is removed.
- **Required**: redeploy (`modal deploy deploy/modal_app.py`) and re-test via
  `./scripts/model.sh backtranslate "<long text>"` before trusting `results.csv`.

## Iteration 13 — paraphrase (T5) gibberish + content-loss (root cause)

- **Symptom**: the `paraphrase` tier (`Vamsi/T5_Paraphrase_Paws` via `/paraphrase`)
  produced filler garbage (`" . - -- -. n . - s n —s"`), dropped most paragraphs
  (~120 words out of ~420), and truncated mid-sentence. `adversarial` inherited the
  corruption because it reuses `/paraphrase`. `naive` and `paraphrase_llm` (Gemini)
  were unaffected.
- **Root cause** (decode constraints, same class of bug as Iteration 12):
  1. `/paraphrase` chunked many sentences into ~400-token blocks, but
     `T5_Paraphrase_Paws` is a **sentence-level** model trained on PAWS
     (single-sentence pairs) — multi-sentence chunks are out of distribution.
  2. It forced `min_length ≈ 0.9× chunk` + `length_penalty=1.0` + beam search +
     `no_repeat_ngram_size=3`. Forcing near-input-length output from a model with
     nothing meaningful to generate produces filler tokens, and the aggressive
     constraints interplay to cause the degenerate output.
- **Fix**: translate **one sentence at a time** and use the model's official
  sampling recipe (`do_sample=True`, `top_k`/`top_p`, `early_stopping=True`)
  with `temperature`/`top_k`/`top_p` scaled by `strength` (1-5). Removed
  `min_length`, `length_penalty`, and beam search.
- **Required**: redeploy (`modal deploy deploy/modal_app.py`), re-test
  `./scripts/model.sh paraphrase "<long text>"` and re-run the `paraphrase` +
  `adversarial` tiers before trusting `results.csv`.
- **Security hardening (same session)**: after the decode fix, a security review
  flagged two DoS/validation issues introduced by the per-sentence loop — now
  fixed: `strength` is validated+clamped to int 1..5 (try/except on
  `int(...)` covering `TypeError`/`ValueError`/`OverflowError`, then
  `max(1, min(5, ...))`), and both `paraphrase` and `backtranslate` cap their
  per-request decode loop at 200 sentences. Residual (deferred): endpoints are
  still public/unauthenticated with no `concurrency_limit` — acceptable for a
  research deploy, but add Modal auth + `concurrency_limit` before any public
  exposure beyond the workspace.

## Iteration 14 — UI fixes + evaluation/charts + source-text swap (this session)

- **UI routing fixed**: `/api/humanize` now routes each mode to the correct
  implementation — `naive` → local WordNet synonym sub (was silently falling
  through to `/paraphrase`), `naive_bt` → `/backtranslate`, `paraphrase` →
  `/paraphrase`, `paraphrase_llm` → Gemini, `adversarial` → `AdversarialLoop`.
- **Copy/paste fixed**: `navigator.clipboard` is unavailable over plain HTTP
  (secure-context only), so `copyTextValue`/`readClipboardText` helpers add a
  hidden-`textarea` + `document.execCommand` fallback.
- **Detect button state**: `Detect` swaps to "Working…" + disabled while the
  request is in flight (matching the Humanize button).
- **XSS hardening**: `renderOut`/`diffText` rendered LLM/user output via
  `innerHTML` unescaped; added `escapeHtml()` on both paths.
- **`ui/app.py` import fix**: added `sys.path.insert(0, ROOT)` so
  `from humanizers import ...` resolves when launched as `python ui/app.py`
  (the `ModuleNotFoundError` surfaced by `start_ui.sh`).
- **Evaluation part**: source texts refactored (removed `climate_change`/
  `artificial_intelligence`/`renaissance_art`; added `pigeon`/`mycelium`/`beaver`,
  all Gemini). Full 6×6 experiment rerun live. UI now **persists** runs to
  `results.csv` (`_score_and_record` + `/api/summary`), and the Compare page shows
  live Chart.js charts with hover + 5s polling.
  - **User decisions**: (a) UI runs append to `results.csv` (single source of
    truth) rather than a separate store; (b) charts use Chart.js via CDN; (c) run
    the experiment live now (real endpoints).
  - **Review-driven fixes**: `_score_and_record` explicitly uses
    `desklib/ai-text-detector-v1.01` (was falling back to RADAR/mock and skewing
    the averages); CSV header wrote+flushed for missing/empty files to avoid a
    CLI/UI header race.
- **Chart re-render fix**: the Compare charts were destroyed/recreated every 5s
  poll; now `ensureChart`/`updateChart` mutate data in place with
  `chart.update("none")` and skip entirely when data is unchanged.

## Iteration 15 — external-skill humanizer tiers (blader/humanizer skill)

- Vendored the MIT-licensed **blader/humanizer** agent skill
  (`skills/humanizer/SKILL.md`, v2.11.2, 35 "signs of AI writing" patterns +
  rewrite process) and added it as a **runtime prompt**, not a code tier logic.
- Added `humanizers/skill_humanizer.py` (`SkillHumanizer` class) prompting an LLM
  to apply the skill to source text. Two tiers to contrast backends:
  - `skill_gemini` → Gemini (`gemini-3.1-flash-lite`, `GEMINI_API_KEY`)
  - `skill_deepseek` → DeepSeek (`deepseek-chat`, `DEEPSEEK_API_KEY` — previously
    unused)
- Rationale: compare an **external, curated skill prompt** against `paraphrase_llm`'s
  short hand-written "sound human" prompt as the *only* variable (prompt quality),
  since both are instruction-tuned LLM rewrites.
- Wired into `run_experiment.py::_resolve_tier` + documented in `config.yaml`.
  Tests + `ruff` + `pytest` green; dry-run smoke run against a temp CSV (so the
  protected `results.csv` is never polluted by stub rows).
- **Result** (live, desklib proxy): both skill tiers are **counterproductive on
  the three AI nature essays** (`pigeon`/`mycelium`/`beaver` stay ~0.99–1.0), and
  they even *raise* the human `writer_voice`/`china_expansion` to ~0.94–0.98 —
  the same failure mode as `paraphrase_llm`. The curated skill improves meaning
  retention (similarity ~0.90–0.98 vs `paraphrase_llm`'s 0.88–0.95) but does **not**
  evade a RAID-trained detector. See `docs/report-facts.md`.

## Outstanding / blockers

- Turnitin manual scoring workflow confirmation (proxy≠Turnitin gap still unmeasured).
- Report writeup (user writes externally; anchored on `docs/report-facts.md`).

## Environment notes (for whoever resumes)

- Python env: `.venv` (3.12). `modal` CLI installed **in the venv** — always use
  `.venv/bin/modal` (plain `modal` is not on PATH).
- Modal volume: `humanizer-models-vol` (model cache).
- Env vars in `.env` (gitignored): `GEMINI_API_KEY`, `DEEPSEEK_API_KEY` (unused),
  `MODAL_DETECT_URL`, `MODAL_PARAPHRASE_URL`, `MODAL_EMBED_URL`,
  `MODAL_BACKTRANSLATE_URL`, `MODAL_HEALTH_URL`, `PROXY_PROVIDER=hf`.
- **Security**: the Modal API token secret was pasted into chat at one point — it
  should be rotated (`modal token new`). The UI and Modal endpoints remain
  unauthenticated (no rate limiting / `concurrency_limit`) — acceptable for a
  research deploy on a local/VPN network, but not for public exposure.
