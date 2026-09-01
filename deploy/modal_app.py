"""Modal app: hosts detect / paraphrase / embed / backtranslate / health.

Deploy: .venv/bin/modal deploy deploy/modal_app.py
App name: humanizer-models (workspace: set by your Modal account)

Each class gets its OWN stable HTTPS URL per web-endpoint method (Modal
fastapi_endpoint behavior — not a shared base path). Record the URLs in .env
as MODAL_DETECT_URL, MODAL_PARAPHRASE_URL, MODAL_EMBED_URL,
MODAL_BACKTRANSLATE_URL, MODAL_HEALTH_URL. NOTE: switching from plain
@app.function endpoints to @app.cls endpoints changes the URL shape — re-copy
all five URLs after your first deploy of this version.

Bug fixes baked in here (see docs/decision-log.md for the full story — do not
regress these):
  - Iteration 6: image needs sentencepiece + protobuf + tiktoken, or T5's fast
    tokenizer silently falls through to a broken tiktoken conversion path.
  - Iteration 6: desklib's forward() must accept & drop **kwargs (token_type_ids)
    since DeBERTa-v3 doesn't need it at inference.
  - Iteration 6: pin transformers<4.49 (desklib custom PreTrainedModel subclass
    breaks on >=4.49's all_tied_weights_keys).
  - Iteration 12/13: translate/paraphrase ONE SENTENCE AT A TIME — meaning
    each individual model input must be a single sentence, never several
    sentences concatenated into one ~400-token block (both MarianMT and
    T5_Paraphrase_Paws are sentence-level models and go out-of-distribution on
    multi-sentence chunks, producing repetition loops / filler-token
    gibberish). Running several *separate* one-sentence inputs together as a
    padded batch is fine — each sequence is still exactly one sentence, we're
    just not looping over them one-by-one in Python anymore. See the
    Perf iteration note below.
  - Iteration 13: do not force min_length / length_penalty / no_repeat_ngram_size
    on MarianMT or T5 decode — those combinations are the documented trigger for
    degenerate output. Use plain beam decoding for MarianMT, and the model's
    official sampling recipe (do_sample, top_k, top_p, temperature scaled by
    `strength`) for T5_Paraphrase_Paws.
  - Iteration 13 (security hardening): `strength` is validated + clamped to
    int 1..5; both paraphrase and backtranslate cap the per-request sentence
    loop at 200 sentences (DoS guard).
  - Perf iteration (this file): the previous version loaded every model from
    scratch inside the request handler on every single call, and never moved
    the model or its inputs onto the GPU despite requesting one — so `detect`
    and `paraphrase` were paying full model-instantiation cost AND running
    inference on CPU every time, and `backtranslate` had no GPU at all. That
    was the actual source of the 15-40s executions, not the models
    themselves. Fixed by: (1) moving model loading into `@modal.enter()` on a
    Modal class so it happens once per container and is reused across warm
    requests, (2) explicitly moving models + inputs to CUDA and running in
    fp16, (3) batching the per-sentence loop into a single padded
    forward/generate call, (4) giving backtranslate a GPU, (5) keeping one
    container warm per function via min_containers so the adversarial loop
    doesn't repeatedly eat cold-start cost.

Outstanding (see decision-log "Outstanding / blockers"): endpoints are public/
unauthenticated with no concurrency_limit — acceptable for a research deploy
on a local/VPN network, add Modal auth + concurrency_limit before wider exposure.
"""
import re

import modal

app = modal.App("humanizer-models")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch",
        "transformers<4.49",
        "sentence-transformers",
        "sentencepiece",
        "protobuf",
        "tiktoken",
        "fastapi[standard]",
    )
)

volume = modal.Volume.from_name("humanizer-models-vol", create_if_missing=True)
MODEL_CACHE = "/cache"

MAX_SENTENCES_PER_REQUEST = 200

#  be  one container warm per class so the adversarial loop (which calls
# detect + paraphrase repeatedly, back and forth) doesn't pay cold-start /
# model-load cost on every hop. Costs a bit of idle GPU time — drop to 0 (or
# swap for scaledown_window=300 to stay warm only after recent traffic) if
# that idle cost matters more than latency for your usage pattern.

# Keep a container warm for a few minutes after its last request, so back-
# to-back calls (like the adversarial loop) hit a warm container without
# paying idle GPU cost around the clock like min_containers=1 did.
SCALEDOWN_WINDOW = 150  # seconds — tune between 120-180 as you like


def _split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()][:MAX_SENTENCES_PER_REQUEST]


def _clamp_strength(strength) -> int:
    try:
        s = int(strength)
    except (TypeError, ValueError, OverflowError):
        s = 3
    return max(1, min(5, s))


# --------------------------------------------------------------------------
# /health
# --------------------------------------------------------------------------
@app.function(image=image)
@modal.fastapi_endpoint(method="GET")
def health():
    return {"status": "ok"}


# --------------------------------------------------------------------------
# /detect  — desklib (default) or RADAR, selected via {"model": "..."}
# Loaded once per container in @modal.enter(), kept on GPU in fp16.
# --------------------------------------------------------------------------
@app.cls(
    image=image,
    gpu="T4",
    volumes={MODEL_CACHE: volume},
    timeout=120,
    scaledown_window=SCALEDOWN_WINDOW,
)
class Detector:
    @modal.enter()
    def load(self):
        import torch
        from transformers import AutoTokenizer, AutoModel

        class DesklibAIDetectionModel(torch.nn.Module):
            def __init__(self, base_model_name):
                super().__init__()
                self.base = AutoModel.from_pretrained(base_model_name)
                hidden = self.base.config.hidden_size
                self.classifier = torch.nn.Linear(hidden, 1)

            def forward(self, input_ids=None, attention_mask=None, **kwargs):
                # **kwargs absorbs-and-drops token_type_ids (DeBERTa doesn't
                # need it at inference) — this is the Iteration 6 fix.
                out = self.base(input_ids=input_ids, attention_mask=attention_mask)
                pooled = out.last_hidden_state.mean(dim=1)
                logit = self.classifier(pooled).squeeze(-1)
                return torch.sigmoid(logit)

        self.device = "cuda"
        name = "desklib/ai-text-detector-v1.01"
        self.desklib_tok = AutoTokenizer.from_pretrained(name, cache_dir=MODEL_CACHE)
        self.desklib_model = (
            DesklibAIDetectionModel(name).to(self.device).half().eval()
        )

        # RADAR is much bigger (Vicuna-7B backbone) — load it lazily, only if
        # a request actually asks for it, but still cache it on the instance
        # so a second request in the same warm container doesn't reload it.
        self._radar = None

    def _load_radar(self):
        if self._radar is None:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification

            name = "TrustSafeAI/RADAR-Vicuna-7B"
            tok = AutoTokenizer.from_pretrained(name, cache_dir=MODEL_CACHE)
            model = (
                AutoModelForSequenceClassification.from_pretrained(
                    name, cache_dir=MODEL_CACHE, torch_dtype=torch.float16
                )
                .to(self.device)
                .eval()
            )
            self._radar = (tok, model)
        return self._radar

    @modal.fastapi_endpoint(method="POST")
    def detect(self, item: dict):
        import torch

        text = item.get("text", "")
        model_name = item.get("model", "desklib/ai-text-detector-v1.01")

        if model_name == "desklib/ai-text-detector-v1.01":
            inputs = self.desklib_tok(
                text, return_tensors="pt", truncation=True, max_length=512
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.inference_mode():
                prob = self.desklib_model(**inputs).float().item()
            return {"ai_probability": prob}

        elif model_name == "TrustSafeAI/RADAR-Vicuna-7B":
            tok, model = self._load_radar()
            inputs = tok(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.inference_mode():
                logits = model(**inputs).logits
                probs = torch.softmax(logits.float(), dim=-1)
                # class 0 = AI, per docs/architecture.md label-direction note.
                prob = probs[0][0].item()
            return {"ai_probability": prob}

        return {"error": f"unknown model {model_name}"}, 400


# --------------------------------------------------------------------------
# /paraphrase — Vamsi/T5_Paraphrase_Paws
# Loaded once per container; sentences are batched into one padded
# generate() call instead of looped one at a time.
# --------------------------------------------------------------------------
@app.cls(
    image=image,
    gpu="T4",
    volumes={MODEL_CACHE: volume},
    timeout=180,
    scaledown_window=SCALEDOWN_WINDOW,
)
class Paraphraser:
    @modal.enter()
    def load(self):
        import torch
        from transformers import T5Tokenizer, T5ForConditionalGeneration

        self.device = "cuda"
        name = "Vamsi/T5_Paraphrase_Paws"
        self.tok = T5Tokenizer.from_pretrained(name, cache_dir=MODEL_CACHE)
        self.model = (
            T5ForConditionalGeneration.from_pretrained(
                name, cache_dir=MODEL_CACHE, torch_dtype=torch.float16
            )
            .to(self.device)
            .eval()
        )

    @modal.fastapi_endpoint(method="POST")
    def paraphrase(self, item: dict):
        import torch

        text = item.get("text", "")
        strength = _clamp_strength(item.get("strength", 3))

        # strength 1-5 scales sampling temperature/top_k/top_p — official
        # sampling recipe, no min_length / length_penalty / beam search
        # (Iteration 13 fix).
        temperature = 0.6 + 0.15 * strength    # ~0.75 .. 1.35
        top_k = 40 + 10 * strength             # 50 .. 90
        top_p = min(0.90 + 0.02 * strength, 1.0)   # 0.92 .. 1.00

        sentences = _split_sentences(text)
        if not sentences:
            return {"paraphrased": ""}

        # Each list item is still exactly one sentence going into the model
        # (Iteration 12/13 constraint unchanged) — padding lets the GPU run
        # them all in one forward pass instead of one Python loop iteration
        # per sentence.
        prompts = [f"paraphrase: {s} </s>" for s in sentences]
        enc = self.tok(
            prompts, return_tensors="pt", truncation=True, max_length=256, padding=True
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}

        with torch.inference_mode():
            out_ids = self.model.generate(
                **enc,
                do_sample=True,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                max_length=256,
                early_stopping=True,
            )

        out_sentences = self.tok.batch_decode(out_ids, skip_special_tokens=True)
        return {"paraphrased": " ".join(out_sentences)}


# --------------------------------------------------------------------------
# /backtranslate — MarianMT en->fr->en
# Previously had NO gpu at all (biggest single cause of its slowness).
# Now on GPU, loaded once per container, sentences batched per hop.
# --------------------------------------------------------------------------
@app.cls(
    image=image,
    gpu="T4",
    volumes={MODEL_CACHE: volume},
    timeout=180,
    scaledown_window=SCALEDOWN_WINDOW,
)
class BackTranslator:
    @modal.enter()
    def load(self):
        import torch
        from transformers import MarianMTModel, MarianTokenizer

        self.device = "cuda"

        en_fr_name = "Helsinki-NLP/opus-mt-en-fr"
        fr_en_name = "Helsinki-NLP/opus-mt-fr-en"

        self.tok_en_fr = MarianTokenizer.from_pretrained(en_fr_name, cache_dir=MODEL_CACHE)
        self.model_en_fr = (
            MarianMTModel.from_pretrained(
                en_fr_name, cache_dir=MODEL_CACHE, torch_dtype=torch.float16
            )
            .to(self.device)
            .eval()
        )

        self.tok_fr_en = MarianTokenizer.from_pretrained(fr_en_name, cache_dir=MODEL_CACHE)
        self.model_fr_en = (
            MarianMTModel.from_pretrained(
                fr_en_name, cache_dir=MODEL_CACHE, torch_dtype=torch.float16
            )
            .to(self.device)
            .eval()
        )

    def _translate_batch(self, sentences, tok, model):
        import torch

        enc = tok(
            sentences, return_tensors="pt", truncation=True, max_length=256, padding=True
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.inference_mode():
            # Plain beam decoding, no no_repeat_ngram_size / repetition_penalty
            # (Iteration 12 fix — those triggered runaway repetition loops).
            out_ids = model.generate(
                **enc, num_beams=4, do_sample=False, early_stopping=True, max_length=256
            )
        return tok.batch_decode(out_ids, skip_special_tokens=True)

    @modal.fastapi_endpoint(method="POST")
    def backtranslate(self, item: dict):
        text = item.get("text", "")
        sentences = _split_sentences(text)
        if not sentences:
            return {"backtranslated": ""}

        fr_sentences = self._translate_batch(sentences, self.tok_en_fr, self.model_en_fr)
        back_sentences = self._translate_batch(fr_sentences, self.tok_fr_en, self.model_fr_en)
        return {"backtranslated": " ".join(back_sentences)}


# --------------------------------------------------------------------------
# /embed — sentence-transformers/all-MiniLM-L6-v2
# Small enough that CPU inference is fine; the win here is purely avoiding
# a fresh model load on every request.
# --------------------------------------------------------------------------
@app.cls(
    image=image,
    volumes={MODEL_CACHE: volume},
    timeout=60,
    scaledown_window=SCALEDOWN_WINDOW,
)
class Embedder:
    @modal.enter()
    def load(self):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2", cache_folder=MODEL_CACHE
        )

    @modal.fastapi_endpoint(method="POST")
    def embed(self, item: dict):
        text = item.get("text", "")
        vec = self.model.encode(text).tolist()
        return {"embedding": vec}
