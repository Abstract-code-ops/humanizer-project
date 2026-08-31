"""Modal app: hosts detect / paraphrase / embed / backtranslate / health.

Deploy: .venv/bin/modal deploy deploy/modal_app.py
App name: humanizer-models (workspace: set by your Modal account)

Each function gets its OWN stable HTTPS URL (Modal fastapi_endpoint behavior —
not a shared base path). Record the five URLs in .env as MODAL_DETECT_URL,
MODAL_PARAPHRASE_URL, MODAL_EMBED_URL, MODAL_BACKTRANSLATE_URL, MODAL_HEALTH_URL.

Bug fixes baked in here (see docs/decision-log.md for the full story — do not
regress these):
  - Iteration 6: image needs sentencepiece + protobuf + tiktoken, or T5's fast
    tokenizer silently falls through to a broken tiktoken conversion path.
  - Iteration 6: desklib's forward() must accept & drop **kwargs (token_type_ids)
    since DeBERTa-v3 doesn't need it at inference.
  - Iteration 6: pin transformers<4.49 (desklib custom PreTrainedModel subclass
    breaks on >=4.49's all_tied_weights_keys).
  - Iteration 12/13: translate/paraphrase ONE SENTENCE AT A TIME. Do not chunk
    multiple sentences into one ~400-token block — both MarianMT and
    T5_Paraphrase_Paws are sentence-level models and go out-of-distribution on
    multi-sentence chunks, producing repetition loops / filler-token gibberish.
  - Iteration 13: do not force min_length / length_penalty / no_repeat_ngram_size
    on MarianMT or T5 decode — those combinations are the documented trigger for
    degenerate output. Use plain beam decoding for MarianMT, and the model's
    official sampling recipe (do_sample, top_k, top_p, temperature scaled by
    `strength`) for T5_Paraphrase_Paws.
  - Iteration 13 (security hardening): `strength` is validated + clamped to
    int 1..5; both paraphrase and backtranslate cap the per-request sentence
    loop at 200 sentences (DoS guard).

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
# --------------------------------------------------------------------------
_DETECTOR_LOADERS = {}


def _load_desklib():
    """Loads desklib/ai-text-detector-v1.01 (custom raw-logit head, not a
    *ForSequenceClassification checkpoint — needs a custom module class)."""
    import torch
    from transformers import AutoTokenizer, AutoModel

    class DesklibAIDetectionModel(torch.nn.Module):
        def __init__(self, base_model_name):
            super().__init__()
            self.base = AutoModel.from_pretrained(base_model_name)
            hidden = self.base.config.hidden_size
            self.classifier = torch.nn.Linear(hidden, 1)

        def forward(self, input_ids=None, attention_mask=None, **kwargs):
            # **kwargs absorbs-and-drops token_type_ids (DeBERTa doesn't need
            # it at inference) — this is the Iteration 6 fix.
            out = self.base(input_ids=input_ids, attention_mask=attention_mask)
            pooled = out.last_hidden_state.mean(dim=1)
            logit = self.classifier(pooled).squeeze(-1)
            return torch.sigmoid(logit)

    name = "desklib/ai-text-detector-v1.01"
    tok = AutoTokenizer.from_pretrained(name, cache_dir=MODEL_CACHE)
    model = DesklibAIDetectionModel(name)
    model.eval()
    return tok, model


def _load_radar():
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    name = "TrustSafeAI/RADAR-Vicuna-7B"
    tok = AutoTokenizer.from_pretrained(name, cache_dir=MODEL_CACHE)
    model = AutoModelForSequenceClassification.from_pretrained(name, cache_dir=MODEL_CACHE)
    model.eval()
    return tok, model


@app.function(image=image, gpu="T4", volumes={MODEL_CACHE: volume}, timeout=120)
@modal.fastapi_endpoint(method="POST")
def detect(item: dict):
    import torch

    text = item.get("text", "")
    model_name = item.get("model", "desklib/ai-text-detector-v1.01")

    if model_name == "desklib/ai-text-detector-v1.01":
        tok, model = _load_desklib()
        inputs = tok(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            prob = model(**inputs).item()
        return {"ai_probability": prob}

    elif model_name == "TrustSafeAI/RADAR-Vicuna-7B":
        tok, model = _load_radar()
        inputs = tok(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
            # class 0 = AI, per docs/architecture.md label-direction note.
            prob = probs[0][0].item()
        return {"ai_probability": prob}

    return {"error": f"unknown model {model_name}"}, 400


# --------------------------------------------------------------------------
# /paraphrase — Vamsi/T5_Paraphrase_Paws, sentence-by-sentence, sampling recipe
# --------------------------------------------------------------------------
@app.function(image=image, gpu="T4", volumes={MODEL_CACHE: volume}, timeout=180)
@modal.fastapi_endpoint(method="POST")
def paraphrase(item: dict):
    import torch
    from transformers import T5Tokenizer, T5ForConditionalGeneration

    text = item.get("text", "")
    strength = _clamp_strength(item.get("strength", 3))

    name = "Vamsi/T5_Paraphrase_Paws"
    tok = T5Tokenizer.from_pretrained(name, cache_dir=MODEL_CACHE)
    model = T5ForConditionalGeneration.from_pretrained(name, cache_dir=MODEL_CACHE)
    model.eval()

    # strength 1-5 scales sampling temperature/top_k/top_p — official sampling
    # recipe, no min_length / length_penalty / beam search (Iteration 13 fix).
    temperature = 0.6 + 0.15 * strength    # ~0.75 .. 1.35
    top_k = 40 + 10 * strength             # 50 .. 90
    top_p = 0.90 + 0.02 * strength         # 0.92 .. 1.00

    sentences = _split_sentences(text)
    out_sentences = []
    for sent in sentences:
        input_text = f"paraphrase: {sent} </s>"
        enc = tok.encode_plus(input_text, return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            out_ids = model.generate(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                do_sample=True,
                temperature=temperature,
                top_k=top_k,
                top_p=min(top_p, 1.0),
                max_length=256,
                early_stopping=True,
            )
        out_sentences.append(tok.decode(out_ids[0], skip_special_tokens=True))

    return {"paraphrased": " ".join(out_sentences)}


# --------------------------------------------------------------------------
# /backtranslate — MarianMT en->fr->en, sentence-by-sentence, plain beam decode
# --------------------------------------------------------------------------
@app.function(image=image, volumes={MODEL_CACHE: volume}, timeout=180)
@modal.fastapi_endpoint(method="POST")
def backtranslate(item: dict):
    import torch
    from transformers import MarianMTModel, MarianTokenizer

    text = item.get("text", "")

    en_fr_name = "Helsinki-NLP/opus-mt-en-fr"
    fr_en_name = "Helsinki-NLP/opus-mt-fr-en"
    tok_en_fr = MarianTokenizer.from_pretrained(en_fr_name, cache_dir=MODEL_CACHE)
    model_en_fr = MarianMTModel.from_pretrained(en_fr_name, cache_dir=MODEL_CACHE)
    tok_fr_en = MarianTokenizer.from_pretrained(fr_en_name, cache_dir=MODEL_CACHE)
    model_fr_en = MarianMTModel.from_pretrained(fr_en_name, cache_dir=MODEL_CACHE)
    model_en_fr.eval()
    model_fr_en.eval()

    def _translate(sentence: str, tok, model) -> str:
        enc = tok(sentence, return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            # Plain beam decoding, no no_repeat_ngram_size / repetition_penalty
            # (Iteration 12 fix — those triggered runaway repetition loops).
            out_ids = model.generate(
                **enc, num_beams=4, do_sample=False, early_stopping=True, max_length=256
            )
        return tok.decode(out_ids[0], skip_special_tokens=True)

    sentences = _split_sentences(text)
    out_sentences = []
    for sent in sentences:
        fr = _translate(sent, tok_en_fr, model_en_fr)
        back = _translate(fr, tok_fr_en, model_fr_en)
        out_sentences.append(back)

    return {"backtranslated": " ".join(out_sentences)}


# --------------------------------------------------------------------------
# /embed — sentence-transformers/all-MiniLM-L6-v2
# --------------------------------------------------------------------------
@app.function(image=image, volumes={MODEL_CACHE: volume}, timeout=60)
@modal.fastapi_endpoint(method="POST")
def embed(item: dict):
    from sentence_transformers import SentenceTransformer

    text = item.get("text", "")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", cache_folder=MODEL_CACHE)
    vec = model.encode(text).tolist()
    return {"embedding": vec}
