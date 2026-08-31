"""Flask UI — thin client over the same Modal endpoints and the same
data/results.csv the CLI pipeline uses. No separate database.

Run: python ui/app.py   (add project root to sys.path below so this works
whether launched as `python ui/app.py` or `python -m ui.app`)
"""
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)  # so `from humanizers import ...` resolves

from flask import Flask, jsonify, render_template, request

from config import load_config
from humanizers import baseline, naive, naive_bt, paraphrase
from humanizers.adversarial_loop import AdversarialLoop
from humanizers._shared import cap_words, has_overlong_token, word_count
from detectors import proxy as proxy_detector
from evaluation import similarity as similarity_eval
from evaluation import readability as readability_eval

app = Flask(__name__)

RESULTS_CSV = os.path.join(ROOT, "data", "results.csv")
CSV_FIELDS = ["sample_id", "tier", "proxy_score", "turnitin_score", "similarity", "readability"]

# Tiers that are NOT surfaced in the UI (measured counterproductive, or
# internal-only) — see docs/BACKEND_SUMMARY.md section 3/9.
HIDDEN_TIERS = {"paraphrase_llm", "skill_gemini", "skill_deepseek", "commercial"}


# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template(
        "index.html",
        word_cap=load_config()["humanize_best_of"]["word_cap"],
        active_page="humanize",
    )


@app.route("/detect")
def detect_page():
    return render_template("detect.html", active_page="detect")


# --------------------------------------------------------------------------
# /api/humanize — the core action: 3-way best-of + 200-word cap
# --------------------------------------------------------------------------
@app.route("/api/humanize", methods=["POST"])
def api_humanize():
    data = request.get_json(force=True)
    raw_text = data.get("text", "")
    cfg = load_config()
    word_cap = cfg["humanize_best_of"]["word_cap"]

    if word_count(raw_text) > word_cap or has_overlong_token(raw_text, max_token_length=15):
        return jsonify({
            "error": "Input exceeds the 200-word limit or contains a token longer than 15 characters. Please shorten it before submitting."
        }), 400

    text = raw_text
    if not text.strip():
        return jsonify({"error": "empty input"}), 400

    detector_model = cfg["detector"]["model"]

    candidates = {}

    # paraphrase
    try:
        candidates["paraphrase"] = paraphrase.humanize(
            text, strength=cfg["paraphrase"]["default_strength"]
        )
    except Exception as e:
        app.logger.warning("paraphrase candidate failed: %s", e)

    # naive_bt
    try:
        candidates["naive_bt"] = naive_bt.humanize(text)
    except Exception as e:
        app.logger.warning("naive_bt candidate failed: %s", e)

    # Temporarily disable the adversarial path during deployment until the Modal
    # backend is stable. Keep the humanize flow to the simpler paraphrase and
    # backtranslate routes for now.
    # try:
    #     candidates["adversarial"] = AdversarialLoop().humanize(text)
    # except Exception as e:
    #     app.logger.warning("adversarial candidate failed: %s", e)

    if not candidates:
        return jsonify({
            "error": "all humanize backends failed (endpoints may be cold-starting — try again)"
        }), 502

    # Score each candidate, keep the lowest (most human) proxy_score.
    scored = []
    for tier_name, candidate_text in candidates.items():
        try:
            s = proxy_detector.score(candidate_text, model=detector_model)
            scored.append((s, tier_name, candidate_text))
        except Exception as e:
            app.logger.warning("scoring candidate %s failed: %s", tier_name, e)

    if not scored:
        return jsonify({"error": "detector scoring failed for all candidates"}), 502

    scored.sort(key=lambda x: x[0])
    best_score, best_tier, best_text = scored[0]

    try:
        sim = similarity_eval.similarity(text, best_text)
    except Exception:
        sim = None
    read = readability_eval.readability(best_text)

    _score_and_record(
        sample_id="ui-run",
        tier="best-of-three",
        proxy_score=best_score,
        similarity=sim,
        readability=read,
    )

    return jsonify({
        "humanized": best_text,
        "mode": "best-of-three",
        "proxy_score": round(best_score, 4),
        "similarity": round(sim, 4) if sim is not None else None,
        "readability": round(read, 2),
        "input_word_count": word_count(raw_text),
        "input_capped": word_count(raw_text) > word_cap,
        "word_cap": word_cap,
    })


# --------------------------------------------------------------------------
# /api/detect — proxy a detector score for arbitrary text
# --------------------------------------------------------------------------
@app.route("/api/detect", methods=["POST"])
def api_detect():
    data = request.get_json(force=True)
    text = data.get("text", "")
    if not text.strip():
        return jsonify({"error": "empty input"}), 400
    cfg = load_config()
    try:
        s = proxy_detector.score(text, model=cfg["detector"]["model"])
    except Exception as e:
        return jsonify({"error": f"detector unavailable: {e}"}), 502
    return jsonify({"ai_probability": round(s, 4)})


# --------------------------------------------------------------------------
# /api/similarity
# --------------------------------------------------------------------------
@app.route("/api/similarity", methods=["POST"])
def api_similarity():
    data = request.get_json(force=True)
    a, b = data.get("a", ""), data.get("b", "")
    if not a.strip() or not b.strip():
        return jsonify({"error": "both `a` and `b` are required"}), 400
    try:
        sim = similarity_eval.similarity(a, b)
    except Exception as e:
        return jsonify({"error": f"similarity service unavailable: {e}"}), 502
    return jsonify({"similarity": round(sim, 4)})


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _score_and_record(sample_id: str, tier: str, proxy_score: float,
                       similarity: float | None, readability: float):
    """Appends a UI-triggered run to results.csv (single source of truth —
    the UI does not maintain a separate store). Explicitly uses the
    configured detector model (desklib) rather than any default/mock, and
    writes+flushes the header for missing/empty files to avoid a CLI/UI
    header race (see docs/decision-log.md Iteration 14)."""
    file_exists = os.path.exists(RESULTS_CSV) and os.path.getsize(RESULTS_CSV) > 0
    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not file_exists:
            writer.writeheader()
            f.flush()
        writer.writerow({
            "sample_id": sample_id,
            "tier": tier,
            "proxy_score": round(proxy_score, 4),
            "turnitin_score": "",
            "similarity": round(similarity, 4) if similarity is not None else "",
            "readability": round(readability, 2),
        })
        f.flush()


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
