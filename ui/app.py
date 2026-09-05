"""Flask UI — thin client over the same Modal endpoints and the same
data/results.csv the CLI pipeline uses. No separate database for the public
UI's needs — the admin observability log below is a deliberate, additive
exception (see its own docstring for why).

Run: python ui/app.py   (add project root to sys.path below so this works
whether launched as `python ui/app.py` or `python -m ui.app`)
"""
import csv
import hmac
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from functools import wraps

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
# Admin observability log — a separate, additive JSONL file, NOT a change to
# results.csv's schema. results.csv stays exactly as the CLI pipeline
# expects it; this log exists purely so an admin can see how individual
# calls performed (which tier won, adversarial's iteration trace, how often
# the Gemini call gets used) — detail that doesn't belong in the CLI's
# results schema and that the CLI doesn't need.
#
# JSONL (one JSON object per line) rather than a DB: append-only, no schema
# migration to manage, trivial to read/tail/grep by hand. At real research-
# project volume this is plenty; it does grow unbounded over time with no
# rotation built in here — periodically archive or truncate
# data/admin_log.jsonl if it gets large.
#
# Contains full input/output text for each call. That's necessary to make
# the log useful, but means treating it with the same care as any log of
# user-submitted text.
# --------------------------------------------------------------------------
ADMIN_LOG = os.path.join(ROOT, "data", "admin_log.jsonl")
ADMIN_LOG_DISPLAY_LIMIT = 200  # most recent N entries shown in the /admin page

# Fails closed: unset ADMIN_TOKEN means the admin routes refuse to serve
# anything, rather than defaulting to open. Set this in Render's env vars
# (or your local .env) — never commit it.
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN")


def require_admin(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not ADMIN_TOKEN:
            return jsonify({"error": "admin routes are disabled (ADMIN_TOKEN not set)"}), 503
        supplied = request.args.get("token", "")
        if not supplied:
            auth_header = request.headers.get("Authorization", "")
            if auth_header.startswith("Bearer "):
                supplied = auth_header[len("Bearer "):]
        if not supplied or not hmac.compare_digest(supplied, ADMIN_TOKEN):
            return jsonify({"error": "unauthorized"}), 401
        return view(*args, **kwargs)
    return wrapped


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

    if word_count(raw_text) > word_cap or has_overlong_token(raw_text, max_token_length=20):
        return jsonify({
            "error": "Input exceeds the 200-word limit or contains a token longer than 20 characters. Please shorten it before submitting."
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

    # Adversarial — re-enabled now that the Modal backend runs the whole
    # rewrite -> detect -> repeat loop server-side in a single call (see
    # deploy/modal_app.py's Adversarial class, POST /adversarial). Keep a
    # reference to the instance (not just its return value) so its
    # last_response — the full iteration trace, llm_calls_used, etc. — is
    # available below for the admin log.
    adv_loop = AdversarialLoop()
    try:
        candidates["adversarial"] = adv_loop.humanize(text)
    except Exception as e:
        app.logger.warning("adversarial candidate failed: %s", e)

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

    _append_admin_log({
        "id": uuid.uuid4().hex[:12],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "endpoint": "humanize",
        "input_word_count": word_count(raw_text),
        "input_capped": word_count(raw_text) > word_cap,
        "candidate_scores": {tier: round(s, 4) for s, tier, _ in scored},
        "winning_tier": best_tier,
        "final_score": round(best_score, 4),
        "similarity": round(sim, 4) if sim is not None else None,
        "readability": round(read, 2),
        # Present whenever the adversarial candidate ran at all, whether or
        # not it ended up winning — this is the detail an admin actually
        # wants visibility into (iteration trace, Gemini call count).
        "adversarial_detail": adv_loop.last_response,
    })

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

    _append_admin_log({
        "id": uuid.uuid4().hex[:12],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "endpoint": "detect",
        "input_word_count": word_count(text),
        "final_score": round(s, 4),
    })

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
# Admin — read-only view over admin_log.jsonl. Token-gated (see
# require_admin above); disabled entirely if ADMIN_TOKEN isn't set.
# --------------------------------------------------------------------------
@app.route("/admin")
@require_admin
def admin_dashboard():
    entries = _read_admin_log()
    recent = list(reversed(entries[-ADMIN_LOG_DISPLAY_LIMIT:]))
    stats = _compute_admin_stats(entries)
    return render_template(
        "admin.html",
        entries=recent,
        stats=stats,
        total_logged=len(entries),
        shown=len(recent),
        token=request.args.get("token", ""),
        active_page="admin",
    )


@app.route("/admin/api/calls")
@require_admin
def admin_api_calls():
    entries = _read_admin_log()
    limit = _clamp_int(request.args.get("limit", ADMIN_LOG_DISPLAY_LIMIT), 1, 2000, ADMIN_LOG_DISPLAY_LIMIT)
    recent = list(reversed(entries[-limit:]))
    return jsonify({
        "total_logged": len(entries),
        "returned": len(recent),
        "calls": recent,
        "stats": _compute_admin_stats(entries),
    })


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _clamp_int(value, lo, hi, default) -> int:
    try:
        v = int(value)
    except (TypeError, ValueError):
        v = default
    return max(lo, min(hi, v))


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


def _append_admin_log(entry: dict):
    """Best-effort append to admin_log.jsonl. Never raises — a logging
    failure should not take down the actual user-facing request."""
    try:
        os.makedirs(os.path.dirname(ADMIN_LOG), exist_ok=True)
        with open(ADMIN_LOG, "a") as f:
            f.write(json.dumps(entry) + "\n")
            f.flush()
    except Exception as e:
        app.logger.warning("admin log write failed: %s", e)


def _read_admin_log() -> list[dict]:
    if not os.path.exists(ADMIN_LOG):
        return []
    entries = []
    with open(ADMIN_LOG) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # skip a corrupted line rather than fail the whole read
    return entries


def _compute_admin_stats(entries: list[dict]) -> dict:
    humanize_entries = [e for e in entries if e.get("endpoint") == "humanize"]
    detect_entries = [e for e in entries if e.get("endpoint") == "detect"]

    tier_wins: dict[str, int] = {}
    for e in humanize_entries:
        tier = e.get("winning_tier")
        if tier:
            tier_wins[tier] = tier_wins.get(tier, 0) + 1

    adv_details = [
        e["adversarial_detail"] for e in humanize_entries
        if e.get("adversarial_detail")
    ]
    adv_target_reached = sum(1 for d in adv_details if d.get("target_reached"))
    adv_iterations = [d.get("iterations_used", 0) for d in adv_details]
    adv_llm_calls = [d.get("llm_calls_used", 0) for d in adv_details]
    adv_deltas = [
        d.get("original_ai_probability", 0) - d.get("ai_probability", 0)
        for d in adv_details
        if d.get("original_ai_probability") is not None and d.get("ai_probability") is not None
    ]

    def _avg(values):
        return round(sum(values) / len(values), 3) if values else None

    return {
        "total_calls": len(entries),
        "humanize_calls": len(humanize_entries),
        "detect_calls": len(detect_entries),
        "tier_wins": tier_wins,
        "adversarial_runs": len(adv_details),
        "adversarial_target_reached": adv_target_reached,
        "adversarial_target_reached_rate": (
            round(adv_target_reached / len(adv_details), 3) if adv_details else None
        ),
        "avg_adversarial_iterations": _avg(adv_iterations),
        "avg_llm_calls_per_run": _avg(adv_llm_calls),
        "total_llm_calls": sum(adv_llm_calls),
        "avg_score_drop": _avg(adv_deltas),
    }


if __name__ == "__main__":
    # threaded=True matters here, not just for perf: with the default
    # single-threaded dev server, a slow /api/humanize or /api/detect call
    # (blocked on requests.post() to Modal for many seconds) makes the
    # *entire server* unresponsive to any other request — including a
    # simple page refresh (GET /), which is why a refresh mid-request looked
    # like a hung white screen: it wasn't stuck, it was queued behind the
    # in-flight request with nowhere to go until that one finished. This
    # doesn't cancel the in-flight Modal call itself (see notes in
    # docs/decision-log.md on that) — it just stops one slow request from
    # blocking every other request, including page loads.
    app.run(debug=True, host="0.0.0.0", port=5000, threaded=True)
