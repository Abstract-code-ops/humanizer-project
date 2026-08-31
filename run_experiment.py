"""CLI pipeline: runs the fixed source texts through every tier, appends
scores to data/results.csv. This is the canonical experiment/record — the
web UI is a thin client over the same Modal endpoints and the same CSV.

Only this module orchestrates across layers (generators / humanizers /
detectors / evaluation).

Usage:
    python run_experiment.py                     # run all tiers, all samples
    python run_experiment.py --tier paraphrase    # single tier
    python run_experiment.py --sample pigeon      # single sample
"""
import argparse
import csv
import os

from config import load_config
from humanizers import baseline, naive, naive_bt, paraphrase, paraphrase_llm
from humanizers.adversarial_loop import AdversarialLoop
from humanizers.skill_humanizer import SkillHumanizer
from detectors import proxy as proxy_detector
from evaluation import similarity as similarity_eval
from evaluation import readability as readability_eval

CSV_FIELDS = ["sample_id", "tier", "proxy_score", "turnitin_score", "similarity", "readability"]


def _resolve_tier(tier_name: str, cfg: dict):
    """Returns a callable text -> str for the given tier name."""
    if tier_name == "baseline":
        return baseline.humanize
    if tier_name == "naive":
        naive_cfg = cfg["naive"]
        return lambda t: naive.humanize(
            t, substitution_rate=naive_cfg["substitution_rate"], seed=naive_cfg["seed"]
        )
    if tier_name == "naive_bt":
        return naive_bt.humanize
    if tier_name == "paraphrase":
        strength = cfg["paraphrase"]["default_strength"]
        return lambda t: paraphrase.humanize(t, strength=strength)
    if tier_name == "paraphrase_llm":
        return paraphrase_llm.humanize
    if tier_name == "adversarial":
        loop = AdversarialLoop()
        return loop.humanize
    if tier_name == "skill_gemini":
        return SkillHumanizer(provider="skill_gemini").humanize
    if tier_name == "skill_deepseek":
        return SkillHumanizer(provider="skill_deepseek").humanize
    raise ValueError(f"Unknown tier: {tier_name}")


def _load_source_text(path: str) -> str:
    with open(path, "r") as f:
        return f.read().strip()


def _append_row(results_csv: str, row: dict):
    file_exists = os.path.exists(results_csv) and os.path.getsize(results_csv) > 0
    os.makedirs(os.path.dirname(results_csv), exist_ok=True)
    with open(results_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not file_exists:
            writer.writeheader()
            f.flush()
        writer.writerow(row)
        f.flush()


def run(tier_filter: str | None = None, sample_filter: str | None = None,
        dry_run: bool = False):
    cfg = load_config()
    results_csv = cfg["results_csv"]

    samples = cfg["source_texts"]
    if sample_filter:
        samples = [s for s in samples if s["id"] == sample_filter]

    tiers = cfg["tiers"]
    if tier_filter:
        tiers = [t for t in tiers if t == tier_filter]

    for sample in samples:
        original = _load_source_text(sample["path"])
        sample_id = sample["id"]

        for tier_name in tiers:
            if dry_run:
                print(f"[dry-run] would humanize {sample_id} with {tier_name}")
                continue

            humanize_fn = _resolve_tier(tier_name, cfg)
            humanized = humanize_fn(original)

            proxy_score = proxy_detector.score(humanized, model=cfg["detector"]["model"])
            sim = similarity_eval.similarity(original, humanized)
            read = readability_eval.readability(humanized)

            out_dir = f"data/outputs/{tier_name}"
            os.makedirs(out_dir, exist_ok=True)
            with open(f"{out_dir}/{sample_id}.txt", "w") as f:
                f.write(humanized)

            _append_row(results_csv, {
                "sample_id": sample_id,
                "tier": tier_name,
                "proxy_score": round(proxy_score, 4),
                "turnitin_score": "",   # backfilled manually later
                "similarity": round(sim, 4),
                "readability": round(read, 2),
            })
            print(f"{sample_id:20s} {tier_name:16s} proxy={proxy_score:.4f} sim={sim:.4f} read={read:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", default=None, help="run only this tier")
    parser.add_argument("--sample", default=None, help="run only this sample")
    parser.add_argument("--dry-run", action="store_true", help="don't call any APIs")
    args = parser.parse_args()
    run(tier_filter=args.tier, sample_filter=args.sample, dry_run=args.dry_run)
