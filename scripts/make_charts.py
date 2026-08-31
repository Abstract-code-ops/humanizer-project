"""Regenerate static report PNGs from data/results.csv.

Usage: python scripts/make_charts.py --results data/results.csv --outdir data/charts
"""
import argparse
import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt


def load_rows(path):
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


def score_by_tier(rows, outdir):
    by_tier = defaultdict(list)
    for r in rows:
        if r["proxy_score"]:
            by_tier[r["tier"]].append(float(r["proxy_score"]))
    tiers = list(by_tier.keys())
    means = [sum(v) / len(v) for v in by_tier.values()]

    plt.figure(figsize=(8, 4.5))
    plt.bar(tiers, means, color="#e07a5f")
    plt.ylabel("mean proxy_score (AI probability)")
    plt.title("Mean AI-detector score by tier")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "score_by_tier.png"), dpi=150)
    plt.close()


def tradeoff_scatter(rows, outdir):
    plt.figure(figsize=(6, 6))
    for r in rows:
        if not r["proxy_score"] or not r["similarity"]:
            continue
        plt.scatter(float(r["similarity"]), float(r["proxy_score"]), alpha=0.6)
    plt.xlabel("similarity (higher = less meaning lost)")
    plt.ylabel("proxy_score (lower = more human)")
    plt.title("Evasion vs. meaning-retention tradeoff")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "tradeoff_scatter.png"), dpi=150)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="data/results.csv")
    parser.add_argument("--outdir", default="data/charts")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    rows = load_rows(args.results)
    score_by_tier(rows, args.outdir)
    tradeoff_scatter(rows, args.outdir)
    print(f"Charts written to {args.outdir}/")
