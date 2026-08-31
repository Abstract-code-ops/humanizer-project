"""Turnitin scoring — manual, not automated (no public Turnitin API).

This module exists so the rest of the codebase can treat Turnitin scoring
uniformly with the automated proxy detector (same conceptual surface), even
though the value is filled in by hand in data/results.csv after a human
submits the humanized output to Turnitin themselves.
"""
import csv


def score(sample_id: str, tier: str, results_csv: str = "data/results.csv") -> float | None:
    """Read a manually-filled turnitin_score from results.csv, if present.

    Returns None if the row doesn't exist yet or the column is still empty
    (the common case — Turnitin scoring is a manual backfill step).
    """
    try:
        with open(results_csv, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["sample_id"] == sample_id and row["tier"] == tier:
                    val = row.get("turnitin_score", "").strip()
                    return float(val) if val else None
    except FileNotFoundError:
        return None
    return None
