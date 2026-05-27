"""Fit a score calibrator from an eval result JSON.

Take any ``eval_results.json`` produced by ``python -m backend.eval``,
split predictions train/test, fit an isotonic calibrator on train, and
report before/after metrics on test.

    python -m training.fit_calibration \
        --predictions baseline_gonzaloa_50_default.json \
        --out models/calibration/default_gonzaloa.json \
        --train-frac 0.7

After fitting, the calibrator can be applied at inference by setting:

    export CALIBRATION_PATH=$PWD/models/calibration/default_gonzaloa.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from backend.eval.metrics import compute_metrics, format_summary
from backend.reliability.calibration import fit_isotonic, save


def _normalize_label(label: str) -> int | None:
    lo = label.strip().lower().replace("-", "_").replace(" ", "_")
    if lo in {"reliable", "true", "real", "credible", "trustworthy",
              "highly_reliable", "generally_reliable", "mostly_true"}:
        return 1
    if lo in {"unreliable", "fake", "false", "satire", "conspiracy",
              "questionable", "pants_fire", "barely_true"}:
        return 0
    return None  # mixed → exclude from calibration


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m training.fit_calibration")
    p.add_argument("--predictions", type=Path, required=True,
                   help="Path to an eval-harness results JSON.")
    p.add_argument("--out", type=Path, required=True,
                   help="Output calibrator JSON path.")
    p.add_argument("--train-frac", type=float, default=0.7,
                   help="Fraction of examples to use for fitting (rest is test). Default 0.7.")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    data = json.loads(args.predictions.read_text())
    preds = data.get("predictions") or []

    # Build (score, label_binary) pairs; drop errors and mixed labels.
    pairs: list[tuple[float, int]] = []
    for r in preds:
        score = r.get("predicted_score")
        label = r.get("label")
        if score is None or label is None:
            continue
        y = _normalize_label(str(label))
        if y is None:
            continue
        pairs.append((float(score), y))

    if len(pairs) < 10:
        print(f"Too few labeled predictions to calibrate: {len(pairs)}", file=sys.stderr)
        return 1

    rng = random.Random(args.seed)
    rng.shuffle(pairs)
    n_train = int(len(pairs) * args.train_frac)
    train, test = pairs[:n_train], pairs[n_train:]
    print(f"Fitting on {len(train)} examples, testing on {len(test)}.")

    cal = fit_isotonic([s for s, _ in train], [y for _, y in train])

    # Before/after metrics on the test split. compute_metrics consumes
    # label strings, so reverse our binary labels back to canonical names.
    def _to_strs(items: list[tuple[float, int]]) -> tuple[list[float], list[str]]:
        return [s for s, _ in items], ["reliable" if y == 1 else "unreliable" for _, y in items]

    raw_scores, raw_labels = _to_strs(test)
    cal_scores = [cal.apply(s) for s in raw_scores]

    metrics_before = compute_metrics(raw_scores, raw_labels)
    metrics_after = compute_metrics(cal_scores, raw_labels)

    print("\n=== Before calibration (test split) ===")
    print(format_summary(metrics_before))
    print("\n=== After calibration (test split) ===")
    print(format_summary(metrics_after))

    save(
        cal,
        args.out,
        metadata={
            "trained_on": str(args.predictions),
            "n_train": len(train),
            "n_test": len(test),
            "metrics_before": metrics_before.to_dict(),
            "metrics_after": metrics_after.to_dict(),
        },
    )
    print(f"\nCalibrator saved to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
