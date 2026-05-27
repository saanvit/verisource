"""Eval CLI: `python -m backend.eval --dataset dev`.

Runs the assess pipeline over a labeled dataset, prints a summary table,
and writes per-example predictions + aggregate metrics to JSON.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from backend.eval.datasets import load_dataset
from backend.eval.metrics import compute_metrics, format_summary
from backend.eval.runner import EvalPrediction, predictions_to_dict, run_eval


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m backend.eval")
    p.add_argument(
        "--dataset",
        default="dev",
        help="Dataset name: 'dev' (bundled) or 'jsonl' (provide --path)",
    )
    p.add_argument("--path", type=Path, help="Path to JSONL dataset (for --dataset jsonl)")
    p.add_argument("--limit", type=int, help="Process at most this many examples")
    p.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Number of examples to assess concurrently (default 4)",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=50.0,
        help="Score threshold for binary reliable/unreliable classification (default 50)",
    )
    p.add_argument(
        "--pipeline",
        choices=["default", "per-claim"],
        default="default",
        help=(
            "Which pipeline to run: 'default' (article-level analyzer + 1 search) "
            "or 'per-claim' (atomic-claim decomposition + per-claim retrieval)."
        ),
    )
    p.add_argument(
        "--skip-cross-reference",
        action="store_true",
        help=(
            "default pipeline only: skip Tavily. The per-claim pipeline always "
            "uses Tavily where available; pass --pipeline default to suppress it."
        ),
    )
    p.add_argument(
        "--nli",
        choices=["llm", "local"],
        default=None,
        help=(
            "Per-claim pipeline only: which stance labeler to use. 'llm' (default) "
            "calls Mistral; 'local' uses the fine-tuned NLI model from "
            "backend/reliability/nli.py (requires `pip install -r requirements-nli.txt`). "
            "Sets NLI_BACKEND for the duration of the run."
        ),
    )
    p.add_argument(
        "--calibration",
        type=Path,
        default=None,
        help=(
            "Path to a calibrator JSON (see training/fit_calibration.py). When "
            "set, the scorer post-processes overall_score through the calibrator. "
            "Use this to A/B raw vs calibrated outputs on the same dataset."
        ),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("eval_results.json"),
        help="Where to write the full JSON result (default eval_results.json)",
    )
    p.add_argument(
        "--no-output",
        action="store_true",
        help="Don't write a JSON results file (still prints summary)",
    )
    return p.parse_args(argv)


def _on_progress(done: int, total: int, pred: EvalPrediction) -> None:
    score = f"{pred.predicted_score:5.1f}" if pred.predicted_score is not None else "  err"
    verdict = pred.predicted_verdict or "error"
    err = f"  [{pred.error}]" if pred.error else ""
    print(
        f"  [{done:>3d}/{total}] {pred.id:<28s} "
        f"label={pred.label:<14s} score={score} verdict={verdict}"
        f"  ({pred.elapsed_seconds:.1f}s){err}",
        file=sys.stderr,
        flush=True,
    )


async def _amain(args: argparse.Namespace) -> int:
    if args.nli:
        os.environ["NLI_BACKEND"] = args.nli
    if args.calibration:
        os.environ["CALIBRATION_PATH"] = str(args.calibration.resolve())
    examples = load_dataset(args.dataset, args.path)
    if args.limit:
        examples = examples[: args.limit]
    print(f"Loaded {len(examples)} examples from dataset={args.dataset}", file=sys.stderr)
    print(
        f"Running pipeline={args.pipeline} (concurrency={args.concurrency}, "
        f"skip_xref={args.skip_cross_reference}, "
        f"nli={os.getenv('NLI_BACKEND', 'llm')}, "
        f"calibration={'on' if args.calibration else 'off'})...",
        file=sys.stderr,
    )

    predictions = await run_eval(
        examples,
        pipeline_name=args.pipeline,
        concurrency=args.concurrency,
        skip_cross_reference=args.skip_cross_reference,
        progress=_on_progress,
    )

    valid = [p for p in predictions if p.predicted_score is not None]
    n_errors = len(predictions) - len(valid)
    print(file=sys.stderr)
    print(f"Completed: {len(valid)} ok, {n_errors} errors", file=sys.stderr)

    if not valid:
        print("No successful predictions — cannot compute metrics.", file=sys.stderr)
        return 1

    metrics = compute_metrics(
        [p.predicted_score for p in valid],  # type: ignore[misc]
        [p.label for p in valid],
        threshold=args.threshold,
    )
    print()
    print(format_summary(metrics))

    if not args.no_output:
        out = {
            "dataset": args.dataset,
            "path": str(args.path) if args.path else None,
            "n_examples": len(examples),
            "n_errors": n_errors,
            "threshold": args.threshold,
            "pipeline": args.pipeline,
            "skip_cross_reference": args.skip_cross_reference,
            "nli_backend": os.getenv("NLI_BACKEND", "llm"),
            "calibration": str(args.calibration) if args.calibration else None,
            "metrics": metrics.to_dict(),
            "predictions": predictions_to_dict(predictions),
        }
        args.output.write_text(json.dumps(out, indent=2))
        print(f"\nWrote {args.output}", file=sys.stderr)

    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    return asyncio.run(_amain(args))


if __name__ == "__main__":
    sys.exit(main())
