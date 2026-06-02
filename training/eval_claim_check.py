"""Evaluate the claim-check pipeline on a claim-verification benchmark (LIAR).

Runs three configurations over the same labeled claims and reports
classification metrics for each, so the writeup can show (a) that
retrieval+stance verification beats a zero-shot LLM judgment, and (b)
what the adversarial-retrieval stress-test adds:

  1. claim-check          — retrieval + NLI stance + adversarial probe
  2. claim-check-no-adv    — ablation: retrieval + NLI stance, no adversarial
  3. zero-shot-baseline    — a single LLM call rating the claim 0-100

Usage:

    python -m training.eval_claim_check \
        --path data/eval/liar_dev.jsonl \
        --limit 40 --concurrency 4 \
        --out eval_claim_check_liar.json

Requires OPENROUTER/MISTRAL + TAVILY keys in the environment (.env). Each
example costs a few API calls per config, so start with a small --limit.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from backend.eval.datasets import load_jsonl
from backend.eval.metrics import compute_metrics, format_summary
from backend.eval.runner import EvalExample, claim_check_pipeline, predictions_to_dict, run_eval
from backend.models import (
    ContentAnalysis,
    CrossReferenceResult,
    DomainReputation,
    ReliabilityReport,
)
from backend.reliability.claims import _parse_json
from backend.reliability.llm_client import chat_json
from backend.reliability.scorer import _verdict
from backend.config import settings

ZERO_SHOT_SYSTEM = """You are a fact-checking assistant. Given a single
factual claim, judge how likely it is to be TRUE based on your own
knowledge, with no external search. Output STRICT JSON only:

{"truth_score": <integer 0-100>}

where 100 = almost certainly true, 0 = almost certainly false."""


def _wrap_score(score: float) -> ReliabilityReport:
    """Minimal ReliabilityReport carrying just an overall score, so the
    zero-shot baseline can flow through the shared eval runner/metrics."""
    score = round(max(0.0, min(100.0, score)), 1)
    return ReliabilityReport(
        overall_score=score,
        verdict=_verdict(score),  # type: ignore[arg-type]
        confidence=0.5,
        domain_reputation=DomainReputation(
            domain="(zero-shot)", score=50.0, bias="n/a", type="unknown", rationale="LLM-only baseline."
        ),
        content_analysis=ContentAnalysis(
            score=score, factuality=score, objectivity=50.0, transparency=50.0,
            sensationalism=50.0, main_claims=[], red_flags=[], citations_present=False,
            summary="Zero-shot LLM truthfulness judgment (no retrieval).",
        ),
        cross_reference=CrossReferenceResult(
            score=score, n_sources=0, n_high_quality=0, sources=[], consensus="no-data"
        ),
        explanation="Zero-shot baseline.",
        weights={"zero_shot": 1.0},
    )


async def _claim_check_adv(ex: EvalExample) -> ReliabilityReport:
    return await claim_check_pipeline(ex, adversarial=True)


async def _claim_check_noadv(ex: EvalExample) -> ReliabilityReport:
    return await claim_check_pipeline(ex, adversarial=False)


async def _zero_shot(ex: EvalExample) -> ReliabilityReport:
    claim = (ex.claim or ex.text or "").strip()
    if not settings.has_llm or not claim:
        return _wrap_score(50.0)
    try:
        raw = await asyncio.to_thread(
            chat_json, ZERO_SHOT_SYSTEM, f"CLAIM:\n{claim}\n\nReturn JSON only.",
            temperature=0.0, json_mode=True,
        )
        score = float(_parse_json(raw).get("truth_score", 50))
    except Exception:
        score = 50.0
    return _wrap_score(score)


CONFIGS = {
    "claim-check": _claim_check_adv,
    "claim-check-no-adv": _claim_check_noadv,
    "zero-shot-baseline": _zero_shot,
}


async def _amain(args: argparse.Namespace) -> int:
    examples = load_jsonl(args.path)
    if args.limit:
        examples = examples[: args.limit]
    print(f"Loaded {len(examples)} claims from {args.path}", file=sys.stderr)

    results: dict = {"dataset": str(args.path), "n": len(examples), "configs": {}}
    for name, pipeline in CONFIGS.items():
        print(f"\n=== {name} ===", file=sys.stderr)
        preds = await run_eval(
            examples, pipeline=pipeline, pipeline_name=name, concurrency=args.concurrency
        )
        scores = [p.predicted_score for p in preds]
        labels = [p.label for p in preds]
        # Drop errored predictions (None score) from the metric computation.
        pairs = [(s, l) for s, l in zip(scores, labels) if s is not None]
        m = compute_metrics([s for s, _ in pairs], [l for _, l in pairs], threshold=args.threshold)
        print(format_summary(m))
        results["configs"][name] = {
            "metrics": m.__dict__ if hasattr(m, "__dict__") else dict(m),
            "n_scored": len(pairs),
            "predictions": predictions_to_dict(preds),
        }

    if not args.no_output:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=lambda o: o.__dict__)
        print(f"\nWrote {args.out}", file=sys.stderr)

    # Compact comparison table.
    print("\nconfig                 acc    macroF1  rocAUC")
    for name, blob in results["configs"].items():
        mt = blob["metrics"]
        print(f"{name:<22s} {mt['accuracy']:.3f}  {mt['macro_f1']:.3f}    {mt.get('roc_auc') or float('nan'):.3f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m training.eval_claim_check")
    p.add_argument("--path", type=Path, default=Path("data/eval/liar_dev.jsonl"))
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--threshold", type=float, default=50.0)
    p.add_argument("--out", type=Path, default=Path("eval_claim_check_liar.json"))
    p.add_argument("--no-output", action="store_true")
    args = p.parse_args(argv)
    return asyncio.run(_amain(args))


if __name__ == "__main__":
    raise SystemExit(main())
