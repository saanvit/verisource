"""Evaluate the citation-audit stance step on the synthetic benchmark.

For each (citing_sentence, source_text) pair, run the same stance labeler
the citation-audit pipeline uses and check whether it correctly tells a
supporting source from a mismatched one. Reports detection metrics for the
actionable signal — catching UNSUPPORTED citations — plus a confusion
matrix.

Usage:
    python -m training.build_citation_benchmark
    python -m training.eval_citation_audit \
        --path data/eval/citation_audit_dev.jsonl \
        --out eval_citation_audit.json

A "supported" prediction means the labeler said the source supports the
sentence; anything else (contradicts/unclear) is treated as "unsupported".
Uses the LLM stance labeler when available (the pipeline's default for
long-source entailment), else the local NLI backend.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from backend.config import settings
from backend.reliability.claims import _label_hits, _label_hits_llm
from backend.search.web_search import SearchHit


def _load(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


async def _label(sentence: str, source: str) -> str:
    hit = SearchHit(title="source", url="https://example.com/source", snippet=source)
    labels = (
        await _label_hits_llm(sentence, [hit])
        if settings.has_llm
        else await _label_hits(sentence, [hit])
    )
    return labels[0] if labels else "unclear"


async def _amain(args: argparse.Namespace) -> int:
    rows = _load(args.path)
    if args.limit:
        rows = rows[: args.limit]
    print(f"Loaded {len(rows)} citation pairs from {args.path}", file=sys.stderr)

    sem = asyncio.Semaphore(args.concurrency)

    async def one(r: dict) -> dict:
        async with sem:
            stance = await _label(r["citing_sentence"], r["source_text"])
        predicted = "supported" if stance == "supports" else "unsupported"
        return {**r, "stance": stance, "predicted": predicted}

    results = await asyncio.gather(*[one(r) for r in rows])

    # Confusion matrix with UNSUPPORTED as the positive (detection) class.
    tp = sum(1 for r in results if r["label"] == "unsupported" and r["predicted"] == "unsupported")
    fn = sum(1 for r in results if r["label"] == "unsupported" and r["predicted"] == "supported")
    fp = sum(1 for r in results if r["label"] == "supported" and r["predicted"] == "unsupported")
    tn = sum(1 for r in results if r["label"] == "supported" and r["predicted"] == "supported")
    n = len(results)
    acc = (tp + tn) / n if n else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    print(f"\nn={n}  backend={'llm' if settings.has_llm else 'local-nli'}")
    print("Detecting UNSUPPORTED citations (positive class = unsupported):")
    print(f"  accuracy={acc:.3f}  precision={prec:.3f}  recall={rec:.3f}  F1={f1:.3f}")
    print("  confusion (rows=true, cols=pred):")
    print(f"               pred-unsupported  pred-supported")
    print(f"  true-unsup        {tp:>3d}              {fn:>3d}")
    print(f"  true-supported    {fp:>3d}              {tn:>3d}")

    out = {
        "dataset": str(args.path),
        "n": n,
        "backend": "llm" if settings.has_llm else "local-nli",
        "metrics": {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
                    "tp": tp, "fn": fn, "fp": fp, "tn": tn},
        "predictions": results,
    }
    if not args.no_output:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"\nWrote {args.out}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m training.eval_citation_audit")
    p.add_argument("--path", type=Path, default=Path("data/eval/citation_audit_dev.jsonl"))
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--out", type=Path, default=Path("eval_citation_audit.json"))
    p.add_argument("--no-output", action="store_true")
    args = p.parse_args(argv)
    return asyncio.run(_amain(args))


if __name__ == "__main__":
    raise SystemExit(main())
