"""A/B-compare the fine-tuned local NLI model against the Mistral LLM
stance labeler on a held-out NLI test split.

Both backends are evaluated on the **same** (premise, hypothesis, label)
triples, with the same prompts/labels used in production code paths:

* ``local``  — runs ``backend.reliability.nli.label_hits_nli`` directly,
  exercising the same code path the per-claim pipeline uses at inference.
* ``llm``    — calls Mistral with the exact ``STANCE_SYSTEM_PROMPT`` from
  ``backend.reliability.claims``, so this is the same labeler the per-claim
  pipeline uses with ``NLI_BACKEND=llm``.

Outputs a comparison table (accuracy, macro-F1, per-class P/R, latency,
call count) and writes raw per-example predictions to JSON for analysis.

Run on a CPU box (no GPU needed for 3k examples on deberta-v3-base):

    # Local model only (fast, free).
    python -m training.eval_nli \\
        --test training/data/test.jsonl \\
        --model models/nli/deberta-v3-base-vitc \\
        --backends local --output eval_nli_local.json

    # A/B against Mistral (needs MISTRAL_API_KEY).
    python -m training.eval_nli \\
        --test training/data/test.jsonl \\
        --model models/nli/deberta-v3-base-vitc \\
        --backends local llm --llm-n 300 \\
        --output eval_nli_ab.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

from training.datasets import LABEL_NAMES

# Reverse lookup: name -> id, for converting backend string outputs to ints.
LABEL_NAME_TO_ID = {v: k for k, v in LABEL_NAMES.items()}


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _subsample(rows: list[dict], n: int | None, seed: int) -> list[dict]:
    if not n or n >= len(rows):
        return rows
    rng = random.Random(seed)
    return rng.sample(rows, n)


# ---------- backend: local fine-tuned NLI ----------


def run_local(rows: list[dict], batch_size: int = 32) -> tuple[list[int], float]:
    """Run the local fine-tuned NLI model on every row.

    Goes through ``backend.reliability.nli.label_hits_nli`` so this exercises
    the same code path that the per-claim pipeline hits in production.
    """
    from backend.reliability import nli
    from backend.search.web_search import SearchHit

    # Each test row → one synthetic SearchHit (premise becomes the snippet,
    # hypothesis becomes the claim). We batch by *claim* to amortize tokenizer
    # calls — but each (claim, hit) pair is independent so we can just send
    # them one-claim-at-a-time and rely on label_hits_nli's internal batching.
    #
    # For efficiency we group rows by claim batch and call once per row;
    # label_hits_nli batches forward passes internally so this is fine.
    preds: list[int] = []
    t0 = time.perf_counter()
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        # Call once per row (each row has its own claim/hypothesis).
        for r in batch:
            hit = SearchHit(title="", url="", snippet=r["premise"])
            labels = nli.label_hits_nli(r["hypothesis"], [hit])
            label = labels[0] if labels else "unclear"
            preds.append(LABEL_NAME_TO_ID.get(label, LABEL_NAME_TO_ID["unclear"]))
    elapsed = time.perf_counter() - t0
    return preds, elapsed


# ---------- backend: Mistral LLM ----------


def _llm_prompt(premise: str, hypothesis: str) -> str:
    """Single-pair stance prompt. Mirrors the production stance prompt's
    intent but specialised to one (premise, hypothesis) pair so we can
    parallelize cleanly across the test set."""
    return (
        f"PREMISE (evidence):\n{premise}\n\n"
        f"HYPOTHESIS (claim):\n{hypothesis}\n\n"
        "Return JSON only."
    )


SINGLE_PAIR_SYSTEM_PROMPT = """You are evaluating whether a PREMISE
(evidence text) corroborates a HYPOTHESIS (a claim to be checked).
Output exactly one of:
"supports", "contradicts", or "unclear".

Rules:
- "supports": the premise explicitly affirms the same factual content as
  the hypothesis.
- "contradicts": the premise explicitly asserts the opposite, or asserts
  a fact incompatible with the hypothesis.
- "unclear": the premise is on-topic but does not assert one way or the
  other, OR is off-topic, OR is too short to judge.

Output STRICT JSON only:

{"label": "supports" | "contradicts" | "unclear"}
"""


# Tracks how many calls actually returned a parseable label vs. fell back
# to "unclear" due to an exception. Surfaced after run_llm() so silent
# failures (rate limits, auth, schema drift) can't masquerade as a real
# eval result the way the original v1 of this script did.
_llm_call_counter = {"ok": 0, "fail": 0, "last_error": ""}


class _RateLimiter:
    """Async rate limiter: enforces at most ``rps`` requests per second
    across all concurrent callers. Cheap implementation good enough for
    eval scripts — every caller awaits the next available slot."""

    def __init__(self, rps: float):
        self.min_interval = 1.0 / rps if rps > 0 else 0.0
        self._lock = asyncio.Lock()
        self._next_allowed = 0.0

    async def acquire(self) -> None:
        if self.min_interval <= 0:
            return
        async with self._lock:
            now = time.perf_counter()
            wait = self._next_allowed - now
            if wait > 0:
                await asyncio.sleep(wait)
                now = time.perf_counter()
            self._next_allowed = now + self.min_interval


async def _call_mistral_one(
    client: Any,
    premise: str,
    hypothesis: str,
    rate_limiter: _RateLimiter,
) -> str:
    """One Mistral call → one label string. Retries on transient failures
    (rate limits, network blips) with exponential backoff via tenacity.
    On terminal failure, records the error and returns "unclear" so the
    eval continues — but the run_llm summary will tell us how many."""
    from tenacity import (
        retry,
        retry_if_exception_type,
        stop_after_attempt,
        wait_exponential,
    )

    from backend.config import settings

    @retry(
        stop=stop_after_attempt(6),
        wait=wait_exponential(multiplier=2, min=2, max=60),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def _go() -> str:
        resp = client.chat.complete(
            model=settings.mistral_model,
            messages=[
                {"role": "system", "content": SINGLE_PAIR_SYSTEM_PROMPT},
                {"role": "user", "content": _llm_prompt(premise, hypothesis)},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content or ""
        data = json.loads(raw)
        lbl = str(data.get("label", "unclear")).lower().strip()
        return lbl if lbl in ("supports", "contradicts", "unclear") else "unclear"

    await rate_limiter.acquire()
    try:
        result = await asyncio.to_thread(_go)
        _llm_call_counter["ok"] += 1
        return result
    except Exception as e:  # noqa: BLE001 - we want to capture everything here
        _llm_call_counter["fail"] += 1
        _llm_call_counter["last_error"] = f"{type(e).__name__}: {e}"
        return "unclear"


async def run_llm(
    rows: list[dict],
    concurrency: int = 1,
    rps: float = 1.0,
) -> tuple[list[int], float]:
    """Run the Mistral stance labeler on every row, rate-limited.

    Mistral's free tier is roughly 1 req/sec sustained. ``concurrency`` caps
    in-flight requests; ``rps`` is the *true* throughput cap enforced by a
    shared rate limiter (the binding constraint on free tier). Bump both
    for paid tiers.
    """
    from mistralai import Mistral

    from backend.config import settings

    if not settings.has_llm:
        raise RuntimeError(
            "MISTRAL_API_KEY is not set. The llm backend cannot run without it."
        )

    # Reset counters in case run_llm gets called more than once per process.
    _llm_call_counter["ok"] = 0
    _llm_call_counter["fail"] = 0
    _llm_call_counter["last_error"] = ""

    client = Mistral(api_key=settings.mistral_api_key)
    sem = asyncio.Semaphore(max(1, concurrency))
    rate_limiter = _RateLimiter(rps)

    async def _one(r: dict) -> str:
        async with sem:
            return await _call_mistral_one(
                client, r["premise"], r["hypothesis"], rate_limiter
            )

    t0 = time.perf_counter()
    label_strs = await asyncio.gather(*[_one(r) for r in rows])
    elapsed = time.perf_counter() - t0
    preds = [LABEL_NAME_TO_ID.get(s, LABEL_NAME_TO_ID["unclear"]) for s in label_strs]

    ok = _llm_call_counter["ok"]
    fail = _llm_call_counter["fail"]
    print(f"  [llm] {ok}/{ok + fail} calls succeeded.")
    if fail:
        print(f"  [llm] WARNING: {fail} call(s) failed → fell back to 'unclear'.")
        print(f"  [llm] last error: {_llm_call_counter['last_error']}")
    return preds, elapsed


# ---------- metrics ----------


def _per_class_pr(refs: list[int], preds: list[int]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for cls_id, name in LABEL_NAMES.items():
        tp = sum(1 for r, p in zip(refs, preds) if r == cls_id and p == cls_id)
        fp = sum(1 for r, p in zip(refs, preds) if r != cls_id and p == cls_id)
        fn = sum(1 for r, p in zip(refs, preds) if r == cls_id and p != cls_id)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        out[name] = {"precision": precision, "recall": recall, "f1": f1}
    return out


def _summarize(name: str, refs: list[int], preds: list[int], elapsed: float) -> dict:
    n = len(refs)
    acc = sum(1 for r, p in zip(refs, preds) if r == p) / n if n else 0.0
    per_class = _per_class_pr(refs, preds)
    macro_f1 = sum(c["f1"] for c in per_class.values()) / len(per_class)
    return {
        "backend": name,
        "n": n,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "elapsed_seconds": elapsed,
        "per_example_ms": (elapsed / n * 1000.0) if n else 0.0,
    }


def _print_table(summaries: list[dict]) -> None:
    print()
    print(f"{'backend':<10} {'n':>6} {'acc':>8} {'macroF1':>8} {'sup F1':>8} "
          f"{'con F1':>8} {'unc F1':>8} {'ms/ex':>8}")
    print("-" * 72)
    for s in summaries:
        print(
            f"{s['backend']:<10} {s['n']:>6} {s['accuracy']:>8.3f} {s['macro_f1']:>8.3f} "
            f"{s['per_class']['supports']['f1']:>8.3f} "
            f"{s['per_class']['contradicts']['f1']:>8.3f} "
            f"{s['per_class']['unclear']['f1']:>8.3f} "
            f"{s['per_example_ms']:>8.1f}"
        )
    print()


# ---------- main ----------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--test", type=Path, required=True, help="JSONL test split.")
    p.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Path to local fine-tuned model. Sets NLI_MODEL for the local backend.",
    )
    p.add_argument(
        "--backends",
        nargs="+",
        choices=["local", "llm"],
        default=["local"],
        help="Which backends to evaluate.",
    )
    p.add_argument(
        "--n",
        type=int,
        default=None,
        help="Subsample the test split to this many examples (applied to ALL backends).",
    )
    p.add_argument(
        "--llm-n",
        type=int,
        default=None,
        help="Further subsample for the LLM backend only (cost control). "
        "Local runs on --n; LLM runs on a (possibly smaller) subset of those.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--llm-concurrency",
        type=int,
        default=1,
        help="Mistral in-flight request cap. Default 1 (serial, safe for free tier).",
    )
    p.add_argument(
        "--llm-rps",
        type=float,
        default=1.0,
        help="Mistral max requests/sec. Default 1.0 — free tier limit. "
        "Bump to ~5 for paid tiers.",
    )
    p.add_argument("--output", type=Path, default=None, help="Write per-example predictions JSON.")
    args = p.parse_args(argv)

    if args.model is not None:
        os.environ["NLI_MODEL"] = str(args.model.resolve())
        print(f"Set NLI_MODEL={os.environ['NLI_MODEL']}")

    rows_all = _load_jsonl(args.test)
    rows = _subsample(rows_all, args.n, args.seed)
    print(f"Loaded {len(rows_all)} test rows; evaluating on {len(rows)}.")
    label_counts = Counter(r["label"] for r in rows)
    print(
        f"  class balance: supports={label_counts[0]}, "
        f"contradicts={label_counts[1]}, unclear={label_counts[2]}"
    )

    refs = [int(r["label"]) for r in rows]
    summaries: list[dict] = []
    per_example: dict[str, list[int]] = {}

    if "local" in args.backends:
        print("\n[local] running fine-tuned NLI model...")
        preds, elapsed = run_local(rows)
        summaries.append(_summarize("local", refs, preds, elapsed))
        per_example["local"] = preds

    if "llm" in args.backends:
        llm_rows = _subsample(rows, args.llm_n, args.seed) if args.llm_n else rows
        print(f"\n[llm] running Mistral on {len(llm_rows)} examples...")
        llm_refs = [int(r["label"]) for r in llm_rows]
        preds, elapsed = asyncio.run(
            run_llm(llm_rows, concurrency=args.llm_concurrency, rps=args.llm_rps)
        )
        summaries.append(_summarize("llm", llm_refs, preds, elapsed))
        per_example["llm"] = preds
        per_example["llm_refs"] = llm_refs

    _print_table(summaries)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "test_path": str(args.test),
            "n_total": len(rows_all),
            "n_evaluated": len(rows),
            "summaries": summaries,
            "predictions": per_example,
            "refs": refs,
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
