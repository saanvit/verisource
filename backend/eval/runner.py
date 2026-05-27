"""Run the assessment pipeline over a labeled dataset.

The runner is intentionally pipeline-agnostic: pass any async callable
``pipeline(example) -> ReliabilityReport`` and the runner takes care of
concurrency, error capture, and prediction collection. The default pipeline
is the production stack (extract → domain reputation → analyzer →
cross_reference → aggregate), so new variants can be A/B-compared by
swapping the callable.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import asdict, dataclass, field
from typing import Awaitable, Callable

import httpx

from backend.eval.datasets import EvalExample
from backend.models import ReliabilityReport
from backend.reliability.analyzer import analyze_content
from backend.reliability.content_extractor import (
    ExtractedContent,
    extract_from_text,
    extract_from_url,
)
from backend.reliability.cross_reference import cross_reference
from backend.reliability.domain_reputation import lookup as lookup_domain
from backend.reliability.pipeline_per_claim import assess_per_claim
from backend.reliability.scorer import aggregate

Pipeline = Callable[[EvalExample], Awaitable[ReliabilityReport]]


@dataclass
class EvalPrediction:
    id: str
    label: str
    predicted_score: float | None
    predicted_verdict: str | None
    confidence: float | None
    elapsed_seconds: float
    error: str | None = None
    report: dict | None = field(default=None, repr=False)


async def _content_for(example: EvalExample) -> ExtractedContent:
    """Build the ExtractedContent for an example.

    If both ``text`` and ``url`` are present, the text is used as the article
    body and the URL is only attached so domain lookups and cross-reference
    exclusion work — this lets the dev set exercise domain priors without
    actually fetching anything over the network.
    """
    if example.text:
        content = extract_from_text(example.text)
        if example.url:
            content = ExtractedContent(
                url=example.url,
                title=content.title,
                text=content.text,
                n_links=content.n_links,
                has_citations=content.has_citations,
            )
        return content
    if example.url:
        return await extract_from_url(example.url)
    raise ValueError(f"{example.id}: no text or url")  # pragma: no cover


async def default_pipeline(
    example: EvalExample, *, skip_cross_reference: bool = False
) -> ReliabilityReport:
    """Production pipeline: domain prior + article-level analyzer + 1 cross-ref search."""
    content = await _content_for(example)
    domain = lookup_domain(example.url)
    analysis = await analyze_content(content)
    if skip_cross_reference:
        from backend.models import CrossReferenceResult

        xref = CrossReferenceResult(
            score=50.0, n_sources=0, n_high_quality=0, sources=[], consensus="no-data"
        )
    else:
        xref = await cross_reference(content, analysis, example.claim)
    return aggregate(domain, analysis, xref, has_text=bool(content.text.strip()))


async def per_claim_pipeline(example: EvalExample) -> ReliabilityReport:
    """Per-claim pipeline: decompose article → verify each claim independently."""
    content = await _content_for(example)
    return await assess_per_claim(content, user_claim=example.claim)


async def _run_one(
    example: EvalExample,
    pipeline: Pipeline,
    sem: asyncio.Semaphore,
) -> EvalPrediction:
    async with sem:
        start = time.perf_counter()
        try:
            report = await pipeline(example)
        except (httpx.HTTPError, ValueError, RuntimeError) as exc:
            return EvalPrediction(
                id=example.id,
                label=example.label,
                predicted_score=None,
                predicted_verdict=None,
                confidence=None,
                elapsed_seconds=time.perf_counter() - start,
                error=f"{type(exc).__name__}: {exc}",
            )
        elapsed = time.perf_counter() - start
        return EvalPrediction(
            id=example.id,
            label=example.label,
            predicted_score=report.overall_score,
            predicted_verdict=report.verdict,
            confidence=report.confidence,
            elapsed_seconds=elapsed,
            report=report.model_dump(mode="json"),
        )


async def run_eval(
    examples: list[EvalExample],
    *,
    pipeline: Pipeline | None = None,
    pipeline_name: str = "default",
    concurrency: int = 4,
    skip_cross_reference: bool = False,
    progress: Callable[[int, int, EvalPrediction], None] | None = None,
) -> list[EvalPrediction]:
    """Run a labeled dataset through a pipeline.

    ``pipeline`` overrides ``pipeline_name``. With neither, the default
    pipeline is used. Built-in names: ``default``, ``per-claim``.
    """
    if pipeline is None:
        if pipeline_name == "default":
            async def _default(ex: EvalExample) -> ReliabilityReport:
                return await default_pipeline(ex, skip_cross_reference=skip_cross_reference)
            pipeline = _default
        elif pipeline_name == "per-claim":
            pipeline = per_claim_pipeline
        else:
            raise ValueError(
                f"unknown pipeline_name {pipeline_name!r} (expected 'default' or 'per-claim')"
            )

    sem = asyncio.Semaphore(max(1, concurrency))
    tasks = [asyncio.create_task(_run_one(ex, pipeline, sem)) for ex in examples]

    results: list[EvalPrediction] = [None] * len(tasks)  # type: ignore[list-item]
    completed = 0
    for coro in asyncio.as_completed(tasks):
        pred = await coro
        # Find the matching slot by id (preserves input order in output).
        idx = next(i for i, ex in enumerate(examples) if ex.id == pred.id and results[i] is None)
        results[idx] = pred
        completed += 1
        if progress:
            progress(completed, len(tasks), pred)
    return results


def predictions_to_dict(predictions: list[EvalPrediction]) -> list[dict]:
    return [
        {k: v for k, v in asdict(p).items() if k != "report"}
        | ({"report": p.report} if p.report is not None else {})
        for p in predictions
    ]
