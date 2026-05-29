"""FastAPI entrypoint for the Source Reliability Assessor."""

from __future__ import annotations

from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.config import settings
from backend.models import AssessRequest, ReliabilityReport
from backend.reliability.analyzer import analyze_content
from backend.reliability.content_extractor import (
    ExtractedContent,
    extract_from_text,
    extract_from_url,
)
from backend.reliability.cross_reference import cross_reference
from backend.reliability.domain_reputation import lookup as lookup_domain
from backend.reliability.pipeline_per_claim import (
    assess_per_claim,
    assess_per_claim_reflective,
)
from backend.reliability.scorer import aggregate

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"

app = FastAPI(
    title="Source Reliability Assessor",
    version="0.1.0",
    description=(
        "Assess the reliability of a news article or source by combining a curated domain "
        "reputation prior, an LLM-driven content analysis, and cross-referencing against "
        "independent sources."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health() -> dict[str, object]:
    return {
        "status": "ok",
        "llm_configured": settings.has_llm,
        "search_configured": settings.has_search,
        "model": settings.mistral_model if settings.has_llm else None,
        "default_mode": "per-claim",
        "modes": ["default", "per-claim", "per-claim-reflective"],
    }


@app.post("/api/assess", response_model=ReliabilityReport)
async def assess(
    req: AssessRequest,
    mode: str = Query(
        default="per-claim",
        pattern="^(default|per-claim|per-claim-reflective)$",
        description=(
            "Which pipeline to run. 'per-claim' (default) decomposes the article into "
            "atomic claims and verifies each independently; 'default' runs the legacy "
            "single-shot analyzer + one cross-reference search; 'per-claim-reflective' "
            "wraps the per-claim pipeline in a Claude-driven self-critique loop that "
            "audits the initial verifications and applies targeted fixes (relabel "
            "mislabeled hits, re-search weak queries, split compound claims)."
        ),
    ),
) -> ReliabilityReport:
    if not req.url and not req.text:
        raise HTTPException(
            status_code=400, detail="Provide either `url` or `text` in the request body."
        )

    if req.url:
        try:
            content: ExtractedContent = await extract_from_url(str(req.url))
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status in (401, 403, 451):
                hint = (
                    f"Site returned {status} (likely a paywall or anti-bot block). "
                    "Try the 'Paste text' tab and paste the article body directly."
                )
            elif status == 404:
                hint = "URL returned 404 — check the link and try again."
            else:
                hint = f"Site returned HTTP {status}. Try the 'Paste text' tab instead."
            raise HTTPException(status_code=400, detail=hint) from exc
        except httpx.HTTPError as exc:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Could not fetch URL ({type(exc).__name__}). "
                    "Try the 'Paste text' tab and paste the article body directly."
                ),
            ) from exc
    else:
        content = extract_from_text(req.text or "")

    if mode == "per-claim-reflective":
        return await assess_per_claim_reflective(content, user_claim=req.claim)
    if mode == "per-claim":
        return await assess_per_claim(content, user_claim=req.claim)

    domain = lookup_domain(str(req.url) if req.url else None)
    analysis = await analyze_content(content)
    xref = await cross_reference(content, analysis, req.claim)
    return aggregate(domain, analysis, xref, has_text=bool(content.text.strip()))


if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


def run() -> None:
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )


if __name__ == "__main__":
    run()
