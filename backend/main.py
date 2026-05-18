"""FastAPI entrypoint for the Source Reliability Assessor."""

from __future__ import annotations

from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
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
    }


@app.post("/api/assess", response_model=ReliabilityReport)
async def assess(req: AssessRequest) -> ReliabilityReport:
    if not req.url and not req.text:
        raise HTTPException(
            status_code=400, detail="Provide either `url` or `text` in the request body."
        )

    if req.url:
        try:
            content: ExtractedContent = await extract_from_url(str(req.url))
        except httpx.HTTPError as exc:
            raise HTTPException(
                status_code=400, detail=f"Could not fetch URL: {exc}"
            ) from exc
    else:
        content = extract_from_text(req.text or "")

    domain = lookup_domain(str(req.url) if req.url else None)
    analysis = await analyze_content(content)
    xref = await cross_reference(content, analysis, req.claim)
    report = aggregate(domain, analysis, xref, has_text=bool(content.text.strip()))
    return report


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
