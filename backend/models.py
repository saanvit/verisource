"""Pydantic schemas shared between API and analysis layers."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, HttpUrl


# ---------- Requests ----------


class AssessRequest(BaseModel):
    url: HttpUrl | None = Field(
        default=None, description="URL of the source to assess. Either url or text required."
    )
    text: str | None = Field(
        default=None, description="Raw text/article to assess if no URL is supplied."
    )
    claim: str | None = Field(
        default=None,
        description="Optional specific claim to fact-check against external sources.",
    )


# ---------- Sub-results ----------


class DomainReputation(BaseModel):
    domain: str
    score: float = Field(..., ge=0, le=100)
    bias: str
    type: str
    rationale: str


class ContentAnalysis(BaseModel):
    """LLM-driven content reliability signals."""

    score: float = Field(..., ge=0, le=100)
    factuality: float = Field(..., ge=0, le=100)
    objectivity: float = Field(..., ge=0, le=100)
    transparency: float = Field(..., ge=0, le=100)
    sensationalism: float = Field(..., ge=0, le=100)
    main_claims: list[str]
    red_flags: list[str]
    citations_present: bool
    summary: str


class CorroboratingSource(BaseModel):
    title: str
    url: str
    domain: str
    domain_score: float
    snippet: str
    agreement: Literal["supports", "contradicts", "unclear"] = "unclear"


class CrossReferenceResult(BaseModel):
    score: float = Field(..., ge=0, le=100)
    n_sources: int
    n_high_quality: int
    sources: list[CorroboratingSource]
    consensus: Literal["strong-support", "weak-support", "mixed", "contradicts", "no-data"]


# ---------- Final response ----------


class ReliabilityReport(BaseModel):
    overall_score: float = Field(..., ge=0, le=100)
    verdict: Literal[
        "highly-reliable",
        "generally-reliable",
        "mixed",
        "questionable",
        "unreliable",
    ]
    confidence: float = Field(..., ge=0, le=1)
    domain_reputation: DomainReputation
    content_analysis: ContentAnalysis
    cross_reference: CrossReferenceResult
    explanation: str
    weights: dict[str, float]
