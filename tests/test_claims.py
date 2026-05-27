"""Tests for the per-claim verification pipeline.

We test the aggregation math against hand-constructed ``ClaimVerification``
inputs and exercise the end-to-end pipeline with the LLM + search functions
monkeypatched, so the tests run offline and deterministically.
"""

import asyncio

from backend.models import ClaimEvidence, ClaimVerification, ContentAnalysis
from backend.reliability import claims as claims_mod
from backend.reliability import pipeline_per_claim as pcm
from backend.reliability.content_extractor import ExtractedContent


# ---------- pure-function unit tests ----------


def test_claim_score_neutral_when_no_evidence():
    assert claims_mod._claim_score(0.0, 0.0, 0, 0) == 50.0


def test_claim_score_high_with_strong_support_and_quality():
    # Full reputation-weighted support, three high-quality hits.
    score = claims_mod._claim_score(1.0, 0.0, 3, 3)
    assert score >= 95.0


def test_claim_score_low_with_strong_contradiction():
    score = claims_mod._claim_score(0.0, 1.0, 3, 1)
    assert score < 5.0


def test_claim_status_thresholds():
    assert claims_mod._claim_status(0.0, 0.0, 0) == "unverified"
    assert claims_mod._claim_status(0.8, 0.0, 4) == "supported"
    assert claims_mod._claim_status(0.0, 0.6, 3) == "contradicted"
    assert claims_mod._claim_status(0.3, 0.3, 3) == "mixed"


def _v(claim: str, score: float, status: str, n_evidence: int = 2, high: int = 1) -> ClaimVerification:
    return ClaimVerification(
        claim=claim,
        score=score,
        status=status,  # type: ignore[arg-type]
        support_ratio=0.5 if status == "supported" else 0.0,
        contradict_ratio=0.5 if status == "contradicted" else 0.0,
        n_evidence=n_evidence,
        n_high_quality=high,
        evidence=[
            ClaimEvidence(
                title="t", url=f"https://example.com/{claim[:8]}",
                domain="example.com", domain_score=80.0,
                snippet="...", agreement="supports" if status == "supported" else "unclear",
            )
        ],
    )


def test_aggregate_claims_empty_returns_neutral():
    fact, xref, cov, flags = pcm._aggregate_claims([])
    assert fact == 50.0
    assert xref.n_sources == 0
    assert xref.consensus == "no-data"
    assert cov == 0.0
    assert flags == []


def test_aggregate_claims_all_supported_yields_high_factuality():
    vs = [_v(f"claim {i}", 90.0, "supported") for i in range(4)]
    fact, xref, cov, flags = pcm._aggregate_claims(vs)
    assert fact >= 85.0
    assert cov == 1.0
    assert xref.consensus == "strong-support"


def test_aggregate_claims_contradicted_lowers_factuality_and_adds_red_flag():
    vs = [
        _v("c1", 8.0, "contradicted"),
        _v("c2", 8.0, "contradicted"),
        _v("c3", 60.0, "supported"),
    ]
    fact, xref, cov, flags = pcm._aggregate_claims(vs)
    assert fact < 50.0
    assert any("contradict" in f.lower() for f in flags)
    assert xref.consensus in ("contradicts", "mixed")


def test_aggregate_claims_low_coverage_adds_flag_and_pulls_toward_50():
    # Two claims, only one has evidence — coverage = 0.5.
    verified = _v("c1", 90.0, "supported")
    unverified = ClaimVerification(
        claim="c2", score=50.0, status="unverified",
        support_ratio=0.0, contradict_ratio=0.0,
        n_evidence=0, n_high_quality=0, evidence=[],
    )
    fact, xref, cov, flags = pcm._aggregate_claims([verified, unverified])
    assert cov == 0.5
    # Pulled toward 50 from the 90 verified score.
    assert 60.0 < fact < 85.0
    assert any("decomposed" in f.lower() and "checked" in f.lower() for f in flags)


def test_aggregate_claims_dedupes_evidence_urls():
    same_url = ClaimEvidence(
        title="t", url="https://shared.example/x",
        domain="shared.example", domain_score=80,
        snippet="...", agreement="supports",
    )
    v1 = ClaimVerification(
        claim="c1", score=80, status="supported", support_ratio=1.0,
        contradict_ratio=0.0, n_evidence=1, n_high_quality=1, evidence=[same_url],
    )
    v2 = ClaimVerification(
        claim="c2", score=80, status="supported", support_ratio=1.0,
        contradict_ratio=0.0, n_evidence=1, n_high_quality=1, evidence=[same_url],
    )
    _, xref, _, _ = pcm._aggregate_claims([v1, v2])
    assert xref.n_sources == 1  # deduped


# ---------- claim-cleanup tests ----------


def test_clean_claim_strips_trailing_punct_and_quotes():
    assert claims_mod._clean_claim('  "The vote was 62 to 38."  ') == "The vote was 62 to 38"


def test_clean_claim_caps_length():
    long = "x " * 300
    assert len(claims_mod._clean_claim(long)) <= 280


# ---------- end-to-end pipeline test with mocked LLM + search ----------


def test_pipeline_with_mocked_llm_and_search(monkeypatch):
    """Stub decompose + verify so we can verify orchestration end-to-end."""

    async def fake_decompose(content: ExtractedContent) -> list[str]:
        return ["Claim A is factual.", "Claim B is also factual."]

    async def fake_verify_all_claims(
        claims_list: list[str], *, exclude_domain: str | None = None, concurrency: int = 4
    ) -> list[ClaimVerification]:
        return [_v(c, 88.0, "supported", n_evidence=3, high=2) for c in claims_list]

    async def fake_analyze(content: ExtractedContent) -> ContentAnalysis:
        return ContentAnalysis(
            score=70.0,
            factuality=60.0,
            objectivity=80.0,
            transparency=75.0,
            sensationalism=80.0,
            main_claims=["legacy-claim"],
            red_flags=[],
            citations_present=True,
            summary="ok",
        )

    monkeypatch.setattr(pcm, "decompose_claims", fake_decompose)
    monkeypatch.setattr(pcm, "verify_all_claims", fake_verify_all_claims)
    monkeypatch.setattr(pcm, "analyze_content", fake_analyze)

    content = ExtractedContent(
        url="https://www.reuters.com/x", title="t", text="some article body", n_links=4,
        has_citations=True,
    )
    report = asyncio.run(pcm.assess_per_claim(content))
    # Factuality replaced by per-claim mean (88).
    assert 87.0 <= report.content_analysis.factuality <= 89.0
    assert report.content_analysis.coverage == 1.0
    assert report.content_analysis.claim_verifications is not None
    assert len(report.content_analysis.claim_verifications) == 2
    # Domain is reuters → high prior → overall well above the legacy baseline.
    assert report.overall_score >= 80
    assert report.verdict in ("highly-reliable", "generally-reliable")


def test_pipeline_falls_back_to_legacy_claims_when_decompose_empty(monkeypatch):
    """If LLM decomposition returns nothing, we fall back to legacy main_claims."""

    async def fake_decompose(content: ExtractedContent) -> list[str]:
        return []  # No LLM key, or model returned junk.

    async def fake_verify_all(claims_list, *, exclude_domain=None, concurrency=4):
        assert claims_list == ["legacy-claim-one"]
        return [_v(claims_list[0], 75.0, "supported")]

    async def fake_analyze(content):
        return ContentAnalysis(
            score=70.0, factuality=60.0, objectivity=75.0, transparency=70.0,
            sensationalism=75.0, main_claims=["legacy-claim-one"], red_flags=[],
            citations_present=False, summary="ok",
        )

    monkeypatch.setattr(pcm, "decompose_claims", fake_decompose)
    monkeypatch.setattr(pcm, "verify_all_claims", fake_verify_all)
    monkeypatch.setattr(pcm, "analyze_content", fake_analyze)

    content = ExtractedContent(
        url="https://www.npr.org/x", title="t", text="body", n_links=2, has_citations=False,
    )
    report = asyncio.run(pcm.assess_per_claim(content))
    assert report.content_analysis.main_claims == ["legacy-claim-one"]
