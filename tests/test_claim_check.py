"""Tests for the standalone claim-check pipeline.

verify_claim is monkeypatched so the wrapper's report construction is
tested offline (no search / no API keys).
"""

from __future__ import annotations

import asyncio

from backend.models import ClaimEvidence, ClaimVerification
from backend.reliability import pipeline_claim_check as pcc


def _verification(status: str, score: float, *, robustness_tag: str | None = None) -> ClaimVerification:
    return ClaimVerification(
        claim="The treaty was signed in 2015.",
        score=score,
        status=status,  # type: ignore[arg-type]
        support_ratio=0.8 if status == "supported" else 0.0,
        contradict_ratio=0.0 if status == "supported" else 0.6,
        n_evidence=3,
        n_high_quality=2,
        evidence=[
            ClaimEvidence(
                title="Source", url="https://reuters.com/x", domain="reuters.com",
                domain_score=92.0, snippet="...", agreement="supports",
            )
        ],
        robustness=85.0 if robustness_tag else None,
        robustness_tag=robustness_tag,  # type: ignore[arg-type]
    )


def test_claim_check_supported_maps_to_report(monkeypatch):
    async def fake_verify(claim, *, exclude_domain=None, adversarial=False):
        assert adversarial is True
        return _verification("supported", 88.0, robustness_tag="survived")

    monkeypatch.setattr(pcc, "verify_claim", fake_verify)
    report = asyncio.run(pcc.assess_claim_check("The treaty was signed in 2015.", adversarial=True))

    assert report.overall_score == 88.0
    assert report.verdict == "highly-reliable"
    ca = report.content_analysis
    assert ca.claim_verifications is not None and len(ca.claim_verifications) == 1
    # Evidence flows into the cross-reference channel for the Evidence tab.
    assert report.cross_reference.n_sources == 3
    assert report.cross_reference.consensus == "strong-support"


def test_claim_check_contradicted_flags_and_lowers_score(monkeypatch):
    async def fake_verify(claim, *, exclude_domain=None, adversarial=False):
        return _verification("contradicted", 18.0, robustness_tag="refuted")

    monkeypatch.setattr(pcc, "verify_claim", fake_verify)
    report = asyncio.run(pcc.assess_claim_check("A false claim.", adversarial=True))

    assert report.overall_score == 18.0
    assert report.verdict == "unreliable"
    # Both a contradiction flag and an adversarial-refuted flag are present.
    assert any("contradict" in f.lower() for f in report.content_analysis.red_flags)
    assert any("adversarial" in f.lower() for f in report.content_analysis.red_flags)


def test_claim_check_blank_claim_raises():
    import pytest

    with pytest.raises(ValueError):
        asyncio.run(pcc.assess_claim_check("   "))
