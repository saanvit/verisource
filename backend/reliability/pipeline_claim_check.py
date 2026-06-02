"""Standalone claim-check pipeline.

Unlike the article pipelines, this one takes a *single user claim* (a
tweet, a headline, a viral assertion) with no source document, retrieves
independent evidence for it, labels stance, and stress-tests it with an
adversarial falsification search. The result is rendered into the shared
``ReliabilityReport`` schema — the verified claim is the centerpiece
(``content_analysis.claim_verifications``), ``overall_score`` is the claim
score, and the retrieved hits populate ``cross_reference`` so the existing
evidence UI and eval harness work unchanged.

This is the "Check a Claim" product mode. It also doubles as an
end-to-end claim-verification eval target (retrieval + stance), which the
article pipelines never exposed because they always start from a document.
"""

from __future__ import annotations

from backend.models import (
    ContentAnalysis,
    CorroboratingSource,
    CrossReferenceResult,
    DomainReputation,
    ReliabilityReport,
)
from backend.reliability.claims import verify_claim
from backend.reliability.pipeline_per_claim import _adversarial_enabled
from backend.reliability.scorer import _verdict

_NEUTRAL_DOMAIN = DomainReputation(
    domain="(standalone claim)",
    score=50.0,
    bias="n/a",
    type="unknown",
    rationale="Standalone claim check — no source document or publisher to score.",
)


def _consensus_for(status: str) -> str:
    return {
        "supported": "strong-support",
        "contradicted": "contradicts",
        "mixed": "mixed",
        "unverified": "no-data",
    }.get(status, "no-data")


def _confidence_for(n_evidence: int, n_high_quality: int) -> float:
    score = 0.3
    if n_evidence >= 3:
        score += 0.2
    score += min(0.4, n_high_quality * 0.15)
    return round(min(1.0, score), 2)


def _explain(claim: str, v) -> str:
    status_phrase = {
        "supported": "is supported by independent reliable sources",
        "contradicted": "is contradicted by independent sources",
        "mixed": "draws mixed support from independent sources",
        "unverified": "could not be corroborated — no usable independent evidence was retrieved",
    }.get(v.status, "could not be verified")
    head = f'This claim {status_phrase} ({v.score}/100).'
    if v.robustness_tag == "refuted":
        head += " An adversarial search for disconfirming evidence surfaced credible contradictions."
    elif v.robustness_tag == "survived":
        head += " It also survived an adversarial search for disconfirming evidence."
    return head


async def assess_claim_check(
    claim: str,
    *,
    adversarial: bool | None = None,
) -> ReliabilityReport:
    """Verify a single standalone claim and wrap it as a ReliabilityReport.

    ``adversarial`` defaults to the per-claim mode's setting
    (ADVERSARIAL_RETRIEVAL env var); pass ``False`` explicitly for the
    ablation baseline.
    """
    claim = (claim or "").strip()
    if not claim:
        raise ValueError("claim-check requires a non-empty claim")

    use_adv = _adversarial_enabled() if adversarial is None else adversarial
    v = await verify_claim(claim, adversarial=use_adv)

    sources = [
        CorroboratingSource(
            title=e.title,
            url=e.url,
            domain=e.domain,
            domain_score=e.domain_score,
            snippet=e.snippet,
            agreement=e.agreement,
        )
        for e in v.evidence
    ]
    xref = CrossReferenceResult(
        score=v.score,
        n_sources=v.n_evidence,
        n_high_quality=v.n_high_quality,
        sources=sources,
        consensus=_consensus_for(v.status),  # type: ignore[arg-type]
    )

    red_flags: list[str] = []
    if v.status == "contradicted":
        red_flags.append("Independent sources contradict this claim.")
    if v.robustness_tag == "refuted":
        red_flags.append(
            "Adversarial search surfaced credible evidence that this claim is false or misleading."
        )

    content = ContentAnalysis(
        score=v.score,
        factuality=v.score,
        objectivity=50.0,
        transparency=50.0,
        sensationalism=50.0,
        main_claims=[claim],
        red_flags=red_flags,
        citations_present=v.n_evidence > 0,
        summary=_explain(claim, v),
        claim_verifications=[v],
        coverage=1.0 if v.n_evidence > 0 else 0.0,
    )

    overall = round(v.score, 1)
    verdict = _verdict(overall)
    return ReliabilityReport(
        overall_score=overall,
        verdict=verdict,  # type: ignore[arg-type]
        confidence=_confidence_for(v.n_evidence, v.n_high_quality),
        domain_reputation=_NEUTRAL_DOMAIN,
        content_analysis=content,
        cross_reference=xref,
        explanation=_explain(claim, v),
        weights={"claim": 1.0},
    )
