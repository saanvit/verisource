"""Per-claim verification pipeline.

Drop-in replacement for the default pipeline that produces the same
``ReliabilityReport`` schema but derives ``factuality`` and cross-reference
from a per-claim decomposition and per-claim retrieval rather than a single
whole-article judgment + single-shot corroboration search.

The orchestration:

    1. ``analyze_content`` (legacy, 1 LLM call) → objectivity, transparency,
       sensationalism, citations_present, summary, red_flags, *initial*
       main_claims and factuality (overridden later).
    2. ``decompose_claims`` (1 LLM call) → atomic claims. Falls back to the
       legacy ``main_claims`` if decomposition produces nothing.
    3. ``verify_all_claims`` (N searches + N LLM calls, concurrent) →
       evidence + labels per claim.
    4. Aggregate the per-claim signals into:
       - factuality   = mean(claim_score) × coverage (coverage smoothed)
       - xref score   = mean(claim_score over verified claims)
       - red flags    = legacy red flags + contradicted-claim warnings
    5. ``scorer.aggregate`` (unchanged) → final report.

If the LLM is not configured at all, this pipeline degrades to the legacy
analyzer's lexical heuristic.

Cost / latency (approximate per article with N claims):
    LLM calls: 2 + N           (was 2)
    Tavily calls: N            (was 1)
    Latency: dominated by N concurrent claim verifications.
"""

from __future__ import annotations

from backend.config import settings
from backend.models import (
    ContentAnalysis,
    CorroboratingSource,
    CrossReferenceResult,
    ReliabilityReport,
)
from backend.reliability.analyzer import analyze_content
from backend.reliability.claims import (
    decompose_claims,
    domain_of,
    verify_all_claims,
)
from backend.reliability.content_extractor import ExtractedContent
from backend.reliability.domain_reputation import lookup as lookup_domain
from backend.reliability.scorer import aggregate


def _aggregate_claims(
    verifications: list,  # list[ClaimVerification]
) -> tuple[float, CrossReferenceResult, float, list[str]]:
    """Roll per-claim signals up to article-level factuality + xref.

    Returns: (factuality, xref_result, coverage, extra_red_flags).
    """
    if not verifications:
        # No claims to verify — caller should fall back to legacy fields.
        return (
            50.0,
            CrossReferenceResult(
                score=50.0, n_sources=0, n_high_quality=0, sources=[], consensus="no-data"
            ),
            0.0,
            [],
        )

    n_total = len(verifications)
    verified = [v for v in verifications if v.n_evidence > 0]
    coverage = len(verified) / n_total

    if not verified:
        # Decomposition succeeded but retrieval found nothing for any claim.
        # Factuality is unknown — return neutral, with low coverage.
        return (
            50.0,
            CrossReferenceResult(
                score=50.0, n_sources=0, n_high_quality=0, sources=[], consensus="no-data"
            ),
            0.0,
            [],
        )

    # Article-level factuality: mean claim score over *verified* claims,
    # penalized by missing coverage. With coverage=1 it's just the mean;
    # with coverage→0 it pulls toward the prior of 50 (unknown).
    mean_verified = sum(v.score for v in verified) / len(verified)
    factuality = round(coverage * mean_verified + (1 - coverage) * 50.0, 1)

    # Article-level xref: mean of per-claim scores over verified claims (no
    # coverage penalty here — the xref signal speaks only for the claims it
    # could actually check; the scorer down-weights it via has_xref signals).
    xref_score = round(mean_verified, 1)

    # Build a flat CrossReferenceResult from the union of per-claim evidence.
    sources: list[CorroboratingSource] = []
    n_high_quality = 0
    seen_urls: set[str] = set()
    n_support = 0
    n_contradict = 0
    for v in verifications:
        for e in v.evidence:
            if e.url in seen_urls:
                continue
            seen_urls.add(e.url)
            if e.domain_score >= 75:
                n_high_quality += 1
            if e.agreement == "supports":
                n_support += 1
            elif e.agreement == "contradicts":
                n_contradict += 1
            sources.append(
                CorroboratingSource(
                    title=e.title,
                    url=e.url,
                    domain=e.domain,
                    domain_score=e.domain_score,
                    snippet=e.snippet,
                    agreement=e.agreement,
                )
            )

    # Consensus: derive from per-claim status counts rather than per-hit
    # counts (more meaningful at the article level — each claim is the
    # natural unit, not each hit).
    n_supported = sum(1 for v in verified if v.status == "supported")
    n_contradicted = sum(1 for v in verified if v.status == "contradicted")
    n_mixed = sum(1 for v in verified if v.status == "mixed")
    n_verified = len(verified)
    support_frac = n_supported / n_verified if n_verified else 0.0
    contra_frac = n_contradicted / n_verified if n_verified else 0.0

    if support_frac >= 0.7 and contra_frac < 0.1:
        consensus = "strong-support"
    elif support_frac >= 0.4 and contra_frac < support_frac:
        consensus = "weak-support"
    elif contra_frac >= 0.4:
        consensus = "contradicts"
    elif support_frac > 0 or contra_frac > 0 or n_mixed > 0:
        consensus = "mixed"
    else:
        consensus = "no-data"

    xref = CrossReferenceResult(
        score=xref_score,
        n_sources=len(sources),
        n_high_quality=n_high_quality,
        sources=sources,
        consensus=consensus,  # type: ignore[arg-type]
    )

    extra_flags: list[str] = []
    contradicted_claims = [v.claim for v in verified if v.status == "contradicted"]
    if contradicted_claims:
        extra_flags.append(
            "Independent sources contradict the following claim(s): "
            + "; ".join(c[:140] for c in contradicted_claims[:3])
            + "."
        )
    if coverage <= 0.5 and n_total >= 2:
        extra_flags.append(
            f"Only {len(verified)}/{n_total} decomposed claims could be independently checked — "
            "the article makes assertions that aren't easily corroborated."
        )

    return factuality, xref, coverage, extra_flags


async def assess_per_claim(
    content: ExtractedContent,
    *,
    user_claim: str | None = None,
) -> ReliabilityReport:
    """Run the per-claim pipeline and return a ReliabilityReport.

    Signature is content-first (rather than request-first) so it can be
    composed with both the API route and the eval runner.
    """
    domain = lookup_domain(content.url)

    # Step 1: legacy article-level analysis for objectivity/transparency/sensationalism.
    analysis: ContentAnalysis = await analyze_content(content)

    # Step 2: decompose claims via LLM. Fall back to legacy main_claims if
    # decomposition fails (no LLM, or it returned nothing usable).
    decomposed = await decompose_claims(content)
    if not decomposed:
        decomposed = list(analysis.main_claims)
    if user_claim and user_claim.strip():
        # Make sure the user-specified claim is checked.
        if user_claim.strip() not in decomposed:
            decomposed.insert(0, user_claim.strip())

    # Step 3: verify each claim independently (Tavily + LLM NLI).
    exclude = domain_of(content.url)
    verifications = await verify_all_claims(decomposed, exclude_domain=exclude)

    # Step 4: roll up per-claim signals to article-level factuality + xref.
    factuality, xref, coverage, extra_flags = _aggregate_claims(verifications)

    # Recompute the content score with the new factuality but reusing the
    # legacy weights so downstream scorer behavior is unchanged.
    new_content_score = round(
        0.40 * factuality
        + 0.25 * analysis.objectivity
        + 0.20 * analysis.transparency
        + 0.15 * analysis.sensationalism,
        1,
    )
    merged_flags = list(dict.fromkeys(list(analysis.red_flags) + extra_flags))

    enriched = analysis.model_copy(
        update={
            "score": new_content_score,
            "factuality": factuality,
            "main_claims": decomposed or analysis.main_claims,
            "red_flags": merged_flags,
            "claim_verifications": verifications,
            "coverage": round(coverage, 3),
        }
    )

    return aggregate(domain, enriched, xref, has_text=bool(content.text.strip()))


def has_llm() -> bool:
    """Convenience — the per-claim pipeline degrades to the legacy heuristic
    without an LLM key, but downstream code may want to gate eagerly."""
    return settings.has_llm
