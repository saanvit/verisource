"""Aggregate domain, content, and cross-reference signals into one score."""

from __future__ import annotations

from backend.models import (
    ContentAnalysis,
    CrossReferenceResult,
    DomainReputation,
    ReliabilityReport,
)
from backend.reliability import calibration


def _verdict(score: float) -> str:
    if score >= 85:
        return "highly-reliable"
    if score >= 70:
        return "generally-reliable"
    if score >= 50:
        return "mixed"
    if score >= 30:
        return "questionable"
    return "unreliable"


def _confidence(
    domain: DomainReputation, xref: CrossReferenceResult, has_text: bool
) -> float:
    score = 0.3
    if domain.type != "unknown":
        score += 0.25
    if has_text:
        score += 0.2
    if xref.n_sources >= 3:
        score += 0.15
    if xref.consensus in ("strong-support", "contradicts"):
        score += 0.1
    return round(min(1.0, score), 2)


def _explain(
    overall: float,
    verdict: str,
    domain: DomainReputation,
    content: ContentAnalysis,
    xref: CrossReferenceResult,
) -> str:
    parts = [
        f"Overall reliability is {overall}/100 ({verdict}).",
        f"Domain prior: {domain.domain} scores {domain.score}/100 "
        f"(type={domain.type}, bias={domain.bias}).",
        f"Content analysis: factuality {content.factuality}, objectivity {content.objectivity}, "
        f"transparency {content.transparency}, sensationalism-restraint {content.sensationalism}.",
    ]
    if xref.n_sources:
        parts.append(
            f"Cross-reference: {xref.n_sources} independent sources retrieved "
            f"({xref.n_high_quality} high-quality), consensus={xref.consensus}."
        )
    else:
        parts.append("Cross-reference: no independent sources retrieved.")
    if content.red_flags:
        parts.append("Red flags: " + "; ".join(content.red_flags) + ".")
    return " ".join(parts)


SATIRE_RED_FLAG = (
    "Source is a known satire/parody publication — content is comedic, not factual."
)
CONSPIRACY_RED_FLAG = (
    "Source is widely classified as a conspiracy/hoax outlet with a history of false claims."
)
STATE_MEDIA_RED_FLAG = (
    "Source is a state-controlled outlet; coverage may reflect government messaging."
)


def aggregate(
    domain: DomainReputation,
    content: ContentAnalysis,
    xref: CrossReferenceResult,
    has_text: bool,
) -> ReliabilityReport:
    """Weighted combination of the three signals.

    Weights are dynamic: when we lack strong evidence in one channel (e.g. domain
    is unknown, or no cross-reference data), we redistribute that weight to the
    remaining channels. Special domain types (satire, conspiracy, state media)
    inject red flags and apply a hard cap on the overall score.
    """

    base_weights = {"domain": 0.35, "content": 0.40, "cross_reference": 0.25}
    weights = dict(base_weights)

    if domain.type == "unknown":
        weights["domain"] *= 0.4
    if domain.type in ("satire", "conspiracy", "state"):
        weights["domain"] = max(weights["domain"], 0.6)
    if not has_text:
        weights["content"] *= 0.5
    if xref.n_sources == 0:
        weights["cross_reference"] = 0.0

    total = sum(weights.values()) or 1.0
    weights = {k: v / total for k, v in weights.items()}

    extra_flags: list[str] = []
    if domain.type == "satire":
        extra_flags.append(SATIRE_RED_FLAG)
    elif domain.type == "conspiracy":
        extra_flags.append(CONSPIRACY_RED_FLAG)
    elif domain.type == "state":
        extra_flags.append(STATE_MEDIA_RED_FLAG)

    if extra_flags:
        merged_flags = list(dict.fromkeys(extra_flags + list(content.red_flags)))
        content = content.model_copy(update={"red_flags": merged_flags})

    overall = (
        weights["domain"] * domain.score
        + weights["content"] * content.score
        + weights["cross_reference"] * xref.score
    )

    if domain.type == "satire":
        overall = min(overall, 10.0)
    elif domain.type == "conspiracy":
        overall = min(overall, 25.0)
    elif domain.type == "state":
        overall = min(overall, 35.0)

    overall = round(max(0.0, min(100.0, overall)), 1)

    # Optional post-processing: if a calibrator is configured (via the
    # CALIBRATION_PATH env var), remap raw → calibrated. Domain-type hard
    # caps (satire/conspiracy/state) were already applied above and are
    # preserved because the calibrator is fit on raw post-cap scores.
    cal = calibration.get_runtime()
    if cal is not None:
        overall = round(cal.apply(overall), 1)

    verdict = _verdict(overall)
    return ReliabilityReport(
        overall_score=overall,
        verdict=verdict,  # type: ignore[arg-type]
        confidence=_confidence(domain, xref, has_text),
        domain_reputation=domain,
        content_analysis=content,
        cross_reference=xref,
        explanation=_explain(overall, verdict, domain, content, xref),
        weights={k: round(v, 3) for k, v in weights.items()},
    )
