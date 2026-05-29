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
    """Short, user-facing summary of the verdict.

    Designed to *complement* the breakdown cards, not duplicate them. The
    cards already show every numeric score; this text surfaces the
    decision-relevant nuance — the strongest positive signal and the
    biggest caveat — in 1-3 plain sentences. Don't add a "Red flags:"
    dump here; red flags get their own section in the UI.
    """
    headline = f"This source is {verdict.replace('-', ' ')} ({overall}/100)."

    # Identify the strongest single positive signal, in priority order:
    # domain reputation > corroboration > content quality.
    strongest: str | None = None
    if domain.type not in ("unknown", "satire", "conspiracy", "state") and domain.score >= 80:
        strongest = (
            f"the publisher ({domain.domain}) is a well-established {domain.type} "
            f"with a strong reliability prior"
        )
    elif xref.consensus == "strong-support":
        strongest = (
            f"{xref.n_high_quality} high-quality independent sources strongly corroborate "
            f"the article's claims"
        )
    elif xref.consensus == "weak-support" and xref.n_high_quality >= 2:
        strongest = (
            f"{xref.n_high_quality} high-quality independent sources partially corroborate "
            f"the article's claims"
        )
    elif content.factuality >= 80:
        strongest = "the article's factual content scores high on an LLM-driven content analysis"

    # Identify the biggest single negative signal/caveat, in priority order:
    # special domain types > LLM-flagged content issues > unknown source.
    caveat: str | None = None
    if domain.type == "satire":
        caveat = "the publisher is a known satire/parody outlet — content is not factual"
    elif domain.type == "conspiracy":
        caveat = "the publisher has a history of false claims and is widely classified as a conspiracy outlet"
    elif domain.type == "state":
        caveat = "the publisher is state-controlled and coverage may reflect government messaging"
    elif xref.consensus == "contradicts":
        caveat = "independent sources contradict key claims in the article"
    elif content.objectivity < 50 or content.sensationalism < 50:
        caveat = "the article's tone is opinionated or sensationalist (see Content analysis)"
    elif content.red_flags:
        n = len(content.red_flags)
        caveat = (
            f"the content analyzer flagged {n} issue{'s' if n != 1 else ''} — "
            "see the Red flags section below"
        )
    elif domain.type == "unknown" and not xref.n_sources:
        caveat = "the source is not in our reputation database and no independent corroboration was retrieved"
    elif domain.type == "unknown":
        caveat = "the source is not in our reputation database — judgment relies on content + corroboration alone"

    sentences = [headline]
    if strongest:
        sentences.append(f"Strongest signal: {strongest}.")
    if caveat:
        sentences.append(f"Caveat: {caveat}.")
    return " ".join(sentences)


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
