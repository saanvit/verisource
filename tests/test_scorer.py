from backend.models import (
    ContentAnalysis,
    CorroboratingSource,
    CrossReferenceResult,
    DomainReputation,
)
from backend.reliability.scorer import aggregate


def _content(score: float = 70.0) -> ContentAnalysis:
    return ContentAnalysis(
        score=score,
        factuality=score,
        objectivity=score,
        transparency=score,
        sensationalism=score,
        main_claims=["Claim A"],
        red_flags=[],
        citations_present=True,
        summary="ok",
    )


def _xref(score: float = 70.0, n: int = 3) -> CrossReferenceResult:
    return CrossReferenceResult(
        score=score,
        n_sources=n,
        n_high_quality=n,
        sources=[
            CorroboratingSource(
                title="t",
                url="https://reuters.com/x",
                domain="reuters.com",
                domain_score=92,
                snippet="...",
                agreement="supports",
            )
            for _ in range(n)
        ],
        consensus="strong-support",
    )


def _domain(score: float = 90.0, type_: str = "wire") -> DomainReputation:
    return DomainReputation(
        domain="reuters.com",
        score=score,
        bias="center",
        type=type_,
        rationale="curated",
    )


def test_high_signal_yields_reliable_verdict():
    report = aggregate(_domain(), _content(85), _xref(85), has_text=True)
    assert report.overall_score >= 80
    assert report.verdict in ("highly-reliable", "generally-reliable")


def test_unknown_domain_redistributes_weight():
    unknown = DomainReputation(
        domain="unknown", score=50, bias="unknown", type="unknown", rationale="r"
    )
    report = aggregate(unknown, _content(85), _xref(85), has_text=True)
    assert report.weights["domain"] < 0.35


def test_no_xref_zeros_that_weight():
    no_xref = CrossReferenceResult(
        score=50.0, n_sources=0, n_high_quality=0, sources=[], consensus="no-data"
    )
    report = aggregate(_domain(), _content(80), no_xref, has_text=True)
    assert report.weights["cross_reference"] == 0.0
    assert sum(report.weights.values()) == 1.0


def test_low_signal_yields_unreliable():
    low_domain = DomainReputation(
        domain="infowars.com", score=8, bias="right", type="conspiracy", rationale="r"
    )
    report = aggregate(low_domain, _content(20), _xref(20), has_text=True)
    assert report.overall_score < 40
    assert report.verdict in ("questionable", "unreliable")


def test_satire_is_capped_and_flagged():
    satire = DomainReputation(
        domain="theonion.com", score=5, bias="mixed", type="satire", rationale="r"
    )
    # Even with a high content score, satire must be capped near 0.
    report = aggregate(satire, _content(95), _xref(95), has_text=True)
    assert report.overall_score <= 10
    assert report.verdict == "unreliable"
    assert any("satire" in f.lower() for f in report.content_analysis.red_flags)


def test_conspiracy_is_capped():
    conspiracy = DomainReputation(
        domain="infowars.com", score=8, bias="right", type="conspiracy", rationale="r"
    )
    report = aggregate(conspiracy, _content(90), _xref(90), has_text=True)
    assert report.overall_score <= 25
