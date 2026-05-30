"""Tests for the opinion-detection + fact-grounding track.

The opinion-extraction LLM call and the premise verification are both
monkeypatched so the tests run offline and deterministically.
"""

import asyncio

from backend.models import ClaimEvidence, ClaimVerification, OpinionGrounding
from backend.reliability import opinions as op
from backend.reliability.content_extractor import ExtractedContent


def _v(claim: str, score: float, status: str) -> ClaimVerification:
    return ClaimVerification(
        claim=claim,
        score=score,
        status=status,  # type: ignore[arg-type]
        support_ratio=0.6 if status == "supported" else 0.0,
        contradict_ratio=0.6 if status == "contradicted" else 0.0,
        n_evidence=2,
        n_high_quality=1,
        evidence=[
            ClaimEvidence(
                title="t", url=f"https://ex.com/{claim[:6]}", domain="ex.com",
                domain_score=80, snippet="...", agreement="supports",
            )
        ],
    )


def _content() -> ExtractedContent:
    return ExtractedContent(url=None, title="t", text="body text", n_links=0, has_citations=False)


# ---------- _grounding_verdict pure-function tests ----------


def test_grounding_verdict_contradicted_short_circuits():
    vers = [_v("a", 90, "supported"), _v("b", 10, "contradicted")]
    verdict, score = op._grounding_verdict(vers)
    assert verdict == "contradicted"  # any contradicted premise wins


def test_grounding_verdict_well_grounded():
    vers = [_v("a", 85, "supported"), _v("b", 80, "supported")]
    verdict, score = op._grounding_verdict(vers)
    assert verdict == "well-grounded"
    assert score >= 70


def test_grounding_verdict_weakly_grounded():
    vers = [_v("a", 60, "mixed"), _v("b", 55, "mixed")]
    verdict, score = op._grounding_verdict(vers)
    assert verdict == "weakly-grounded"
    assert 45 <= score < 70


def test_grounding_verdict_unsupported_all_unverified():
    vers = [_v("a", 50, "unverified"), _v("b", 50, "unverified")]
    verdict, score = op._grounding_verdict(vers)
    assert verdict == "unsupported"


def test_grounding_verdict_no_premises_is_unsupported():
    verdict, score = op._grounding_verdict([])
    assert verdict == "unsupported"
    assert score == 50.0


# ---------- end-to-end ground_opinions with mocks ----------


def test_ground_opinions_maps_premises_back_to_owners(monkeypatch):
    # Opinion 1 has two premises, opinion 2 has one. Verify they're routed
    # back to the right owner after the flattened batch verification.
    async def fake_extract(content):
        return [
            {"opinion": "Op one", "stance_summary": "s1",
             "premises": ["p1a", "p1b"]},
            {"opinion": "Op two", "stance_summary": "s2",
             "premises": ["p2a"]},
        ]

    async def fake_verify_all(premises, *, exclude_domain=None, concurrency=4, adversarial=False):
        # Return a verification per premise, in order.
        score_map = {"p1a": 90, "p1b": 80, "p2a": 30}
        status_map = {"p1a": "supported", "p1b": "supported", "p2a": "unverified"}
        return [_v(p, score_map[p], status_map[p]) for p in premises]

    monkeypatch.setattr(op, "_extract_opinions", fake_extract)
    monkeypatch.setattr(op, "verify_all_claims", fake_verify_all)

    groundings = asyncio.run(op.ground_opinions(_content(), exclude_domain=None))
    assert len(groundings) == 2
    # Opinion 1: two supported premises (mean 85) → well-grounded
    assert groundings[0].opinion == "Op one"
    assert len(groundings[0].premises) == 2
    assert groundings[0].verdict == "well-grounded"
    # Opinion 2: one unverified premise → unsupported
    assert groundings[1].opinion == "Op two"
    assert len(groundings[1].premises) == 1
    assert groundings[1].verdict == "unsupported"


def test_ground_opinions_empty_when_no_opinions(monkeypatch):
    async def fake_extract(content):
        return []

    monkeypatch.setattr(op, "_extract_opinions", fake_extract)
    groundings = asyncio.run(op.ground_opinions(_content(), exclude_domain=None))
    assert groundings == []


def test_grounding_is_json_serializable(monkeypatch):
    async def fake_extract(content):
        return [{"opinion": "Op", "stance_summary": "s", "premises": ["p"]}]

    async def fake_verify_all(premises, *, exclude_domain=None, concurrency=4, adversarial=False):
        return [_v(p, 88, "supported") for p in premises]

    monkeypatch.setattr(op, "_extract_opinions", fake_extract)
    monkeypatch.setattr(op, "verify_all_claims", fake_verify_all)

    groundings = asyncio.run(op.ground_opinions(_content(), exclude_domain=None))
    # Must round-trip through pydantic dump (it goes through FastAPI).
    dumped = [g.model_dump() for g in groundings]
    assert dumped[0]["verdict"] == "well-grounded"
    assert dumped[0]["premises"][0]["verification"]["claim"] == "p"


# ---------- red-flag generation ----------


def test_opinion_red_flags_for_unsupported_and_contradicted():
    groundings = [
        OpinionGrounding(opinion="Bad take A", stance_summary="s", premises=[],
                         grounding_score=20, verdict="unsupported"),
        OpinionGrounding(opinion="Bad take B", stance_summary="s", premises=[],
                         grounding_score=10, verdict="contradicted"),
        OpinionGrounding(opinion="Fine take", stance_summary="s", premises=[],
                         grounding_score=80, verdict="well-grounded"),
    ]
    flags = op.opinion_red_flags(groundings)
    assert len(flags) == 2  # only the unsupported + contradicted ones
    assert any("contradict" in f.lower() for f in flags)
    assert any("not supported" in f.lower() for f in flags)
