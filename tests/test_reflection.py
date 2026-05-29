"""Tests for the self-reflective verification agent.

The critique LLM and Tavily search are both monkeypatched so the loop
runs deterministically and offline. We test:

* Empty actions → terminates after 1 round (no infinite loop)
* relabel_hit flips an evidence's agreement and recomputes score/status
* research_claim replaces the targeted verification
* split_claim removes the original and appends N atoms
* Early termination when all score deltas are below the 5-point threshold
* Critique LLM failure is caught and recorded as a noop round
* assess_per_claim (non-reflective) regressionally still returns
  reflection_trace=None
"""

import asyncio
import json
from typing import Any

import pytest

from backend.models import (
    ClaimEvidence,
    ClaimVerification,
    ReflectionAction,
)
from backend.reliability import claims as claims_mod
from backend.reliability import pipeline_per_claim as pcm
from backend.reliability import reflection as reflection_mod
from backend.reliability.content_extractor import ExtractedContent
from backend.search.web_search import SearchHit


# ---------- helpers ----------


def _evidence(url: str, agreement: str = "unclear", domain_score: float = 82.0) -> ClaimEvidence:
    return ClaimEvidence(
        title=f"title-{url}",
        url=url,
        domain=url.split("/")[2] if "://" in url else url,
        domain_score=domain_score,
        snippet=f"snippet for {url}",
        agreement=agreement,  # type: ignore[arg-type]
    )


def _verification(
    claim: str,
    evidence: list[ClaimEvidence],
    *,
    score: float = 50.0,
    status: str = "unverified",
) -> ClaimVerification:
    n_high = sum(1 for e in evidence if e.domain_score >= 75)
    return ClaimVerification(
        claim=claim,
        score=score,
        status=status,  # type: ignore[arg-type]
        support_ratio=0.0,
        contradict_ratio=0.0,
        n_evidence=len(evidence),
        n_high_quality=n_high,
        evidence=evidence,
    )


def _content() -> ExtractedContent:
    return ExtractedContent(
        url=None, title="t", text="body text", n_links=0, has_citations=False,
    )


def _make_critique_returning(payload: dict[str, Any]):
    """Build a fake _call_critique_llm that returns canned JSON."""
    def fake(_payload: str) -> str:
        return json.dumps(payload)
    return fake


# ---------- empty actions terminates ----------


def test_empty_actions_terminates_after_one_round(monkeypatch):
    monkeypatch.setattr(
        reflection_mod, "_call_critique_llm",
        _make_critique_returning({
            "critique": "Looks fine.",
            "issues": [],
            "actions": [],
        }),
    )
    verifications = [_verification("c1", [_evidence("https://a.example/1")])]
    result, trace = asyncio.run(
        reflection_mod.run_reflection_loop(
            verifications, content=_content(), exclude_domain=None,
        )
    )
    assert len(trace) == 1
    assert trace[0].actions == []
    assert result == verifications  # untouched


# ---------- relabel_hit ----------


def test_relabel_hit_flips_label_and_recomputes(monkeypatch):
    # Two high-quality unclear hits → status should be "unverified" initially.
    # After relabeling both to "supports", status should become "supported"
    # and the score should jump.
    ev1 = _evidence("https://a.example/1", agreement="unclear", domain_score=82)
    ev2 = _evidence("https://b.example/2", agreement="unclear", domain_score=82)
    initial = [_verification("claim X", [ev1, ev2], score=58.0, status="unverified")]

    # Round 1 returns two relabel actions; round 2 returns empty.
    call_log: list[int] = []

    def fake_critique(_payload: str) -> str:
        call_log.append(1)
        if len(call_log) == 1:
            return json.dumps({
                "critique": "Both hits clearly support — relabel.",
                "issues": ["mislabeled"],
                "actions": [
                    {
                        "type": "relabel_hit",
                        "claim": "claim X",
                        "hit_url": "https://a.example/1",
                        "new_label": "supports",
                        "reason": "snippet affirms",
                    },
                    {
                        "type": "relabel_hit",
                        "claim": "claim X",
                        "hit_url": "https://b.example/2",
                        "new_label": "supports",
                        "reason": "snippet affirms",
                    },
                ],
            })
        return json.dumps({"critique": "Done.", "issues": [], "actions": []})

    monkeypatch.setattr(reflection_mod, "_call_critique_llm", fake_critique)

    result, trace = asyncio.run(
        reflection_mod.run_reflection_loop(
            initial, content=_content(), exclude_domain=None,
        )
    )
    assert len(result) == 1
    updated = result[0]
    # Both evidence should now be "supports"
    assert all(e.agreement == "supports" for e in updated.evidence)
    # Status should have improved
    assert updated.status == "supported"
    # Score should have gone up
    assert updated.score > 58.0
    # Trace should record the round
    assert len(trace) >= 1
    assert len(trace[0].actions) == 2
    assert "claim X" in trace[0].score_deltas
    assert trace[0].score_deltas["claim X"] > 0


def test_relabel_hit_unknown_claim_is_skipped(monkeypatch):
    monkeypatch.setattr(
        reflection_mod, "_call_critique_llm",
        _make_critique_returning({
            "critique": "...",
            "issues": [],
            "actions": [{
                "type": "relabel_hit",
                "claim": "DOES NOT EXIST",
                "hit_url": "https://a.example/1",
                "new_label": "supports",
                "reason": "...",
            }],
        }),
    )
    initial = [_verification("real claim", [_evidence("https://a.example/1")])]
    result, trace = asyncio.run(
        reflection_mod.run_reflection_loop(
            initial, content=_content(), exclude_domain=None,
        )
    )
    # Action was a no-op (unknown claim), so verification is unchanged.
    assert result[0].evidence[0].agreement == "unclear"
    assert trace[0].score_deltas == {}


# ---------- research_claim ----------


def test_research_claim_replaces_verification(monkeypatch):
    initial = [_verification("the claim", [_evidence("https://old.example/x")], score=50.0)]

    async def fake_web_search(query, k=5, exclude_domain=None):
        # Verify the alternative query is what got passed in
        assert query == "specific alt query"
        return [SearchHit(title="new title", url="https://new.example/y", snippet="new snippet")]

    async def fake_label_hits(_claim, _hits):
        return ["supports"]

    monkeypatch.setattr(reflection_mod, "web_search", fake_web_search)
    monkeypatch.setattr(reflection_mod, "_label_hits", fake_label_hits)
    monkeypatch.setattr(
        reflection_mod, "_call_critique_llm",
        _make_critique_returning({
            "critique": "Re-search.",
            "issues": ["weak_query"],
            "actions": [{
                "type": "research_claim",
                "claim": "the claim",
                "alternative_query": "specific alt query",
                "reason": "original too generic",
            }],
        }),
    )

    result, trace = asyncio.run(
        reflection_mod.run_reflection_loop(
            initial, content=_content(), exclude_domain=None,
        )
    )
    assert len(result) == 1
    assert result[0].evidence[0].url == "https://new.example/y"
    assert result[0].evidence[0].agreement == "supports"
    assert len(trace[0].actions) == 1


# ---------- split_claim ----------


def test_split_claim_replaces_with_atoms(monkeypatch):
    initial = [_verification("compound A and B", [_evidence("https://x.example/1")])]

    async def fake_verify_all_claims(subclaims, exclude_domain=None, concurrency=4):
        return [
            _verification(s, [_evidence(f"https://atom.example/{i}", agreement="supports")],
                          score=85.0, status="supported")
            for i, s in enumerate(subclaims)
        ]

    monkeypatch.setattr(reflection_mod, "verify_all_claims", fake_verify_all_claims)
    monkeypatch.setattr(
        reflection_mod, "_call_critique_llm",
        _make_critique_returning({
            "critique": "Compound — split.",
            "issues": ["compound_claim"],
            "actions": [{
                "type": "split_claim",
                "claim": "compound A and B",
                "subclaims": ["atom A", "atom B"],
                "reason": "two assertions",
            }],
        }),
    )

    result, trace = asyncio.run(
        reflection_mod.run_reflection_loop(
            initial, content=_content(), exclude_domain=None,
        )
    )
    claims_after = [v.claim for v in result]
    assert "compound A and B" not in claims_after
    assert "atom A" in claims_after
    assert "atom B" in claims_after
    # The original claim's score-erasure shows up as a negative delta:
    assert trace[0].score_deltas.get("compound A and B", 0.0) < 0
    # The new atoms register positive deltas vs the 50-prior:
    assert trace[0].score_deltas.get("atom A", 0.0) > 0


# ---------- termination on small deltas ----------


def test_loop_terminates_when_deltas_below_threshold(monkeypatch):
    # Round 1's action produces a tiny score delta (< 5 points). Loop
    # should terminate via the "all deltas below threshold" convergence
    # rule without calling critique a second time.
    #
    # Setup: 10 hits, 5 supports + 5 unclear, all low domain_score (no
    # high-quality bonus). Relabeling one unclear → supports moves
    # support_ratio from 0.5 → 0.6, score from 72.5 → 77 (delta 4.5).
    evs = (
        [_evidence(f"https://a.example/{i}", agreement="supports", domain_score=20)
         for i in range(5)] +
        [_evidence(f"https://b.example/{i}", agreement="unclear", domain_score=20)
         for i in range(5)]
    )
    # Use the recompute helper so the initial score reflects the formula
    # exactly (and isn't off-by-rounding from a hand-picked literal).
    seed = ClaimVerification(
        claim="c", score=0, status="unverified",
        support_ratio=0, contradict_ratio=0, n_evidence=10, n_high_quality=0,
        evidence=evs,
    )
    initial = [claims_mod.recompute_verification_after_relabel(seed)]

    call_count = {"n": 0}

    def fake_critique(_payload: str) -> str:
        call_count["n"] += 1
        return json.dumps({
            "critique": "tweak one",
            "issues": [],
            "actions": [{
                "type": "relabel_hit",
                "claim": "c",
                "hit_url": "https://b.example/0",
                "new_label": "supports",
                "reason": "...",
            }],
        })

    monkeypatch.setattr(reflection_mod, "_call_critique_llm", fake_critique)
    result, trace = asyncio.run(
        reflection_mod.run_reflection_loop(
            initial, content=_content(), exclude_domain=None, max_rounds=3,
        )
    )
    # Single small-delta round → loop converges and stops after round 1.
    assert call_count["n"] == 1
    assert len(trace) == 1
    # Sanity check: the actual delta should be < 5 (that's why we stopped).
    delta = trace[0].score_deltas.get("c", 0.0)
    assert 0 < abs(delta) < 5.0, f"unexpected delta magnitude: {delta}"


# ---------- critique failure becomes noop round ----------


def test_critique_failure_records_noop_and_exits(monkeypatch):
    def fake_critique(_payload: str) -> str:
        raise RuntimeError("simulated network failure")
    monkeypatch.setattr(reflection_mod, "_call_critique_llm", fake_critique)

    initial = [_verification("c", [_evidence("https://a.example/1")])]
    result, trace = asyncio.run(
        reflection_mod.run_reflection_loop(
            initial, content=_content(), exclude_domain=None, max_rounds=2,
        )
    )
    assert len(trace) == 1
    assert trace[0].actions == []
    assert "critique_call_failed" in trace[0].issues
    # Verifications untouched
    assert result == initial


# ---------- backward compat: non-reflective pipeline ----------


def test_assess_per_claim_returns_none_reflection_trace(monkeypatch):
    """Regression guard: the standard per-claim pipeline must not start
    populating reflection_trace by accident."""

    async def fake_decompose(content):
        return ["one", "two"]

    async def fake_verify_all(claims_list, *, exclude_domain=None, concurrency=4):
        return [_verification(c, [_evidence(f"https://e.example/{i}")], score=70.0,
                              status="supported") for i, c in enumerate(claims_list)]

    async def fake_analyze(content):
        from backend.models import ContentAnalysis
        return ContentAnalysis(
            score=70, factuality=60, objectivity=75, transparency=70, sensationalism=75,
            main_claims=["legacy"], red_flags=[], citations_present=False, summary="ok",
        )

    monkeypatch.setattr(pcm, "decompose_claims", fake_decompose)
    monkeypatch.setattr(pcm, "verify_all_claims", fake_verify_all)
    monkeypatch.setattr(pcm, "analyze_content", fake_analyze)

    report = asyncio.run(
        pcm.assess_per_claim(
            ExtractedContent(url="https://www.npr.org/x", title="t", text="body",
                             n_links=2, has_citations=False),
        )
    )
    assert report.content_analysis.reflection_trace is None


def test_assess_per_claim_reflective_populates_trace(monkeypatch):
    """Smoke test: the reflective variant attaches a non-None trace."""

    async def fake_decompose(content):
        return ["alpha", "beta"]

    async def fake_verify_all(claims_list, *, exclude_domain=None, concurrency=4):
        return [_verification(c, [_evidence(f"https://e.example/{i}")], score=70.0,
                              status="supported") for i, c in enumerate(claims_list)]

    async def fake_analyze(content):
        from backend.models import ContentAnalysis
        return ContentAnalysis(
            score=70, factuality=60, objectivity=75, transparency=70, sensationalism=75,
            main_claims=["legacy"], red_flags=[], citations_present=False, summary="ok",
        )

    def fake_critique(_payload: str) -> str:
        return json.dumps({"critique": "fine", "issues": [], "actions": []})

    monkeypatch.setattr(pcm, "decompose_claims", fake_decompose)
    monkeypatch.setattr(pcm, "verify_all_claims", fake_verify_all)
    monkeypatch.setattr(pcm, "analyze_content", fake_analyze)
    monkeypatch.setattr(reflection_mod, "_call_critique_llm", fake_critique)

    report = asyncio.run(
        pcm.assess_per_claim_reflective(
            ExtractedContent(url="https://www.npr.org/x", title="t", text="body",
                             n_links=2, has_citations=False),
        )
    )
    assert report.content_analysis.reflection_trace is not None
    assert len(report.content_analysis.reflection_trace) >= 1
