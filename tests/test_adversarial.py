"""Tests for adversarial retrieval + robustness scoring.

The adversarial-query LLM call, the Tavily search, and the stance
labeler are monkeypatched so the tests run offline and assert the
two-search orchestration + robustness logic.
"""

import asyncio

from backend.reliability import claims as claims_mod
from backend.search.web_search import SearchHit


# ---------- _robustness pure-function tests ----------


def test_robustness_untested_when_not_run():
    score, tag = claims_mod._robustness(["supports", "supports"], adversarial_ran=False)
    assert tag == "untested"
    assert score == 50.0


def test_robustness_survived_when_no_contradictions():
    score, tag = claims_mod._robustness(
        ["supports", "unclear", "supports"], adversarial_ran=True
    )
    assert tag == "survived"
    assert score >= 75


def test_robustness_refuted_when_contradictions_dominate():
    score, tag = claims_mod._robustness(
        ["contradicts", "contradicts", "unclear"], adversarial_ran=True
    )
    assert tag == "refuted"
    assert score <= 30


def test_robustness_weakened_when_minority_contradicts():
    score, tag = claims_mod._robustness(
        ["supports", "supports", "supports", "contradicts"], adversarial_ran=True
    )
    assert tag == "weakened"


def test_robustness_empty_labels_is_untested():
    score, tag = claims_mod._robustness([], adversarial_ran=True)
    assert tag == "untested"


# ---------- dedup ----------


def test_dedup_hits_keeps_first_occurrence():
    hits = [
        SearchHit(title="normal", url="https://a.com/x", snippet="n"),
        SearchHit(title="adversarial-dupe", url="https://a.com/x", snippet="a"),
        SearchHit(title="other", url="https://b.com/y", snippet="o"),
    ]
    out = claims_mod._dedup_hits(hits)
    assert len(out) == 2
    assert out[0].title == "normal"  # first (normal-search) copy wins


# ---------- verify_claim with adversarial ----------


def test_verify_claim_adversarial_runs_two_searches(monkeypatch):
    calls = {"search": 0}

    async def fake_web_search(query, k=5, exclude_domain=None):
        calls["search"] += 1
        if "debunked" in query or "false" in query:
            # adversarial search → returns a contradicting-looking hit
            return [SearchHit(title="adv", url="https://adv.com/1", snippet="...")]
        return [SearchHit(title="normal", url="https://norm.com/1", snippet="...")]

    def fake_adv_query(claim):
        return '{"query": "claim debunked false no evidence"}'

    async def fake_label_hits(claim, hits):
        # Label the normal hit supports, the adversarial hit contradicts.
        return ["supports" if "norm" in h.url else "contradicts" for h in hits]

    from types import SimpleNamespace

    monkeypatch.setattr(claims_mod, "web_search", fake_web_search)
    monkeypatch.setattr(claims_mod, "_call_adversarial_query", fake_adv_query)
    monkeypatch.setattr(claims_mod, "_label_hits", fake_label_hits)
    # The adversarial path is gated on settings.has_llm + settings.has_search.
    # settings is a frozen dataclass, so swap the whole object for a stub.
    monkeypatch.setattr(claims_mod, "settings", SimpleNamespace(has_llm=True, has_search=True))

    v = asyncio.run(claims_mod.verify_claim("Some claim", adversarial=True))
    assert calls["search"] == 2  # normal + adversarial
    assert v.adversarial_query == "claim debunked false no evidence"
    assert v.robustness is not None
    assert v.robustness_tag in ("survived", "weakened", "refuted")
    # One supports + one contradicts → contradictions are a plurality → refuted
    assert v.robustness_tag == "refuted"
    assert v.n_evidence == 2  # both hits retained (distinct URLs)


def test_verify_claim_non_adversarial_leaves_robustness_none(monkeypatch):
    async def fake_web_search(query, k=5, exclude_domain=None):
        return [SearchHit(title="n", url="https://n.com/1", snippet="...")]

    async def fake_label_hits(claim, hits):
        return ["supports"]

    monkeypatch.setattr(claims_mod, "web_search", fake_web_search)
    monkeypatch.setattr(claims_mod, "_label_hits", fake_label_hits)

    v = asyncio.run(claims_mod.verify_claim("Some claim", adversarial=False))
    assert v.robustness is None
    assert v.robustness_tag is None
    assert v.adversarial_query is None


def test_verify_all_claims_forwards_adversarial(monkeypatch):
    seen = {"adversarial": None}

    async def fake_verify_claim(claim, *, exclude_domain=None, adversarial=False):
        seen["adversarial"] = adversarial
        from backend.reliability.claims import _build_verification
        return _build_verification(claim, [], [])

    monkeypatch.setattr(claims_mod, "verify_claim", fake_verify_claim)
    asyncio.run(claims_mod.verify_all_claims(["c1"], adversarial=True))
    assert seen["adversarial"] is True
