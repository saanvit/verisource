"""Tests for the NLI backend dispatch in claims._label_hits and the
unclear-margin decision rule in nli._pick_label.

We don't exercise the real transformer here — that requires the heavy ML
stack from requirements-nli.txt — but we do confirm that the dispatch
honors the NLI_BACKEND env var, that the local path is wired up
correctly with a stub, and that the unclear-margin decision rule
behaves the way the live demo trace motivated.
"""

import asyncio

from backend.reliability import claims as claims_mod
from backend.reliability import nli as nli_mod
from backend.search.web_search import SearchHit


# ---------- _pick_label decision-rule tests ----------
# These exercise the pure-function decision logic without loading the
# transformer. Index 0/1/2 = supports/contradicts/unclear in our scheme.

_OURS = {0: "supports", 1: "contradicts", 2: "unclear"}
# Off-the-shelf checkpoints use the standard NLI order — same softmax
# vector should produce the same normalized label.
_OFFTHESHELF = {0: "entailment", 1: "neutral", 2: "contradiction"}


def test_pick_label_clean_supports():
    # Model is confident — plain argmax should win.
    assert nli_mod._pick_label([0.85, 0.05, 0.10], _OURS, 0.15) == "supports"


def test_pick_label_clean_contradicts():
    assert nli_mod._pick_label([0.05, 0.80, 0.15], _OURS, 0.15) == "contradicts"


def test_pick_label_unclear_wins_when_dominant():
    # Unclear leads by > margin → commit to unclear.
    assert nli_mod._pick_label([0.10, 0.10, 0.80], _OURS, 0.15) == "unclear"


def test_pick_label_overrides_marginal_unclear_for_supports():
    # The exact SB17 failure mode: 0.40 unclear / 0.35 supports / 0.25
    # contradicts. Plain argmax would say "unclear" — we want "supports".
    assert nli_mod._pick_label([0.35, 0.25, 0.40], _OURS, 0.15) == "supports"


def test_pick_label_overrides_marginal_unclear_for_contradicts():
    # Symmetric: marginal unclear with contradicts as runner-up.
    assert nli_mod._pick_label([0.20, 0.38, 0.42], _OURS, 0.15) == "contradicts"


def test_pick_label_threshold_boundary():
    # Exactly at the margin → unclear should win (margin is the minimum
    # lead required, not strictly-greater-than).
    # 0.50 unclear, 0.35 supports → lead 0.15 == margin.
    assert nli_mod._pick_label([0.35, 0.15, 0.50], _OURS, 0.15) == "unclear"
    # 0.49 unclear, 0.35 supports → lead 0.14 < margin → commit to supports.
    assert nli_mod._pick_label([0.35, 0.16, 0.49], _OURS, 0.15) == "supports"


def test_pick_label_works_on_offtheshelf_label_order():
    # Off-the-shelf checkpoints use entailment/neutral/contradiction;
    # our decision rule must aggregate by normalized label, not by
    # raw index, so the same softmax vector should produce the same
    # outcome here as it does on our local fine-tune.
    # 0.35 entailment / 0.25 neutral / 0.40 contradiction — argmax says
    # contradiction. Note neutral here is at index 1, NOT index 2.
    assert nli_mod._pick_label([0.35, 0.25, 0.40], _OFFTHESHELF, 0.15) == "contradicts"
    # 0.35 entailment / 0.40 neutral / 0.25 contradiction — argmax says
    # neutral by only 0.05 → commit to supports.
    assert nli_mod._pick_label([0.35, 0.40, 0.25], _OFFTHESHELF, 0.15) == "supports"


def test_pick_label_zero_margin_is_plain_argmax():
    # When margin is 0, the function reduces to plain argmax (any unclear
    # lead is enough).
    assert nli_mod._pick_label([0.34, 0.33, 0.33], _OURS, 0.0) == "supports"


# ---------- existing backend-dispatch helpers + tests ----------


def _hits(n: int = 2) -> list[SearchHit]:
    return [
        SearchHit(title=f"t{i}", url=f"https://ex.com/{i}", snippet=f"s{i}")
        for i in range(n)
    ]


def test_default_backend_is_llm(monkeypatch):
    monkeypatch.delenv("NLI_BACKEND", raising=False)
    assert claims_mod._nli_backend() == "llm"


def test_backend_env_var_is_normalized(monkeypatch):
    monkeypatch.setenv("NLI_BACKEND", " LOCAL ")
    assert claims_mod._nli_backend() == "local"


def test_label_hits_dispatches_to_local_when_configured(monkeypatch):
    """When NLI_BACKEND=local, _label_hits should call the local labeler."""
    monkeypatch.setenv("NLI_BACKEND", "local")

    called = {"llm": 0, "local": 0}

    async def fake_llm(claim, hits):
        called["llm"] += 1
        return ["supports"] * len(hits)

    async def fake_local(claim, hits):
        called["local"] += 1
        return ["contradicts"] * len(hits)

    monkeypatch.setattr(claims_mod, "_label_hits_llm", fake_llm)
    monkeypatch.setattr(claims_mod, "_label_hits_local", fake_local)

    labels = asyncio.run(claims_mod._label_hits("c", _hits(3)))
    assert called["local"] == 1
    assert called["llm"] == 0
    assert labels == ["contradicts", "contradicts", "contradicts"]


def test_label_hits_dispatches_to_llm_by_default(monkeypatch):
    monkeypatch.delenv("NLI_BACKEND", raising=False)

    called = {"llm": 0, "local": 0}

    async def fake_llm(claim, hits):
        called["llm"] += 1
        return ["supports"] * len(hits)

    async def fake_local(claim, hits):
        called["local"] += 1
        return ["contradicts"] * len(hits)

    monkeypatch.setattr(claims_mod, "_label_hits_llm", fake_llm)
    monkeypatch.setattr(claims_mod, "_label_hits_local", fake_local)

    labels = asyncio.run(claims_mod._label_hits("c", _hits(2)))
    assert called["llm"] == 1
    assert called["local"] == 0
    assert labels == ["supports", "supports"]


def test_label_hits_empty_short_circuits(monkeypatch):
    monkeypatch.setenv("NLI_BACKEND", "local")

    async def fake_local(claim, hits):  # should not be called
        raise AssertionError("local labeler called with no hits")

    monkeypatch.setattr(claims_mod, "_label_hits_local", fake_local)
    assert asyncio.run(claims_mod._label_hits("c", [])) == []


def test_local_labeler_handles_import_error_gracefully(monkeypatch):
    """If the heavy ML deps aren't installed (or model load fails for
    any reason), _label_hits_local should not crash the request — it
    falls back to 'unclear' for every hit."""

    def _boom(*_args, **_kwargs):
        # Simulates the production failure modes: torch/transformers
        # unavailable, model checkpoint missing, OOM during load, etc.
        raise ImportError("transformers not installed in this environment")

    monkeypatch.setattr(nli_mod, "label_hits_nli", _boom)
    labels = asyncio.run(claims_mod._label_hits_local("c", _hits(2)))
    assert labels == ["unclear", "unclear"]
