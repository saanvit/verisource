"""Tests for the NLI backend dispatch in claims._label_hits.

We don't exercise the real transformer here — that requires the heavy ML
stack from requirements-nli.txt — but we do confirm that the dispatch
honors the NLI_BACKEND env var and that the local path is wired up
correctly with a stub.
"""

import asyncio

from backend.reliability import claims as claims_mod
from backend.search.web_search import SearchHit


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
    """If the heavy ML deps aren't installed, _label_hits_local should
    not crash the request — it returns 'unclear' for every hit."""

    # Force the heavy import in _label_hits_local to fail by stubbing the
    # import system. We do this by patching the function itself to do the
    # same try/except that the production code does, against a sentinel
    # ImportError.
    import sys

    real_module = sys.modules.pop("backend.reliability.nli", None)

    class _BrokenLoader:
        def __getattr__(self, _name):
            raise ImportError("transformers not installed in this environment")

    sys.modules["backend.reliability.nli"] = _BrokenLoader()  # type: ignore[assignment]
    try:
        labels = asyncio.run(claims_mod._label_hits_local("c", _hits(2)))
        assert labels == ["unclear", "unclear"]
    finally:
        if real_module is not None:
            sys.modules["backend.reliability.nli"] = real_module
        else:
            sys.modules.pop("backend.reliability.nli", None)
