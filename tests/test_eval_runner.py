"""End-to-end tests for the eval runner using the heuristic pipeline.

These tests do not require any API keys — the analyzer falls back to the
deterministic lexical heuristic and cross-reference is explicitly skipped.
The bundled dev set is small enough to run fast.
"""

import asyncio
from pathlib import Path

from backend.eval.datasets import EvalExample, load_dev, load_jsonl
from backend.eval.metrics import compute_metrics
from backend.eval.runner import run_eval


def test_load_dev_dataset_has_examples_of_every_label():
    examples = load_dev()
    labels = {ex.label for ex in examples}
    assert {"reliable", "unreliable", "satire", "mixed"} <= labels
    # Every example must have at least one of text/url.
    assert all(ex.text or ex.url for ex in examples)


def test_load_jsonl_rejects_missing_label(tmp_path: Path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"id": "x", "text": "hello"}\n')
    try:
        load_jsonl(p)
    except ValueError as exc:
        assert "label" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_load_jsonl_rejects_no_text_or_url(tmp_path: Path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"id": "x", "label": "reliable"}\n')
    try:
        load_jsonl(p)
    except ValueError as exc:
        assert "text" in str(exc) or "url" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_runner_smoke_on_dev_set_without_api_keys():
    """The pipeline should produce a score for every dev example offline.

    With no MISTRAL_API_KEY we fall through to the lexical heuristic and with
    skip_cross_reference=True we never hit Tavily. Domain reputation still
    drives most of the signal — so satire/conspiracy domains land low and
    wire-service domains land high, which is enough to validate the harness.
    """
    examples = load_dev()
    predictions = asyncio.run(
        run_eval(examples, concurrency=8, skip_cross_reference=True)
    )
    assert len(predictions) == len(examples)
    # Every prediction should succeed offline.
    assert all(p.error is None for p in predictions), [
        (p.id, p.error) for p in predictions if p.error
    ]
    assert all(p.predicted_score is not None for p in predictions)

    metrics = compute_metrics(
        [p.predicted_score for p in predictions],  # type: ignore[misc]
        [p.label for p in predictions],
    )
    # The dev set is curated so the baseline pipeline should land well above
    # chance — if this drops, either the pipeline regressed or the dev set
    # was tampered with.
    assert metrics.accuracy >= 0.75, metrics.accuracy
    assert metrics.macro_f1 >= 0.7, metrics.macro_f1


def test_runner_handles_text_only_example():
    examples = [
        EvalExample(id="raw", label="reliable", text="This is a short piece of text."),
    ]
    predictions = asyncio.run(run_eval(examples, concurrency=1, skip_cross_reference=True))
    assert predictions[0].predicted_score is not None
