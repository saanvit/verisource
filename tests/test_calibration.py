"""Tests for the isotonic-regression calibrator.

We pick small inputs where the expected fit is derivable by hand so that
regressions in the PAV implementation are caught immediately.
"""

import json
from pathlib import Path

import pytest

from backend.reliability.calibration import (
    IsotonicCalibrator,
    fit_isotonic,
    load,
    save,
)


# ---------- PAV correctness ----------


def test_perfectly_separable_yields_step_at_threshold():
    # Scores 10/20/30 unreliable, 70/80/90 reliable → PAV should produce
    # a step function: low scores → 0, high scores → 100.
    scores = [10.0, 20.0, 30.0, 70.0, 80.0, 90.0]
    labels = [0, 0, 0, 1, 1, 1]
    cal = fit_isotonic(scores, labels)
    assert cal.apply(15) == 0.0
    assert cal.apply(80) == 100.0
    # Anything in the gap interpolates monotonically between 0 and 100.
    mid = cal.apply(50)
    assert 0 <= mid <= 100


def test_random_labels_give_constant_calibrator():
    # All labels at 0.5 prevalence with identical scores → PAV pools
    # everything into one block at 0.5.
    scores = [50.0] * 10
    labels = [0, 1] * 5
    cal = fit_isotonic(scores, labels)
    # Any input maps to the single block's value (50 / 100).
    assert cal.apply(50) == 50.0


def test_monotonicity_is_preserved():
    # Random scores + labels — the fit must be monotone non-decreasing.
    import random

    rng = random.Random(42)
    n = 60
    scores = [rng.uniform(0, 100) for _ in range(n)]
    # Skewed: higher score → higher P(reliable) on average.
    labels = [1 if (s / 100 + rng.uniform(-0.3, 0.3)) > 0.5 else 0 for s in scores]
    cal = fit_isotonic(scores, labels)
    # Sample the fitted function densely; check monotone.
    grid = [i for i in range(0, 101)]
    fitted = cal.apply_many([float(g) for g in grid])
    for a, b in zip(fitted, fitted[1:]):
        assert a <= b + 1e-9, f"non-monotone at {a} > {b}"


def test_perfect_calibration_is_approximately_identity():
    # If raw_score / 100 already matches empirical P(reliable), the fit
    # should approximately preserve scores. Use many samples to reduce
    # noise.
    import random

    rng = random.Random(0)
    n = 2000
    scores: list[float] = []
    labels: list[int] = []
    for _ in range(n):
        s = rng.uniform(0, 100)
        scores.append(s)
        labels.append(1 if rng.random() < s / 100 else 0)
    cal = fit_isotonic(scores, labels)
    # At 30, the fit should be within ±10 of 30; same at 70.
    assert abs(cal.apply(30.0) - 30.0) < 10.0
    assert abs(cal.apply(70.0) - 70.0) < 10.0


def test_brier_strictly_improves_on_training_data():
    # PAV is the optimal isotonic Brier minimizer — on the *training*
    # data it cannot do worse than the identity baseline.
    import random

    rng = random.Random(7)
    n = 200
    scores = [rng.uniform(0, 100) for _ in range(n)]
    labels = [1 if rng.uniform(0, 1) < (s / 100) ** 1.5 else 0 for s in scores]  # mis-calibrated
    cal = fit_isotonic(scores, labels)

    def brier(s_list: list[float]) -> float:
        return sum((s / 100 - y) ** 2 for s, y in zip(s_list, labels)) / len(labels)

    raw = brier(scores)
    fitted = brier(cal.apply_many(scores))
    assert fitted <= raw + 1e-9, (raw, fitted)


# ---------- Edge cases ----------


def test_empty_calibrator_is_identity():
    cal = IsotonicCalibrator(x=[], y=[])
    for s in (0.0, 25.0, 50.0, 75.0, 100.0):
        assert cal.apply(s) == s


def test_apply_clips_outside_training_range():
    # Train on [10, 50] only.
    cal = fit_isotonic([10, 20, 30, 40, 50], [0, 0, 1, 1, 1])
    assert cal.apply(0) == cal.apply(10)
    assert cal.apply(100) == cal.apply(50)


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        fit_isotonic([1.0, 2.0], [0])


# ---------- I/O ----------


def test_save_and_load_roundtrip(tmp_path: Path):
    cal = fit_isotonic([10, 50, 90], [0, 0, 1])
    p = tmp_path / "c.json"
    save(cal, p, metadata={"trained_on": "test", "extra": "yes"})

    raw = json.loads(p.read_text())
    assert raw["method"] == "isotonic-pav"
    assert raw["trained_on"] == "test"
    assert raw["extra"] == "yes"

    loaded = load(p)
    for s in (0.0, 30.0, 50.0, 70.0, 100.0):
        assert loaded.apply(s) == cal.apply(s)


# ---------- Runtime hook ----------


def test_get_runtime_returns_none_when_env_var_unset(monkeypatch):
    from backend.reliability import calibration as cal_mod

    monkeypatch.delenv("CALIBRATION_PATH", raising=False)
    # Reset module cache so the test is order-independent.
    cal_mod._runtime_calibrator = None
    cal_mod._loaded_path = None
    assert cal_mod.get_runtime() is None


def test_get_runtime_loads_and_caches(tmp_path: Path, monkeypatch):
    from backend.reliability import calibration as cal_mod

    cal = fit_isotonic([10, 50, 90], [0, 0, 1])
    p = tmp_path / "c.json"
    save(cal, p)

    monkeypatch.setenv("CALIBRATION_PATH", str(p))
    cal_mod._runtime_calibrator = None
    cal_mod._loaded_path = None
    first = cal_mod.get_runtime()
    assert first is not None
    # Second call hits the cache.
    second = cal_mod.get_runtime()
    assert second is first


def test_scorer_applies_calibrator_when_configured(tmp_path: Path, monkeypatch):
    """An end-to-end check that aggregate() routes scores through the calibrator."""
    from backend.models import (
        ContentAnalysis,
        CorroboratingSource,
        CrossReferenceResult,
        DomainReputation,
    )
    from backend.reliability import calibration as cal_mod
    from backend.reliability.scorer import aggregate

    # Build a calibrator that maps every raw score onto 30 (constant).
    p = tmp_path / "const.json"
    cal = IsotonicCalibrator(x=[0.0, 100.0], y=[30.0, 30.0])
    save(cal, p)

    monkeypatch.setenv("CALIBRATION_PATH", str(p))
    cal_mod._runtime_calibrator = None
    cal_mod._loaded_path = None

    domain = DomainReputation(domain="x.com", score=80, bias="center", type="news", rationale="r")
    content = ContentAnalysis(
        score=80, factuality=80, objectivity=80, transparency=80, sensationalism=80,
        main_claims=["c"], red_flags=[], citations_present=True, summary="ok",
    )
    xref = CrossReferenceResult(
        score=80, n_sources=3, n_high_quality=2,
        sources=[CorroboratingSource(
            title="t", url="https://e.com", domain="e.com", domain_score=80,
            snippet="s", agreement="supports",
        )],
        consensus="strong-support",
    )
    report = aggregate(domain, content, xref, has_text=True)
    # Even though raw inputs are all 80, the calibrator forces overall to 30.
    assert report.overall_score == 30.0
