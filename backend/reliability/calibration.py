"""Score calibration for the reliability pipelines.

The raw ``overall_score`` produced by the pipelines is *informative* but
not *calibrated* — a score of 70 doesn't necessarily mean "70% chance the
article is reliable." This module fits a monotonic mapping from raw
scores onto empirically-observed reliable fractions using **isotonic
regression** (pool-adjacent-violators), then applies that mapping at
inference time.

Why isotonic and not Platt:
* We have enough labels (a few hundred) for a non-parametric fit.
* Isotonic preserves the score ordering (no rank inversions), so the
  underlying ranking metrics (ROC-AUC) are unchanged.
* It handles non-sigmoid-shaped miscalibration — common when a pipeline
  is *conservative*, producing scores clustered around the prior of 50.

Calibrator file format (JSON):

    {
      "version": 1,
      "method": "isotonic-pav",
      "trained_on": "<dataset path>",
      "n_train": <int>,
      "n_test":  <int>,
      "metrics_before": {...},
      "metrics_after":  {...},
      "x": [thresholds, ascending],
      "y": [calibrated scores 0-100, monotone non-decreasing]
    }

The calibrator is applied by linear interpolation in (x, y); raw scores
outside the training range clip to the nearest endpoint.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class IsotonicCalibrator:
    """Piecewise-linear monotone map raw_score → calibrated_score (0-100).

    Built from sample weights of 1.0 by default; weights are accepted for
    importance-weighted fits but unused in the basic path.
    """

    x: list[float]
    y: list[float]

    def apply(self, score: float) -> float:
        """Map one raw score (0-100) to a calibrated score (0-100)."""
        if not self.x:
            return float(score)
        # Clip outside the training range to the nearest endpoint.
        if score <= self.x[0]:
            return float(self.y[0])
        if score >= self.x[-1]:
            return float(self.y[-1])
        # Linear interpolation.
        lo, hi = 0, len(self.x) - 1
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if self.x[mid] <= score:
                lo = mid
            else:
                hi = mid
        x0, x1 = self.x[lo], self.x[lo + 1]
        y0, y1 = self.y[lo], self.y[lo + 1]
        if x1 == x0:
            return float(y0)
        t = (score - x0) / (x1 - x0)
        return float(y0 + t * (y1 - y0))

    def apply_many(self, scores: list[float]) -> list[float]:
        return [self.apply(s) for s in scores]

    def to_dict(self) -> dict[str, Any]:
        return {"x": list(self.x), "y": list(self.y)}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "IsotonicCalibrator":
        return cls(x=[float(v) for v in d["x"]], y=[float(v) for v in d["y"]])


# ---------- Pool Adjacent Violators ----------


def _pav(x: list[float], y_target: list[float], w: list[float]) -> tuple[list[float], list[float]]:
    """Pool-adjacent-violators isotonic regression.

    Inputs are sorted ascending by x; y_target are the target values (here
    in [0, 1] — empirical reliable fractions); w are sample weights.

    Returns (x_thresholds, y_fit) — the fitted step function, returned as
    block midpoints with their pooled target value. The caller turns this
    into a piecewise-linear function suitable for interpolation.
    """
    n = len(x)
    if n == 0:
        return [], []
    # First, pool entries with identical x into one block — PAV's
    # adjacent-merging assumes strict x ordering, so tied x values must
    # be combined upfront or the merge step won't reach them.
    blocks: list[list[float]] = []
    i = 0
    while i < n:
        j = i
        while j + 1 < n and x[j + 1] == x[i]:
            j += 1
        total_wy = sum(w[k] * y_target[k] for k in range(i, j + 1))
        total_w = sum(w[k] for k in range(i, j + 1))
        blocks.append([total_wy, total_w, x[i], x[i]])
        i = j + 1
    # PAV: merge adjacent blocks while a violation exists.
    i = 0
    while i + 1 < len(blocks):
        b0 = blocks[i]
        b1 = blocks[i + 1]
        v0 = b0[0] / b0[1]
        v1 = b1[0] / b1[1]
        if v0 <= v1:
            i += 1
            continue
        # Merge b0 and b1 (violation: out of order).
        merged = [b0[0] + b1[0], b0[1] + b1[1], b0[2], b1[3]]
        blocks[i : i + 2] = [merged]
        # Step back to re-check the previous pair.
        if i > 0:
            i -= 1
    # Materialize as a piecewise-linear function: place a knot at each
    # block boundary, with the fitted value on each side.
    xs: list[float] = []
    ys: list[float] = []
    for b in blocks:
        total_w_y, total_w, x_lo, x_hi = b
        v = total_w_y / total_w
        xs.extend([x_lo, x_hi])
        ys.extend([v, v])
    # Deduplicate consecutive identical x's, keeping the first one.
    out_x: list[float] = []
    out_y: list[float] = []
    for xi, yi in zip(xs, ys):
        if out_x and xi == out_x[-1]:
            out_y[-1] = yi  # keep the later, which equals the same block's y
            continue
        out_x.append(xi)
        out_y.append(yi)
    return out_x, out_y


# ---------- Fit ----------


def fit_isotonic(
    scores: list[float],
    labels: list[int],
    *,
    weights: list[float] | None = None,
) -> IsotonicCalibrator:
    """Fit raw_score → P(reliable) and rescale onto a 0-100 calibrated score.

    ``scores`` are raw 0-100 values; ``labels`` are 0/1 (1=reliable).
    Returns a calibrator that maps raw → calibrated 0-100.
    """
    if not scores:
        return IsotonicCalibrator(x=[], y=[])
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have the same length")
    w = weights or [1.0] * len(scores)
    # Sort by raw score ascending.
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    sx = [float(scores[i]) for i in order]
    sy = [float(labels[i]) for i in order]
    sw = [float(w[i]) for i in order]
    xs, ps = _pav(sx, sy, sw)  # ps in [0, 1]
    # Map probabilities to 0-100 calibrated scores. We keep this linear so
    # downstream thresholding at 50 maps to P(reliable) = 0.5.
    ys = [round(p * 100.0, 2) for p in ps]
    return IsotonicCalibrator(x=xs, y=ys)


# ---------- I/O ----------


def save(calibrator: IsotonicCalibrator, path: Path | str, *, metadata: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out: dict[str, Any] = {
        "version": 1,
        "method": "isotonic-pav",
        **(metadata or {}),
        **calibrator.to_dict(),
    }
    path.write_text(json.dumps(out, indent=2))


def load(path: Path | str) -> IsotonicCalibrator:
    data = json.loads(Path(path).read_text())
    return IsotonicCalibrator.from_dict(data)


# ---------- Singleton for runtime use ----------

_runtime_calibrator: IsotonicCalibrator | None = None
_loaded_path: str | None = None


def get_runtime() -> IsotonicCalibrator | None:
    """Lazy-load the calibrator pointed to by the ``CALIBRATION_PATH`` env var.

    Returns None when the env var is unset or missing — the pipeline then
    leaves raw scores untouched. Cached after the first successful load
    so we don't hit disk on every request.
    """
    import os

    global _runtime_calibrator, _loaded_path
    path = os.getenv("CALIBRATION_PATH")
    if not path:
        return None
    if _runtime_calibrator is not None and _loaded_path == path:
        return _runtime_calibrator
    try:
        _runtime_calibrator = load(path)
        _loaded_path = path
        return _runtime_calibrator
    except Exception:  # pragma: no cover - corrupt file or missing path
        return None
