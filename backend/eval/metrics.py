"""Classification + calibration metrics for reliability scores.

The pipeline emits a 0-100 ``overall_score``. We map it to a probability
(score / 100) and to a binary label by thresholding (default 50). Examples
labeled ``mixed`` are excluded from binary classification metrics — they
exist so a future ordinal metric can still use them, but precision/recall
on a binary task would be ill-defined for them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import sqrt


# Canonical binary labels used internally.
RELIABLE = "reliable"
UNRELIABLE = "unreliable"
MIXED = "mixed"


@dataclass
class ConfusionMatrix:
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    @property
    def accuracy(self) -> float:
        n = self.tp + self.fp + self.tn + self.fn
        return (self.tp + self.tn) / n if n else 0.0

    def precision(self, positive: str) -> float:
        if positive == RELIABLE:
            denom = self.tp + self.fp
            return self.tp / denom if denom else 0.0
        denom = self.tn + self.fn
        return self.tn / denom if denom else 0.0

    def recall(self, positive: str) -> float:
        if positive == RELIABLE:
            denom = self.tp + self.fn
            return self.tp / denom if denom else 0.0
        denom = self.tn + self.fp
        return self.tn / denom if denom else 0.0

    def f1(self, positive: str) -> float:
        p = self.precision(positive)
        r = self.recall(positive)
        return 2 * p * r / (p + r) if (p + r) else 0.0


@dataclass
class CalibrationBin:
    lo: float
    hi: float
    n: int
    mean_pred: float
    frac_reliable: float


@dataclass
class EvalMetrics:
    n: int
    n_skipped_mixed: int
    threshold: float
    confusion: ConfusionMatrix
    accuracy: float
    precision_reliable: float
    recall_reliable: float
    f1_reliable: float
    precision_unreliable: float
    recall_unreliable: float
    f1_unreliable: float
    macro_f1: float
    roc_auc: float
    brier: float
    mae: float
    calibration: list[CalibrationBin] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["confusion"] = asdict(self.confusion)
        d["calibration"] = [asdict(b) for b in self.calibration]
        return d


def _roc_auc(scores: list[float], labels: list[int]) -> float:
    """Mann-Whitney U formulation of ROC-AUC, with mid-rank for ties."""
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    paired = sorted(zip(scores, labels), key=lambda x: x[0])
    ranks = [0.0] * len(paired)
    i = 0
    while i < len(paired):
        j = i
        while j + 1 < len(paired) and paired[j + 1][0] == paired[i][0]:
            j += 1
        avg_rank = (i + j) / 2 + 1  # 1-indexed average rank
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        i = j + 1

    sum_ranks_pos = sum(r for r, (_, y) in zip(ranks, paired) if y == 1)
    u = sum_ranks_pos - n_pos * (n_pos + 1) / 2
    return u / (n_pos * n_neg)


def _brier(probs: list[float], labels: list[int]) -> float:
    if not labels:
        return 0.0
    return sum((p - y) ** 2 for p, y in zip(probs, labels)) / len(labels)


def _calibration_bins(
    probs: list[float], labels: list[int], n_bins: int = 10
) -> list[CalibrationBin]:
    bins: list[CalibrationBin] = []
    for i in range(n_bins):
        lo = i / n_bins
        hi = (i + 1) / n_bins
        # Last bin is inclusive on the right.
        bucket = [
            (p, y)
            for p, y in zip(probs, labels)
            if (lo <= p < hi) or (i == n_bins - 1 and p == hi)
        ]
        if not bucket:
            bins.append(CalibrationBin(lo=lo, hi=hi, n=0, mean_pred=0.0, frac_reliable=0.0))
            continue
        mean_pred = sum(p for p, _ in bucket) / len(bucket)
        frac = sum(y for _, y in bucket) / len(bucket)
        bins.append(
            CalibrationBin(lo=lo, hi=hi, n=len(bucket), mean_pred=mean_pred, frac_reliable=frac)
        )
    return bins


def _normalize_label(label: str) -> str:
    """Map dataset-specific labels to {reliable, unreliable, mixed}."""
    lo = label.strip().lower().replace("-", "_").replace(" ", "_")
    if lo in {"reliable", "true", "real", "credible", "trustworthy", "highly_reliable",
              "generally_reliable", "mostly_true"}:
        return RELIABLE
    if lo in {"unreliable", "fake", "false", "satire", "conspiracy", "questionable",
              "pants_fire", "barely_true"}:
        return UNRELIABLE
    return MIXED


def compute_metrics(
    predicted_scores: list[float],
    true_labels: list[str],
    *,
    threshold: float = 50.0,
    n_bins: int = 10,
) -> EvalMetrics:
    """Compute classification + calibration metrics.

    ``predicted_scores`` are 0-100. ``true_labels`` are arbitrary strings that
    get normalized into {reliable, unreliable, mixed}; mixed examples are
    skipped from binary classification metrics but counted for MAE.
    """

    if len(predicted_scores) != len(true_labels):
        raise ValueError("predicted_scores and true_labels must be the same length")

    cm = ConfusionMatrix()
    probs: list[float] = []
    bin_labels: list[int] = []
    abs_err_sum = 0.0
    abs_err_n = 0
    n_mixed = 0

    for score, raw_label in zip(predicted_scores, true_labels):
        norm = _normalize_label(raw_label)
        # Centered around the threshold: 0 if unreliable, 1 if reliable, 0.5 if mixed.
        target = 1.0 if norm == RELIABLE else (0.0 if norm == UNRELIABLE else 0.5)
        prob = max(0.0, min(1.0, score / 100.0))
        abs_err_sum += abs(prob - target)
        abs_err_n += 1

        if norm == MIXED:
            n_mixed += 1
            continue
        y = 1 if norm == RELIABLE else 0
        probs.append(prob)
        bin_labels.append(y)

        predicted_reliable = score >= threshold
        if y == 1 and predicted_reliable:
            cm.tp += 1
        elif y == 1 and not predicted_reliable:
            cm.fn += 1
        elif y == 0 and predicted_reliable:
            cm.fp += 1
        else:
            cm.tn += 1

    auc = _roc_auc(probs, bin_labels) if probs else float("nan")
    brier = _brier(probs, bin_labels)
    calibration = _calibration_bins(probs, bin_labels, n_bins=n_bins) if probs else []

    return EvalMetrics(
        n=len(predicted_scores),
        n_skipped_mixed=n_mixed,
        threshold=threshold,
        confusion=cm,
        accuracy=cm.accuracy,
        precision_reliable=cm.precision(RELIABLE),
        recall_reliable=cm.recall(RELIABLE),
        f1_reliable=cm.f1(RELIABLE),
        precision_unreliable=cm.precision(UNRELIABLE),
        recall_unreliable=cm.recall(UNRELIABLE),
        f1_unreliable=cm.f1(UNRELIABLE),
        macro_f1=(cm.f1(RELIABLE) + cm.f1(UNRELIABLE)) / 2,
        roc_auc=auc,
        brier=brier,
        mae=abs_err_sum / abs_err_n if abs_err_n else 0.0,
        calibration=calibration,
    )


def format_summary(metrics: EvalMetrics) -> str:
    cm = metrics.confusion
    auc_str = f"{metrics.roc_auc:.3f}" if metrics.roc_auc == metrics.roc_auc else "n/a"
    lines = [
        f"n={metrics.n}  (mixed skipped from binary metrics: {metrics.n_skipped_mixed})",
        f"threshold (reliable if score >= {metrics.threshold:.0f})",
        "",
        "             pred-reliable   pred-unreliable",
        f"true-reliable     {cm.tp:>5d}            {cm.fn:>5d}",
        f"true-unreliable   {cm.fp:>5d}            {cm.tn:>5d}",
        "",
        f"accuracy:           {metrics.accuracy:.3f}",
        f"precision(reliable):{metrics.precision_reliable:.3f}   "
        f"recall(reliable):  {metrics.recall_reliable:.3f}   "
        f"F1(reliable):  {metrics.f1_reliable:.3f}",
        f"precision(unrel.): {metrics.precision_unreliable:.3f}   "
        f"recall(unrel.):   {metrics.recall_unreliable:.3f}   "
        f"F1(unrel.):   {metrics.f1_unreliable:.3f}",
        f"macro-F1:           {metrics.macro_f1:.3f}",
        f"ROC-AUC:            {auc_str}",
        f"Brier:              {metrics.brier:.4f}",
        f"MAE(prob,target):   {metrics.mae:.4f}",
    ]
    return "\n".join(lines)
