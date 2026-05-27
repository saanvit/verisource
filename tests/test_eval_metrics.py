"""Tests for the eval metric math.

We pick small inputs where every metric can be derived by hand, so that
future changes to the metric implementation don't silently regress.
"""

import math

from backend.eval.metrics import (
    ConfusionMatrix,
    _calibration_bins,
    _roc_auc,
    compute_metrics,
)


def test_confusion_matrix_perfect():
    cm = ConfusionMatrix(tp=10, fp=0, tn=10, fn=0)
    assert cm.accuracy == 1.0
    assert cm.precision("reliable") == 1.0
    assert cm.recall("reliable") == 1.0
    assert cm.f1("reliable") == 1.0
    assert cm.f1("unreliable") == 1.0


def test_confusion_matrix_random():
    # 4 of 8 right.
    cm = ConfusionMatrix(tp=2, fp=2, tn=2, fn=2)
    assert cm.accuracy == 0.5
    assert cm.precision("reliable") == 0.5
    assert cm.recall("reliable") == 0.5
    assert cm.f1("reliable") == 0.5


def test_roc_auc_perfect_separation():
    scores = [0.1, 0.2, 0.3, 0.8, 0.9, 1.0]
    labels = [0, 0, 0, 1, 1, 1]
    assert _roc_auc(scores, labels) == 1.0


def test_roc_auc_reversed():
    scores = [0.9, 0.8, 0.7, 0.2, 0.1, 0.0]
    labels = [0, 0, 0, 1, 1, 1]
    assert _roc_auc(scores, labels) == 0.0


def test_roc_auc_random_with_ties():
    # All scores tied → AUC = 0.5 regardless of label distribution.
    scores = [0.5, 0.5, 0.5, 0.5]
    labels = [1, 0, 1, 0]
    assert _roc_auc(scores, labels) == 0.5


def test_roc_auc_handcomputed():
    # Two positives (0.8, 0.6), two negatives (0.4, 0.2). All four pos-vs-neg
    # pairs satisfy pos > neg, so AUC = 1.0.
    assert _roc_auc([0.8, 0.4, 0.6, 0.2], [1, 0, 1, 0]) == 1.0
    # Re-label so positives are 0.4 and 0.6, negatives are 0.8 and 0.2.
    # Pairs: (0.4,0.8) wrong, (0.4,0.2) right, (0.6,0.8) wrong, (0.6,0.2) right.
    # 2 of 4 correct → AUC = 0.5.
    assert _roc_auc([0.4, 0.8, 0.6, 0.2], [1, 0, 1, 0]) == 0.5
    # Now use [0.8,0.4,0.7,0.2] with same labels: positives 0.8, 0.7 vs
    # negatives 0.4, 0.2 → all pairs correct → AUC = 1.0.
    assert _roc_auc([0.8, 0.4, 0.7, 0.2], [1, 0, 1, 0]) == 1.0


def test_calibration_bins_count_and_means():
    probs = [0.05, 0.15, 0.55, 0.95, 0.95]
    labels = [0, 0, 1, 1, 1]
    bins = _calibration_bins(probs, labels, n_bins=10)
    # 10 bins, only some populated.
    assert len(bins) == 10
    populated = [b for b in bins if b.n > 0]
    assert sum(b.n for b in populated) == 5
    # The 0.95s land in the last bin (inclusive on the right).
    last = bins[-1]
    assert last.n == 2
    assert last.frac_reliable == 1.0


def test_compute_metrics_end_to_end():
    # 6 examples; 3 reliable scored high, 3 unreliable scored low — perfect.
    scores = [90.0, 80.0, 70.0, 20.0, 15.0, 5.0]
    labels = ["reliable", "reliable", "reliable", "unreliable", "unreliable", "unreliable"]
    m = compute_metrics(scores, labels, threshold=50.0)
    assert m.n == 6
    assert m.n_skipped_mixed == 0
    assert m.accuracy == 1.0
    assert m.macro_f1 == 1.0
    assert m.roc_auc == 1.0
    # Brier on perfectly calibrated extremes is small but not zero.
    assert 0 <= m.brier < 0.1


def test_compute_metrics_with_mixed_examples_skipped_from_classification():
    # The mixed example does not affect classification but does affect MAE.
    scores = [90.0, 10.0, 50.0]
    labels = ["reliable", "unreliable", "mixed"]
    m = compute_metrics(scores, labels)
    assert m.n == 3
    assert m.n_skipped_mixed == 1
    assert m.confusion.tp == 1
    assert m.confusion.tn == 1
    assert m.accuracy == 1.0
    # MAE is computed over all 3 — mixed target = 0.5, pred = 0.5 → 0.
    assert math.isclose(m.mae, (0.1 + 0.1 + 0.0) / 3, rel_tol=1e-6)


def test_compute_metrics_alternative_label_names_are_normalized():
    # "true" / "fake" are common in fake-news datasets — should be normalized.
    scores = [85.0, 10.0]
    labels = ["true", "fake"]
    m = compute_metrics(scores, labels)
    assert m.accuracy == 1.0
    assert m.confusion.tp == 1
    assert m.confusion.tn == 1


def test_compute_metrics_satire_treated_as_unreliable():
    scores = [5.0, 80.0]
    labels = ["satire", "reliable"]
    m = compute_metrics(scores, labels)
    assert m.accuracy == 1.0
    assert m.confusion.tn == 1  # satire predicted as unreliable, low score


def test_compute_metrics_wrong_predictions_lower_f1():
    scores = [10.0, 90.0]  # inverted
    labels = ["reliable", "unreliable"]
    m = compute_metrics(scores, labels)
    assert m.accuracy == 0.0
    assert m.f1_reliable == 0.0
    assert m.f1_unreliable == 0.0
