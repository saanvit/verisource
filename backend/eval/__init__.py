"""Evaluation harness for the Source Reliability Assessor.

Provides a pluggable pipeline interface and reference implementations so that
new pipeline variants (per-claim verification, fine-tuned NLI cross-reference,
etc.) can be A/B-compared against the current production pipeline on the
same labeled examples.
"""

from backend.eval.datasets import EvalExample, load_dataset
from backend.eval.metrics import EvalMetrics, compute_metrics
from backend.eval.runner import (
    EvalPrediction,
    default_pipeline,
    per_claim_pipeline,
    run_eval,
)

__all__ = [
    "EvalExample",
    "EvalMetrics",
    "EvalPrediction",
    "compute_metrics",
    "default_pipeline",
    "load_dataset",
    "per_claim_pipeline",
    "run_eval",
]
