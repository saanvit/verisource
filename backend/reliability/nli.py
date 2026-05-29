"""Local NLI stance labeler.

A drop-in alternative to the Mistral-based stance call in
``claims._label_hits``. Loads a 3-class sequence classifier (premise =
evidence, hypothesis = claim) and returns one of ``supports`` /
``contradicts`` / ``unclear`` per (claim, hit) pair.

Two ways to configure the model:

* ``NLI_MODEL`` env var → HF Hub id or local checkpoint path. If unset,
  falls back to a public off-the-shelf NLI model. After fine-tuning via
  ``training/train_nli.py`` you can simply

      export NLI_MODEL=/abs/path/to/models/nli/deberta-v3-large-fever-vitc

  and inference will use your checkpoint.

* ``NLI_DEVICE`` env var → ``cuda`` / ``mps`` / ``cpu``. Auto-detected if
  unset.

The default off-the-shelf model is fine-tuned on MNLI + FEVER + ANLI +
others, with the standard 3-class entailment/contradiction/neutral
output, which we map to our labels.

All heavy imports are lazy so the rest of the project does not need
``transformers`` / ``torch`` installed unless this module is actually
used.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Any

from backend.search.web_search import SearchHit

DEFAULT_MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
# Off-the-shelf models output labels in the standard NLI order; our local
# fine-tuned checkpoints use the order defined in training.datasets.
_OFFTHESHELF_LABEL_TO_OURS = {
    "entailment": "supports",
    "ENTAILMENT": "supports",
    "supports": "supports",
    "SUPPORTS": "supports",
    "contradiction": "contradicts",
    "CONTRADICTION": "contradicts",
    "refutes": "contradicts",
    "REFUTES": "contradicts",
    "contradicts": "contradicts",
    "neutral": "unclear",
    "NEUTRAL": "unclear",
    "unclear": "unclear",
    "UNCLEAR": "unclear",
    "NOT ENOUGH INFO": "unclear",
}


@dataclass
class _Loaded:
    tokenizer: Any
    model: Any
    device: str
    id2label: dict[int, str]


_state: _Loaded | None = None
_load_lock = threading.Lock()


def _detect_device() -> str:
    forced = (os.getenv("NLI_DEVICE") or "").strip().lower()
    if forced in ("cuda", "mps", "cpu"):
        return forced
    try:
        import torch  # local import: only when used

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:  # pragma: no cover - torch missing or broken
        pass
    return "cpu"


def _load() -> _Loaded:
    global _state
    if _state is not None:
        return _state
    with _load_lock:
        if _state is not None:
            return _state
        from transformers import (  # local import
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )
        import torch

        model_id = os.getenv("NLI_MODEL", DEFAULT_MODEL)
        device = _detect_device()
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSequenceClassification.from_pretrained(model_id)
        model.eval()
        if device != "cpu":
            model = model.to(device)
        # Resolve label mapping. HF stores it as id2label in config, but
        # some checkpoints lack it; fall back to position-based ordering.
        id2label_raw = getattr(model.config, "id2label", None) or {}
        id2label: dict[int, str] = {}
        for k, v in id2label_raw.items():
            try:
                id2label[int(k)] = str(v)
            except (TypeError, ValueError):
                continue
        if not id2label:
            id2label = {0: "supports", 1: "contradicts", 2: "unclear"}
        _state = _Loaded(tokenizer=tokenizer, model=model, device=device, id2label=id2label)
        return _state


def _normalize_label(raw: str) -> str:
    return _OFFTHESHELF_LABEL_TO_OURS.get(raw, _OFFTHESHELF_LABEL_TO_OURS.get(raw.lower(), "unclear"))


def _unclear_margin() -> float:
    """How much the unclear class must beat its best rival to actually win."""
    raw = os.getenv("NLI_UNCLEAR_MARGIN", "0.15").strip()
    try:
        return max(0.0, min(1.0, float(raw)))
    except ValueError:
        return 0.15


def _pick_label(probs: list[float], id2label: dict[int, str], unclear_margin: float) -> str:
    """Map a per-class probability vector to a single normalized label.

    Decision rule: if "unclear" is the argmax but only beats the better
    of supports/contradicts by less than ``unclear_margin``, commit to
    the better non-unclear class instead. This counteracts the
    empirically-observed conservatism of NLI models trained on clean
    benchmarks when they see messy real-news evidence — they often
    return ~0.40 unclear / ~0.35 supports for clearly-supporting hits,
    which plain argmax would call "unclear" and the per-claim aggregator
    would treat as zero signal.

    All probability comparisons happen on the *normalized* (our 3-class)
    label space, so this works identically for off-the-shelf checkpoints
    (entailment/contradiction/neutral) and our fine-tuned checkpoint
    (supports/contradicts/unclear).
    """
    # Aggregate probability mass by our 3-class scheme — unknown raw
    # labels fall through to "unclear" via _normalize_label.
    by_label: dict[str, float] = {"supports": 0.0, "contradicts": 0.0, "unclear": 0.0}
    for i, p in enumerate(probs):
        raw = id2label.get(int(i), str(i))
        by_label[_normalize_label(raw)] += float(p)

    sup = by_label["supports"]
    con = by_label["contradicts"]
    unc = by_label["unclear"]

    # If unclear isn't the top class, plain argmax of the other two.
    if unc < sup or unc < con:
        return "supports" if sup >= con else "contradicts"

    # Unclear is the top class. Check whether its lead over the better
    # of supports/contradicts is wide enough to trust.
    best_other = max(sup, con)
    if unc - best_other >= unclear_margin:
        return "unclear"
    return "supports" if sup >= con else "contradicts"


def label_hits_nli(
    claim: str,
    hits: list[SearchHit],
    *,
    batch_size: int = 16,
    max_length: int = 256,
) -> list[str]:
    """Return ``supports`` / ``contradicts`` / ``unclear`` per hit.

    Matches the shape of ``claims._label_hits`` so it can be swapped in
    transparently.
    """
    if not hits:
        return []

    state = _load()
    import torch

    # Build (premise, hypothesis) pairs. The evidence snippet is the
    # premise; the claim is the hypothesis. We optionally prepend the
    # title to give the model a bit more context.
    premises: list[str] = []
    for h in hits:
        prefix = (h.title or "").strip()
        snippet = (h.snippet or "").strip()
        premises.append(f"{prefix}\n{snippet}" if prefix else snippet)
    hypotheses = [claim] * len(hits)

    margin = _unclear_margin()
    labels_out: list[str] = []
    for start in range(0, len(hits), batch_size):
        batch_p = premises[start : start + batch_size]
        batch_h = hypotheses[start : start + batch_size]
        inputs = state.tokenizer(
            batch_p,
            batch_h,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        if state.device != "cpu":
            inputs = {k: v.to(state.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = state.model(**inputs).logits
            # Softmax over the 3 classes — we need actual probabilities
            # for the unclear-margin decision rule, not just argmax.
            probs = torch.softmax(logits, dim=-1).tolist()
        for prob_vector in probs:
            labels_out.append(_pick_label(prob_vector, state.id2label, margin))
    return labels_out


def available() -> bool:
    """Cheap probe: does this environment have the deps installed?"""
    try:
        import transformers  # noqa: F401
        import torch  # noqa: F401

        return True
    except Exception:
        return False
