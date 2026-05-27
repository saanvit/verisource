"""Build a unified NLI dataset for fine-tuning the stance labeler.

Loads FEVER + VitaminC (and optionally SciFact / ANLI) from the Hugging
Face Hub, normalizes labels to our 3-class scheme

    0 = supports        ("entails")
    1 = contradicts     ("refutes")
    2 = unclear         ("neutral" / "not enough info")

and yields ``(premise, hypothesis, label)`` records where:

    premise    = the evidence text from a retrieved hit
    hypothesis = the claim to be verified

This matches both the NLI convention used by DeBERTa-v3-large and the
labels emitted by ``backend.reliability.claims._label_hits``.

Run as a script to materialize a train/val/test split locally:

    python -m training.datasets --out training/data/nli_combined.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

# All heavy imports are lazy so importing this module without the NLI
# extras installed (e.g. for type-checking) doesn't fail.


LABEL_SUPPORTS = 0
LABEL_CONTRADICTS = 1
LABEL_UNCLEAR = 2

LABEL_NAMES = {
    LABEL_SUPPORTS: "supports",
    LABEL_CONTRADICTS: "contradicts",
    LABEL_UNCLEAR: "unclear",
}


# ---------- per-source loaders ----------

def _load_fever(max_per_label: int | None = None) -> list[dict]:
    """FEVER (Thorne et al. 2018). HF dataset id: ``fever`` / ``v1.0``.

    FEVER labels: SUPPORTS, REFUTES, NOT ENOUGH INFO. We use the
    pre-joined ``fever`` config that ships claim + evidence text rows.
    """
    from datasets import load_dataset

    raw = load_dataset("fever", "v1.0", split="train", trust_remote_code=True)
    label_map = {
        "SUPPORTS": LABEL_SUPPORTS,
        "REFUTES": LABEL_CONTRADICTS,
        "NOT ENOUGH INFO": LABEL_UNCLEAR,
    }
    out: list[dict] = []
    counts: Counter = Counter()
    for row in raw:
        label_str = row.get("label") or row.get("verifiable") or ""
        if label_str not in label_map:
            continue
        label = label_map[label_str]
        if max_per_label and counts[label] >= max_per_label:
            continue
        # FEVER's evidence is page-and-line-id pointers, not text. The
        # "fever" config bundles claim text but typically not the evidence
        # text — users who want the snippet need to also load Wikipedia.
        # For our purposes (training a stance labeler against retrieval
        # snippets), we treat the claim's own context window — but if the
        # row carries evidence_sentence (some forks add this), use it.
        claim = (row.get("claim") or "").strip()
        evidence = (
            row.get("evidence_sentence")
            or row.get("evidence_text")
            or row.get("evidence")
            or ""
        )
        if isinstance(evidence, list):
            # Fall back: any string in the structured-evidence list.
            evidence = " ".join(str(e) for e in evidence if e)
        evidence = str(evidence).strip()
        if not claim or not evidence:
            continue
        out.append(
            {
                "source": "fever",
                "premise": evidence[:1024],
                "hypothesis": claim[:512],
                "label": label,
            }
        )
        counts[label] += 1
    return out


def _load_vitaminc(max_per_label: int | None = None) -> list[dict]:
    """VitaminC (Schuster et al. 2021). HF dataset id: ``tals/vitaminc``.

    Provides claim + evidence text and three labels (SUPPORTS / REFUTES /
    NOT ENOUGH INFO). Notable for adversarially-perturbed evidence pairs,
    which makes the model more robust to subtle wording changes than
    FEVER alone.
    """
    from datasets import load_dataset

    raw = load_dataset("tals/vitaminc", split="train")
    label_map = {
        "SUPPORTS": LABEL_SUPPORTS,
        "REFUTES": LABEL_CONTRADICTS,
        "NOT ENOUGH INFO": LABEL_UNCLEAR,
    }
    out: list[dict] = []
    counts: Counter = Counter()
    for row in raw:
        ls = row.get("label")
        if ls not in label_map:
            continue
        label = label_map[ls]
        if max_per_label and counts[label] >= max_per_label:
            continue
        premise = (row.get("evidence") or "").strip()
        hypothesis = (row.get("claim") or "").strip()
        if not premise or not hypothesis:
            continue
        out.append(
            {
                "source": "vitaminc",
                "premise": premise[:1024],
                "hypothesis": hypothesis[:512],
                "label": label,
            }
        )
        counts[label] += 1
    return out


def _load_scifact(max_per_label: int | None = None) -> list[dict]:
    """SciFact (Wadden et al. 2020). HF dataset id: ``allenai/scifact``.

    Smaller (~1.4K) but science-domain — useful for claims about studies
    and medical findings, which are common adversarial cases for our app.
    """
    from datasets import load_dataset

    raw = load_dataset("allenai/scifact", "claims", split="train")
    out: list[dict] = []
    counts: Counter = Counter()
    # SciFact labels per evidence document: SUPPORT / CONTRADICT / NEI.
    # We unroll (claim, evidence_doc, label) into per-sentence records
    # using the cited_doc_ids and a separate corpus pull.
    corpus = load_dataset("allenai/scifact", "corpus", split="train")
    doc_text: dict[int, str] = {
        int(row["doc_id"]): " ".join(row.get("abstract") or []) for row in corpus
    }
    label_map = {"SUPPORT": LABEL_SUPPORTS, "CONTRADICT": LABEL_CONTRADICTS}
    for row in raw:
        claim = (row.get("claim") or "").strip()
        if not claim:
            continue
        evidence: dict = row.get("evidence") or {}
        cited_ids = row.get("cited_doc_ids") or []
        if not evidence:
            # No evidence labeled → treat as unclear against any cited doc.
            for doc_id in cited_ids:
                premise = doc_text.get(int(doc_id), "").strip()
                if not premise:
                    continue
                if max_per_label and counts[LABEL_UNCLEAR] >= max_per_label:
                    break
                out.append(
                    {
                        "source": "scifact",
                        "premise": premise[:1024],
                        "hypothesis": claim[:512],
                        "label": LABEL_UNCLEAR,
                    }
                )
                counts[LABEL_UNCLEAR] += 1
            continue
        for doc_id_str, anns in evidence.items():
            premise = doc_text.get(int(doc_id_str), "").strip()
            if not premise:
                continue
            label_str = anns[0].get("label") if anns else None
            if label_str not in label_map:
                continue
            label = label_map[label_str]
            if max_per_label and counts[label] >= max_per_label:
                continue
            out.append(
                {
                    "source": "scifact",
                    "premise": premise[:1024],
                    "hypothesis": claim[:512],
                    "label": label,
                }
            )
            counts[label] += 1
    return out


# ---------- public surface ----------


SOURCES = {
    "fever": _load_fever,
    "vitaminc": _load_vitaminc,
    "scifact": _load_scifact,
}


def build_combined(
    sources: list[str] | None = None,
    max_per_label_per_source: int | None = None,
    seed: int = 0,
    val_frac: float = 0.02,
    test_frac: float = 0.02,
) -> dict[str, list[dict]]:
    """Build the unified train/val/test splits.

    The split is example-level and stratified by label. For tiny
    ``max_per_label_per_source`` values (smoke tests), test/val may end up
    empty for some labels; that's acceptable for sanity checks.
    """
    sources = sources or ["fever", "vitaminc"]
    rng = random.Random(seed)

    all_records: list[dict] = []
    for s in sources:
        if s not in SOURCES:
            raise ValueError(f"unknown source: {s!r} (expected one of {list(SOURCES)})")
        all_records.extend(SOURCES[s](max_per_label=max_per_label_per_source))

    rng.shuffle(all_records)

    by_label: dict[int, list[dict]] = {
        LABEL_SUPPORTS: [],
        LABEL_CONTRADICTS: [],
        LABEL_UNCLEAR: [],
    }
    for r in all_records:
        by_label[r["label"]].append(r)

    train: list[dict] = []
    val: list[dict] = []
    test: list[dict] = []
    for _, recs in by_label.items():
        n = len(recs)
        n_val = int(n * val_frac)
        n_test = int(n * test_frac)
        val.extend(recs[:n_val])
        test.extend(recs[n_val : n_val + n_test])
        train.extend(recs[n_val + n_test :])
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return {"train": train, "val": val, "test": test}


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--sources", nargs="+", default=["fever", "vitaminc"])
    p.add_argument("--out", type=Path, default=Path("training/data"))
    p.add_argument(
        "--max-per-label",
        type=int,
        default=None,
        help="Cap examples per (source, label) — useful for smoke tests.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val-frac", type=float, default=0.02)
    p.add_argument("--test-frac", type=float, default=0.02)
    args = p.parse_args(argv)

    splits = build_combined(
        sources=args.sources,
        max_per_label_per_source=args.max_per_label,
        seed=args.seed,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
    )
    for name, rows in splits.items():
        out_path = args.out / f"{name}.jsonl"
        write_jsonl(rows, out_path)
        counts = Counter(r["label"] for r in rows)
        print(
            f"{name}: {len(rows):>8d} rows  →  {out_path}  "
            f"(supports={counts[LABEL_SUPPORTS]}, "
            f"contradicts={counts[LABEL_CONTRADICTS]}, "
            f"unclear={counts[LABEL_UNCLEAR]})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
