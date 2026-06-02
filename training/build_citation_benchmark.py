"""Build a synthetic mismatched-citation benchmark for the citation audit.

Citation audit asks: does the source a sentence links to actually support
that sentence? There is no off-the-shelf labeled corpus for this, so we
construct one that isolates the core decision (sentence vs. source-text
entailment) without depending on live web fetches:

  * POSITIVES  — each citing sentence paired with a source snippet that
                 does support it  (label "supported").
  * NEGATIVES  — the same sentence paired with a *different* base's source
                 snippet (label "unsupported") — i.e. a mislinked citation.

Writes ``data/eval/citation_audit_dev.jsonl`` with rows:
    {"id", "label": "supported"|"unsupported",
     "citing_sentence": "...", "source_text": "..."}

Reproducible (seeded); no network or API keys required to build.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

# (citing sentence, a source snippet that genuinely supports it).
BASES: list[tuple[str, str]] = [
    (
        "The James Webb Space Telescope launched on December 25, 2021.",
        "NASA's James Webb Space Telescope launched on 25 December 2021 aboard an "
        "Ariane 5 rocket from French Guiana, beginning its journey to the L2 point.",
    ),
    (
        "Water boils at 100 degrees Celsius at sea level.",
        "At standard atmospheric pressure (sea level), the boiling point of water is "
        "100 °C (212 °F); the value drops at higher altitude as pressure decreases.",
    ),
    (
        "The Great Wall of China is not visible from the Moon with the naked eye.",
        "Contrary to popular myth, the Great Wall of China cannot be seen from the Moon "
        "with the unaided eye; astronauts have confirmed it is not distinguishable.",
    ),
    (
        "Python is a high-level, general-purpose programming language.",
        "Python is a high-level, general-purpose programming language emphasizing code "
        "readability with significant indentation; it is dynamically typed.",
    ),
    (
        "The 2015 Paris Agreement aims to limit global warming to well below 2 °C.",
        "Under the Paris Agreement, adopted in 2015, parties aim to hold the increase in "
        "global average temperature to well below 2 °C above pre-industrial levels.",
    ),
    (
        "The human heart has four chambers.",
        "The human heart is divided into four chambers: two upper atria and two lower "
        "ventricles, which pump blood through the circulatory system.",
    ),
    (
        "Mount Everest is the highest mountain above sea level on Earth.",
        "Mount Everest, in the Himalayas, is Earth's highest mountain above sea level, "
        "with a summit elevation of about 8,849 metres.",
    ),
    (
        "The COVID-19 vaccines authorized in 2020 used mRNA technology.",
        "The first COVID-19 vaccines authorized in late 2020, from Pfizer-BioNTech and "
        "Moderna, are based on messenger RNA (mRNA) technology.",
    ),
    (
        "Shakespeare wrote the play Hamlet.",
        "Hamlet is a tragedy written by William Shakespeare sometime between 1599 and "
        "1601; it is among his most frequently performed plays.",
    ),
    (
        "The speed of light in a vacuum is about 299,792 kilometres per second.",
        "The speed of light in vacuum, denoted c, is exactly 299,792,458 metres per "
        "second — roughly 299,792 km/s — and is a universal physical constant.",
    ),
    (
        "Brazil is the largest country in South America by area.",
        "Brazil is the largest country in both South America and Latin America, covering "
        "about 8.5 million square kilometres, nearly half the continent's land area.",
    ),
    (
        "Insulin is produced by the pancreas.",
        "Insulin is a hormone produced by the beta cells of the pancreatic islets; it "
        "regulates blood glucose by promoting cellular uptake of glucose.",
    ),
]


def build(seed: int) -> list[dict]:
    rng = random.Random(seed)
    rows: list[dict] = []
    n = len(BASES)
    for i, (sentence, source) in enumerate(BASES):
        # Positive: sentence with its own supporting source.
        rows.append(
            {
                "id": f"cite-pos-{i}",
                "label": "supported",
                "citing_sentence": sentence,
                "source_text": source,
            }
        )
        # Negative: sentence with a different base's (mismatched) source.
        j = rng.choice([k for k in range(n) if k != i])
        rows.append(
            {
                "id": f"cite-neg-{i}",
                "label": "unsupported",
                "citing_sentence": sentence,
                "source_text": BASES[j][1],
            }
        )
    rng.shuffle(rows)
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m training.build_citation_benchmark")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("data/eval/citation_audit_dev.jsonl"))
    args = p.parse_args(argv)

    rows = build(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    n_pos = sum(1 for r in rows if r["label"] == "supported")
    print(f"Wrote {len(rows)} pairs ({n_pos} supported / {len(rows) - n_pos} unsupported) → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
