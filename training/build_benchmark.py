"""Build a real-world labeled benchmark for the eval harness.

Downloads a public fake-news dataset from the Hugging Face Hub (or a
local CSV), stratified-samples ``--n`` examples per label, and writes
the unified ``data/eval/`` JSONL schema so the existing
``backend.eval`` CLI can consume it.

Quick start:

    pip install datasets       # ~250MB
    python -m training.build_benchmark --dataset gonzaloa --n 500 \
        --out data/eval/fake_news_500.jsonl

    python -m backend.eval --dataset jsonl \
        --path data/eval/fake_news_500.jsonl \
        --pipeline default --skip-cross-reference \
        --concurrency 8 --output baseline_fake_news_500.json

Currently supported HF datasets:

* ``gonzaloa``  — ``GonzaloA/fake_news`` (ISOT-derived, ~24K train + 8K
  test + 8K val; balanced; real news primarily Reuters-syndicated).
  Defaults to the ``test`` split so you're not measuring on data the
  upstream task trained on.
* ``pulk17``    — ``Pulk17/Fake-News-Detection-dataset`` (ISOT-derived,
  ~30K, balanced).
* ``mrm8488``   — ``mrm8488/fake-news`` (~45K, balanced; no title col).

Generic CSV mode:

    python -m training.build_benchmark --csv my_data.csv \
        --text-col body --label-col verdict \
        --label-map '{"true":"reliable","false":"unreliable","mixture":"mixed"}' \
        --n 500 --out data/eval/my_data_500.jsonl

Known limitations of ISOT-derived datasets: real-news articles in ISOT
were scraped from Reuters and tend to begin with "(Reuters) -" while
fake articles came from less consistent sources. Classifiers can pick
up on these stylistic artifacts rather than the underlying claims, so
treat raw accuracy numbers on these benchmarks as upper bounds; for a
defensible writeup, also report numbers on at least one non-ISOT
dataset (NELA-GT, AVeriTeC, etc.).
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path


DEFAULT_MAX_TEXT_CHARS = 3500


def _norm_label_01(value) -> str | None:
    """Most fake-news datasets use 1=real, 0=fake. Map to our labels."""
    try:
        v = int(value)
    except (TypeError, ValueError):
        return None
    if v == 1:
        return "reliable"
    if v == 0:
        return "unreliable"
    return None


def _record(
    id_: str,
    text: str,
    label: str,
    *,
    title: str | None = None,
    url: str | None = None,
    source: str | None = None,
    max_chars: int = DEFAULT_MAX_TEXT_CHARS,
) -> dict | None:
    text = (text or "").strip()
    if len(text) < 80:
        return None  # too short to assess
    text = text[:max_chars]
    rec: dict = {"id": id_, "label": label, "text": text}
    if url:
        rec["url"] = url
    meta: dict = {}
    if title:
        meta["title"] = title[:280]
    if source:
        meta["source"] = source
    if meta:
        rec["meta"] = meta
    return rec


# ---------- HF dataset loaders ----------


def _load_gonzaloa(split: str, max_chars: int) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("GonzaloA/fake_news", split=split)
    out: list[dict] = []
    for i, row in enumerate(ds):
        label = _norm_label_01(row.get("label"))
        if not label:
            continue
        rec = _record(
            id_=f"gonzaloa-{split}-{i}",
            text=row.get("text") or "",
            label=label,
            title=row.get("title"),
            source="GonzaloA/fake_news",
            max_chars=max_chars,
        )
        if rec:
            out.append(rec)
    return out


def _load_pulk17(split: str, max_chars: int) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("Pulk17/Fake-News-Detection-dataset", split=split)
    out: list[dict] = []
    for i, row in enumerate(ds):
        label = _norm_label_01(row.get("label"))
        if not label:
            continue
        rec = _record(
            id_=f"pulk17-{split}-{i}",
            text=row.get("text") or "",
            label=label,
            title=row.get("title"),
            source="Pulk17/Fake-News-Detection-dataset",
            max_chars=max_chars,
        )
        if rec:
            out.append(rec)
    return out


def _load_mrm8488(split: str, max_chars: int) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("mrm8488/fake-news", split=split)
    out: list[dict] = []
    for i, row in enumerate(ds):
        label = _norm_label_01(row.get("label"))
        if not label:
            continue
        rec = _record(
            id_=f"mrm8488-{split}-{i}",
            text=row.get("text") or "",
            label=label,
            source="mrm8488/fake-news",
            max_chars=max_chars,
        )
        if rec:
            out.append(rec)
    return out


HF_LOADERS = {
    "gonzaloa": (_load_gonzaloa, "test"),
    "pulk17": (_load_pulk17, "train"),
    "mrm8488": (_load_mrm8488, "train"),
}


# ---------- Generic CSV loader ----------


def _load_csv(
    path: Path, text_col: str, label_col: str, label_map: dict[str, str], max_chars: int
) -> list[dict]:
    import csv

    out: list[dict] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            raw = (row.get(label_col) or "").strip()
            mapped = label_map.get(raw) or label_map.get(raw.lower())
            if not mapped:
                continue
            rec = _record(
                id_=f"csv-{path.stem}-{i}",
                text=row.get(text_col) or "",
                label=mapped,
                title=row.get("title"),
                url=row.get("url"),
                source=str(path.name),
                max_chars=max_chars,
            )
            if rec:
                out.append(rec)
    return out


# ---------- Sampling ----------


def stratified_sample(records: list[dict], n: int, seed: int) -> list[dict]:
    """Return ~n records, ~equally split across observed labels.

    Examples are sampled without replacement within each label bucket.
    """
    rng = random.Random(seed)
    by_label: dict[str, list[dict]] = {}
    for r in records:
        by_label.setdefault(r["label"], []).append(r)

    n_labels = len(by_label)
    if n_labels == 0:
        return []
    per_label = max(1, n // n_labels)

    out: list[dict] = []
    for label, bucket in by_label.items():
        rng.shuffle(bucket)
        out.extend(bucket[:per_label])
    rng.shuffle(out)
    return out


# ---------- CLI ----------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m training.build_benchmark")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--dataset",
        choices=list(HF_LOADERS),
        help="HF Hub dataset shorthand (downloads via datasets lib).",
    )
    src.add_argument("--csv", type=Path, help="Path to a CSV with text + label columns.")

    p.add_argument("--split", default=None, help="HF split (default: dataset-specific test/train).")
    p.add_argument("--text-col", default="text", help="CSV mode: text column name.")
    p.add_argument("--label-col", default="label", help="CSV mode: label column name.")
    p.add_argument(
        "--label-map",
        default=None,
        help='CSV mode: JSON dict mapping CSV labels to {reliable,unreliable,mixed}. '
        'E.g. \'{"true":"reliable","false":"unreliable"}\'.',
    )
    p.add_argument("--n", type=int, default=500, help="Approx total examples (split across labels).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--max-text-chars",
        type=int,
        default=DEFAULT_MAX_TEXT_CHARS,
        help="Truncate article text to this many chars. Default 3500.",
    )
    p.add_argument("--out", type=Path, required=True, help="Output JSONL path.")
    args = p.parse_args(argv)

    if args.csv:
        if not args.label_map:
            raise SystemExit(
                "--label-map is required when using --csv "
                "(e.g. '{\"true\":\"reliable\",\"false\":\"unreliable\"}')"
            )
        try:
            label_map = json.loads(args.label_map)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"--label-map must be valid JSON: {exc}") from exc
        records = _load_csv(
            args.csv, args.text_col, args.label_col, label_map, args.max_text_chars
        )
    else:
        loader, default_split = HF_LOADERS[args.dataset]
        split = args.split or default_split
        print(f"Loading HF dataset shorthand={args.dataset!r} split={split!r}...")
        records = loader(split, args.max_text_chars)

    print(f"Loaded {len(records)} usable records (after length filter).")
    sampled = stratified_sample(records, args.n, args.seed)
    counts = Counter(r["label"] for r in sampled)
    print(f"Sampled {len(sampled)} → {dict(counts)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in sampled:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
