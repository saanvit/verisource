"""Dataset loading for the eval harness.

A unified ``EvalExample`` schema lets us mix the bundled dev set with any
labeled corpus the user drops in as JSONL (NELA-GT samples, ISOT,
FakeNewsNet, etc.). Each line:

    {"id": "...", "label": "reliable|unreliable|mixed|satire|...",
     "text": "...", "url": "https://...", "claim": "optional",
     "true_score": 92.0}

``label`` is required. Everything else is optional but at least one of
``text`` or ``url`` must be present.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from backend.config import DATA_DIR


@dataclass
class EvalExample:
    id: str
    label: str
    text: str | None = None
    url: str | None = None
    claim: str | None = None
    true_score: float | None = None
    meta: dict | None = None

    def __post_init__(self) -> None:
        if not self.text and not self.url:
            raise ValueError(f"Example {self.id}: needs at least one of text/url")


def load_jsonl(path: Path | str) -> list[EvalExample]:
    path = Path(path)
    examples: list[EvalExample] = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if "label" not in obj:
                raise ValueError(f"{path}:{line_no}: missing required field 'label'")
            examples.append(
                EvalExample(
                    id=str(obj.get("id") or f"{path.stem}-{line_no}"),
                    label=str(obj["label"]),
                    text=obj.get("text"),
                    url=obj.get("url"),
                    claim=obj.get("claim"),
                    true_score=(
                        float(obj["true_score"]) if obj.get("true_score") is not None else None
                    ),
                    meta=obj.get("meta"),
                )
            )
    if not examples:
        raise ValueError(f"{path}: no examples found")
    return examples


def load_dev() -> list[EvalExample]:
    """Load the small bundled dev set used for smoke-testing the harness."""
    return load_jsonl(DATA_DIR / "eval" / "dev.jsonl")


def load_dataset(name: str, path: Path | str | None = None) -> list[EvalExample]:
    """Dispatch on a friendly dataset name.

    ``dev`` loads the bundled set. ``jsonl`` requires an explicit path and
    accepts the generic schema described in the module docstring — this is
    the path to use for NELA-GT samples, ISOT, FakeNewsNet, etc., once
    converted into the unified JSONL.
    """
    if name == "dev":
        return load_dev()
    if name == "jsonl":
        if not path:
            raise ValueError("--path is required for --dataset jsonl")
        return load_jsonl(path)
    raise ValueError(f"unknown dataset: {name!r} (expected 'dev' or 'jsonl')")
