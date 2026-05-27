"""Fine-tune a DeBERTa-v3 (or compatible) NLI model on the unified
FEVER + VitaminC (+ SciFact) corpus produced by ``training/datasets.py``.

Run on a GPU box:

    pip install -r requirements-nli.txt
    python -m training.datasets --out training/data        # one-time, ~5 min
    python -m training.train_nli \
        --train training/data/train.jsonl \
        --val   training/data/val.jsonl \
        --model microsoft/deberta-v3-large \
        --output models/nli/deberta-v3-large-fever-vitc \
        --epochs 2 --batch-size 16 --lr 1e-5 --bf16

Expected runtime: ~3-4 hours on one A100 80GB. Drop to ``--model
microsoft/deberta-v3-base`` for a ~5x speedup at ~2-3% accuracy cost.

After training, point the inference module at the checkpoint:

    export NLI_MODEL=$(pwd)/models/nli/deberta-v3-large-fever-vitc

The model is loaded as a 3-class sequence classifier with the label
mapping defined in ``training.datasets`` (0=supports, 1=contradicts,
2=unclear).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.datasets import LABEL_NAMES


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _build_dataset(rows: list[dict]):
    from datasets import Dataset

    return Dataset.from_list(
        [
            {
                "premise": r["premise"],
                "hypothesis": r["hypothesis"],
                "labels": int(r["label"]),
            }
            for r in rows
        ]
    )


def _tokenize_fn(tokenizer, max_length: int):
    def _fn(batch):
        return tokenizer(
            batch["premise"],
            batch["hypothesis"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )

    return _fn


def _compute_metrics(eval_pred):
    import numpy as np
    from sklearn.metrics import f1_score, precision_score, recall_score

    preds = np.argmax(eval_pred.predictions, axis=-1)
    refs = eval_pred.label_ids
    out = {
        "accuracy": float((preds == refs).mean()),
        "macro_f1": float(f1_score(refs, preds, average="macro", zero_division=0)),
    }
    for cls, name in LABEL_NAMES.items():
        out[f"precision_{name}"] = float(
            precision_score(refs, preds, labels=[cls], average="macro", zero_division=0)
        )
        out[f"recall_{name}"] = float(
            recall_score(refs, preds, labels=[cls], average="macro", zero_division=0)
        )
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--val", type=Path, required=True)
    p.add_argument(
        "--model", default="microsoft/deberta-v3-large",
        help="HF model id or local path. Recommended: microsoft/deberta-v3-large.",
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--warmup-ratio", type=float, default=0.06)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument(
        "--bf16", action="store_true",
        help="Use bfloat16 mixed precision (recommended on A100/H100/MPS).",
    )
    p.add_argument("--fp16", action="store_true", help="Use fp16 mixed precision.")
    p.add_argument("--gradient-accumulation", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--logging-steps", type=int, default=100)
    p.add_argument("--save-steps", type=int, default=2000)
    p.add_argument("--eval-steps", type=int, default=2000)
    args = p.parse_args(argv)

    # Heavy imports happen here so this file can be imported without torch.
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    set_seed(args.seed)

    print(f"Loading model {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    id2label = {i: name for i, name in LABEL_NAMES.items()}
    label2id = {v: k for k, v in id2label.items()}
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=len(LABEL_NAMES),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    print(f"Loading datasets from {args.train} and {args.val}...")
    train_rows = _load_jsonl(args.train)
    val_rows = _load_jsonl(args.val)
    print(f"  train: {len(train_rows):>8d} rows")
    print(f"  val:   {len(val_rows):>8d} rows")

    train_ds = _build_dataset(train_rows)
    val_ds = _build_dataset(val_rows)

    tok_fn = _tokenize_fn(tokenizer, args.max_length)
    train_ds = train_ds.map(tok_fn, batched=True, remove_columns=["premise", "hypothesis"])
    val_ds = val_ds.map(tok_fn, batched=True, remove_columns=["premise", "hypothesis"])

    training_args = TrainingArguments(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        bf16=args.bf16,
        fp16=args.fp16,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        report_to=[],
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=_compute_metrics,
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving final model to {args.output}...")
    trainer.save_model(str(args.output))
    tokenizer.save_pretrained(str(args.output))

    final = trainer.evaluate()
    print("Final eval:")
    for k, v in final.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
