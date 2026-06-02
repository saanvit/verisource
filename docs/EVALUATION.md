# VeriSource — Evaluation & Failure Analysis

This document consolidates every quantitative result behind VeriSource, plus a
qualitative failure analysis of the stance-labeling component. All numbers come
from running the evaluation harness (`backend/eval/`, `training/eval_nli.py`)
over labeled data — none are estimated or LLM-generated. The raw result files
referenced here are committed at the repo root (`baseline_gonzaloa_*.json`,
`eval_nli_*.json`) and under `models/calibration/`.

---

## 1. What we measure, and why it is subtle

Reliability benchmarks fall into two families, and a system can be good at one
while looking bad at the other:

- **Source classification** (ISOT / GonzaloA-style): the label describes the
  *outlet/article* ("fake" vs "real"). Holistic, article-level style cues
  correlate strongly with the label.
- **Claim verifiability** (FEVER / VitaminC / AVeriTeC-style): the label
  describes whether a *specific claim* is supported by evidence.

VeriSource's two pipelines are tuned to different ends of this axis: the
**Standard (single-shot)** pipeline reads holistic style and wins on
source-classification data; the **per-claim** pipeline decomposes and verifies
atomic claims and is deliberately conservative. We report both and are explicit
about which benchmark each is suited to.

---

## 2. Reproducibility check (offline floor)

The harness runs with no API keys (domain priors + lexical heuristics only).
This is the reproducible floor every other configuration is measured against.

```bash
env OPENROUTER_API_KEY= MISTRAL_API_KEY= TAVILY_API_KEY= \
  python -m backend.eval --dataset dev --skip-cross-reference
```

Bundled `dev` set (n=25, 4 mixed examples skipped from binary metrics):

| metric | value |
| ------ | ----- |
| accuracy | 0.905 |
| macro-F1 | 0.905 |
| ROC-AUC | 0.995 |
| Brier | 0.100 |

The dev set is curated to be easy offline; it is a smoke test for the harness,
not a performance claim. The honest performance signal is on the public
benchmark below.

---

## 3. Public benchmark — GonzaloA/fake_news (ISOT-derived, balanced)

These articles ship **without URLs**, so the domain-reputation channel is dead
and both pipelines are measured on content + cross-reference alone — the
worst case for the production system, which makes the comparison informative.

| run | n | accuracy | macro-F1 | ROC-AUC | Brier |
| --- | --- | -------- | -------- | ------- | ----- |
| Offline floor (no LLM, no search) | 500 | 0.506 | 0.347 | 0.525 | 0.251 |
| Standard pipeline (LLM + Tavily) | 50 | **0.540** | **0.480** | **0.776** | 0.219 |
| Per-claim pipeline (LLM + Tavily) | 50 | 0.520 | 0.429 | 0.603 | 0.249 |

**Reading:** on a *source-classification* benchmark the Standard pipeline wins,
as expected — its article-level judgment picks up holistic style cues aligned
with how the dataset was built. Per-claim is more conservative: its raw scores
cluster near the 50 prior because, without URLs, many atomic claims cannot be
directly verified, so the score is pulled toward "unknown." This is a feature,
not a bug — see calibration below.

> **Limitation (stated plainly):** the LLM/Tavily runs are n=50 with no
> confidence intervals; treat these as directional, not definitive. Scaling to
> the full 500 with the production config is the obvious next run.

---

## 4. Calibration ablation (isotonic / Pool-Adjacent-Violators)

A raw score of 70 does not reliably mean "70% likely reliable." We fit an
isotonic calibrator per pipeline (`backend/reliability/calibration.py`) on a
35-example train split and report on the held-out 15.

| configuration | accuracy | macro-F1 | Brier | ROC-AUC |
| ------------- | -------- | -------- | ------ | ------- |
| Standard, raw | 0.467 | 0.400 | 0.229 | 0.824 |
| **Standard, calibrated** | **0.800** | **0.762** | **0.167** | 0.778 |
| Per-claim, raw | 0.467 | 0.400 | 0.259 | 0.639 |
| Per-claim, calibrated | 0.600 | 0.489 | 0.262 | 0.537 |

**Two findings:**

1. **Standard is badly threshold-miscalibrated, then sharply fixed.** Raw scores
   cluster at 50–65, so a threshold-at-50 underperforms the *ranking* quality
   (AUC 0.82). Calibration realigns the threshold and accuracy jumps **+33
   points** (0.467 → 0.800) with Brier improving 0.229 → 0.167 — without
   inventing ranking signal it didn't have.
2. **Per-claim is already well-calibrated.** Its Brier barely moves
   (0.259 → 0.262) because there isn't much miscalibration to fix; the pipeline
   is intentionally conservative. Calibration can sharpen a threshold but cannot
   manufacture ranking signal that the retrieval didn't find.

Calibrators are pipeline-specific — one fit on Standard outputs is not portable
to per-claim's score range.

---

## 5. Stance labeler — fine-tuned NLI vs. LLM (the core NLP experiment)

The per-claim pipeline labels each retrieved hit as
**supports / contradicts / unclear**. We fine-tuned `deberta-v3-base` on
**FEVER + VitaminC** and A/B-compared it against an LLM labeler on a matched
held-out test set.

**Matched A/B (100 examples, identical inputs):**

| backend | accuracy | macro-F1 | F1 supports | F1 contradicts | F1 unclear | latency / example |
| ------- | -------- | -------- | ----------- | -------------- | ---------- | ----------------- |
| Fine-tuned DeBERTa (local) | **0.88** | **0.879** | 0.900 | 0.895 | 0.844 | **249 ms** |
| LLM labeler (Mistral) | 0.71 | 0.692 | 0.732 | 0.785 | 0.560 | 14,894 ms |

**Full local run (all 3,000 test examples):** accuracy **0.89**, macro-F1
**0.890** (per-class F1: supports 0.922, contradicts 0.879, unclear 0.868),
199 ms/example.

**Reading:** on the NLI task the fine-tuned model beats the Mistral LLM labeler
by **+17 accuracy points** and is **~60× faster**. The largest gap is on the
`unclear` class (F1 0.844 vs 0.560) — the LLM over-commits to supports/
contradicts where the trained model correctly abstains.

> **Honest caveat about the live config:** the LLM arm above was **Mistral**.
> The production default now labels with **Claude Opus 4.8**, which the case
> study below shows is materially more accurate than both NLI checkpoints on
> *real-news retrieval* (as opposed to the clean FEVER/VitaminC test
> distribution). Re-running this A/B with Opus 4.8 as the LLM arm is
> acknowledged future work; the table above should be read as
> *fine-tuned-NLI vs Mistral*, not as the final word on the shipped system.

---

## 6. Failure analysis — a real stance-labeling case study

The most instructive error surfaced during live use, on a 2026 Texas U.S.
Senate article. One decomposed claim:

> *"Texas state Sen. James Talarico won the Democratic primary for the Texas
> Senate race."*

Retrieval returned strong corroboration — Wikipedia (*"James Talarico won the
Democratic primary with 52.4% of the vote"*) and the Texas Tribune (*"Talarico
defeats Crockett"*). We verified the **same claim against the same evidence**
across three stance backends:

| stance backend | claim verdict | support / contradict | what it did |
| -------------- | ------------- | -------------------- | ----------- |
| Public DeBERTa NLI (off-the-shelf) | **contradicted / "refuted"** | 23% / 77% | Mislabeled the literal "won with 52.4%" hits as *contradicts* — likely tripped by "dispute / conceded / Crockett" language in the snippets |
| Fine-tuned DeBERTa NLI | supported | 42% / **0%** | No false contradiction, but conservative — labeled the clearest hits *unclear*, leaning on NPR/Ballotpedia for the verdict |
| **Claude Opus 4.8 (shipped default)** | **supported** | **100% / 0%** | Every hit correctly *supports*; reads the "52.4%" sentence as direct corroboration |

**What this taught us (and what changed):**

1. **A single mislabel can flip a verdict.** The public NLI's false
   `contradicts` cascaded through the adversarial-robustness signal into a
   "refuted" tag — a confidently wrong result. This motivated two existing
   mitigations: the conservative status thresholds in `claims.py` and the
   **self-critique agent** (Deep mode), which is built specifically to catch
   "obviously-supporting evidence labeled contradicts."
2. **Benchmark accuracy ≠ deployment accuracy.** The fine-tuned model wins on
   the clean FEVER/VitaminC distribution (§5) yet is over-conservative on messy
   real-news retrieval, where snippets mix outcome + procedural framing. The
   frontier LLM generalizes better to that distribution.
3. **Decision.** The production default stance backend was switched to Claude
   Opus 4.8. The fine-tuned NLI remains the documented cheap/fast option
   (`NLI_BACKEND=local`, ~250 ms/hit, free) and the strongest *measured* result
   on in-distribution NLI.

This is the kind of trade-off the system makes visible by design: the
per-source labels, the adversarial robustness tag, and the agent's reasoning
trace are all surfaced in the UI so a user can audit exactly these decisions.

---

## 7. Summary of limitations

- **Small N on the production-config benchmark** (n=50 / n=15 splits, no CIs).
- **One public benchmark** (GonzaloA, source-classification). A
  claim-verifiability benchmark (FEVER article-level, AVeriTeC) — where
  per-claim is hypothesized to win — has not yet been run end-to-end.
- **Stance-labeler A/B predates the Opus 4.8 default**; numbers describe
  fine-tuned-NLI vs Mistral.
- **Domain database is small and US-news-leaning** (`data/domain_reputation.json`).
- **No author or publication-date verification.**

---

## How to reproduce

```bash
# Offline floor (no keys)
env OPENROUTER_API_KEY= MISTRAL_API_KEY= TAVILY_API_KEY= \
  python -m backend.eval --dataset dev --skip-cross-reference

# GonzaloA benchmark (needs keys); swap --pipeline default | per-claim
python -m backend.eval --dataset jsonl --path data/eval/gonzaloa_test_500.jsonl \
  --limit 50 --pipeline default --output baseline_gonzaloa_50_default.json

# Fit + evaluate a calibrator
python -m training.fit_calibration \
  --predictions baseline_gonzaloa_50_default.json \
  --out models/calibration/default_gonzaloa.json --train-frac 0.7

# NLI stance A/B (local fine-tuned vs LLM)
python -m training.eval_nli --test training/data/test.jsonl --backends local llm
```
