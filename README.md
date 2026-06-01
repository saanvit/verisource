# VeriSource — Evidence · Triangulation · Reliability

VeriSource is an LLM-powered tool that evaluates how reliable a news article or
written source is. Given a URL or pasted text, it decomposes the article into
atomic claims, checks each against independent web sources, and produces a 0-100
reliability score, a verdict (`highly-reliable` → `unreliable`), and a structured
breakdown of the evidence.

It ships three analysis depths — **Quick** (single-shot whole-article judgment),
**Standard** (per-claim decomposition + retrieval), and **Deep** (adds a
self-critique agent that audits and fixes its own verifications) — plus a
fine-tuned local NLI stance labeler and an isotonic score calibrator.

## How reliability is assessed

Three independent signals are computed and combined:

1. **Domain reputation prior** (`backend/reliability/domain_reputation.py`).
   A curated database (`data/domain_reputation.json`) of well-known domains
   with scores, bias, and source-type metadata sourced from Media Bias/Fact
   Check, Ad Fontes, and AllSides. Falls back to TLD priors (`.gov`, `.edu`,
   `.ac.uk`) and finally a neutral prior.
2. **Content analysis** (`backend/reliability/analyzer.py`). The article is
   sent to the configured **LLM** — Anthropic Claude (via OpenRouter) by
   default, with **Mistral** as a fallback — using a strict-JSON system prompt
   that scores factuality, objectivity, transparency, and
   sensationalism-restraint, and extracts main claims and red flags. If no LLM
   is configured, a deterministic lexical heuristic is used as a fallback.
3. **Cross-reference** (`backend/reliability/cross_reference.py` /
   `backend/reliability/claims.py`). Each claim (or a user-supplied claim) is
   searched on the open web via **Tavily**, excluding the original domain.
   Hits are stance-labeled *supports / contradicts / unclear* — by the LLM, or
   by a fine-tuned local NLI model (`NLI_BACKEND=local`) — and weighted by
   their own domain reputation to produce a corroboration score and a consensus
   label (`strong-support`, `weak-support`, `mixed`, `contradicts`, `no-data`).

The three signals are fused (`backend/reliability/scorer.py`) with **dynamic
weights** — when one channel has weak evidence (unknown domain, no text, no
search hits), its weight is reduced and redistributed.

```
overall = w_d · domain_score + w_c · content_score + w_x · xref_score
defaults: w_d=0.35, w_c=0.40, w_x=0.25
```

## Project layout

```
cs153project/
├── backend/
│   ├── main.py                       FastAPI app + static frontend mount
│   ├── config.py                     env-driven settings
│   ├── models.py                     pydantic schemas
│   ├── reliability/
│   │   ├── content_extractor.py      trafilatura/BS4 extraction
│   │   ├── domain_reputation.py      curated prior lookup
│   │   ├── analyzer.py               LLM content analysis
│   │   ├── llm_client.py             OpenRouter (Claude) → Mistral routing
│   │   ├── claims.py                 atomic-claim decomposition + stance labeling
│   │   ├── pipeline_per_claim.py     per-claim orchestration
│   │   ├── reflection.py             self-critique agent loop
│   │   ├── opinions.py               opinion grounding (premise verification)
│   │   ├── nli.py                    local fine-tuned NLI stance labeler
│   │   ├── cross_reference.py        single-shot corroboration (Quick mode)
│   │   ├── calibration.py            isotonic (PAV) score calibration
│   │   └── scorer.py                 weighted fusion → final report
│   ├── search/
│   │   └── web_search.py             Tavily client
│   └── eval/                         evaluation harness (metrics, datasets, runner)
├── frontend/
│   ├── index.html                    single-page UI
│   ├── styles.css                    VeriSource brand identity
│   ├── app.js                        UI logic + API calls
│   └── reveal.js                     staged-reveal animation orchestrator
├── training/                         NLI fine-tune, calibration fit, benchmark builders
├── data/
│   ├── domain_reputation.json        curated reputation DB
│   └── eval/                         labeled benchmark datasets
├── models/
│   ├── nli/                          fine-tuned DeBERTa checkpoint
│   └── calibration/                  fitted isotonic calibrators
├── tests/                            pytest unit tests
├── requirements.txt
├── requirements-nli.txt
├── .env.example
└── README.md
```

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env and add an LLM key + a search key:
#   OPENROUTER_API_KEY  (preferred; Claude Sonnet via OpenRouter)
#   MISTRAL_API_KEY     (fallback LLM, used only if OpenRouter is unset)
#   TAVILY_API_KEY      (web search / cross-reference)
```

The keys are optional — without them the app still runs, degraded:
* No LLM key (`OPENROUTER_API_KEY` / `MISTRAL_API_KEY`): content analysis falls
  back to a lexical heuristic.
* No `TAVILY_API_KEY`: cross-reference is skipped (its weight goes to 0).

## Run

```bash
python -m backend.main
```

Then open <http://localhost:8000>.

## API

### `GET /api/health`

Returns whether the LLM and search are configured.

### `POST /api/assess`

```json
{
  "url":   "https://www.reuters.com/world/...",
  "text":  null,
  "claim": "Optional specific claim to fact-check"
}
```

Either `url` or `text` is required. The optional `?mode=` query parameter
selects which pipeline to run:

* `mode=per-claim` (default) — **Standard** depth. Atomic-claim decomposition +
  per-claim retrieval and stance labeling. Returns a richer report including
  `content_analysis.claim_verifications` and `content_analysis.coverage`.
* `mode=per-claim-reflective` — **Deep** depth. The per-claim pipeline wrapped
  in a self-critique agent loop that audits and fixes its own verifications;
  adds `content_analysis.reflection_trace`.
* `mode=default` — **Quick** depth. Single-shot analyzer + one cross-reference
  search.

(The frontend surfaces these as the **Quick / Standard / Deep** analysis-depth
toggle.)

Returns a `ReliabilityReport`:

```json
{
  "overall_score": 82.4,
  "verdict": "generally-reliable",
  "confidence": 0.85,
  "domain_reputation":  { "...": "..." },
  "content_analysis":   { "factuality": 86, "...": "..." },
  "cross_reference":    { "consensus": "strong-support", "...": "..." },
  "explanation":        "Overall reliability is 82.4/100 ...",
  "weights":            { "domain": 0.35, "content": 0.40, "cross_reference": 0.25 }
}
```

Example:

```bash
curl -X POST http://localhost:8000/api/assess \
  -H 'Content-Type: application/json' \
  -d '{"url":"https://www.reuters.com/world/europe/..."}'
```

## Tests

```bash
pytest -q
```

The unit tests exercise the domain-reputation lookup and the score-aggregation
logic without requiring API keys. The LLM and Tavily calls are isolated
behind interfaces with deterministic fallbacks for offline use.

## Evaluation

The eval harness (`backend/eval/`) runs the assess pipeline over a labeled
dataset and computes classification + calibration metrics. Use it to
A/B-compare pipeline variants on the same examples.

```bash
# Smoke-test on the bundled dev set (no API keys required).
python -m backend.eval --dataset dev --skip-cross-reference

# Full run including Tavily cross-reference (requires both API keys).
python -m backend.eval --dataset dev

# Per-claim pipeline: decompose into atomic claims, verify each independently.
python -m backend.eval --dataset dev --pipeline per-claim

# Your own labeled set, in the unified JSONL schema.
python -m backend.eval --dataset jsonl --path data/eval/my_corpus.jsonl \
    --concurrency 8 --output my_corpus_results.json
```

### Dataset format

JSONL, one example per line:

```json
{"id": "...", "label": "reliable|unreliable|mixed|satire",
 "text": "...", "url": "https://...", "claim": "optional",
 "true_score": 92.0}
```

`label` is required. At least one of `text`/`url` is required. Common
fake-news labels (`true`, `fake`, `false`, `pants_fire`, etc.) are normalized
automatically. To plug in a public corpus, convert it to this schema:

* **NELA-GT** (Norregaard et al.) — source-level reliability labels via NewsGuard / MBFC.
* **ISOT Fake News** (Kaggle) — binary fake/true article corpus.
* **FakeNewsNet** (Shu et al.) — PolitiFact + GossipCop claim-level labels.
* **AVeriTeC** (Schlichtkrull et al.) — newer evidence-grounded fact-checking benchmark.

### Output

The harness prints a summary table (confusion matrix, precision/recall/F1
per class, macro-F1, ROC-AUC, Brier score, MAE) and writes
`eval_results.json` with per-example predictions plus full `ReliabilityReport`
for downstream analysis.

### Comparing pipeline variants

`run_eval()` accepts any async `pipeline(example) -> ReliabilityReport`
callable. Built-in names: `default`, `per-claim`.

```python
from backend.eval import load_dataset, run_eval, compute_metrics

examples = load_dataset("dev")
baseline  = await run_eval(examples, pipeline_name="default")
candidate = await run_eval(examples, pipeline_name="per-claim")
```

The dev set is curated such that the current baseline scores ≈0.90 accuracy
/ ≈0.99 ROC-AUC offline (domain priors + lexical heuristics only) — this is
the floor; pipeline changes should be measured against the *same* set with
API keys enabled, where there's more headroom on hard examples (raw-text
conspiracy claims, unknown domains).

### Public benchmark: GonzaloA/fake_news

Build a 500-example labeled benchmark from a public HuggingFace dataset
(ISOT-derived, balanced reliable/unreliable):

```bash
pip install datasets
python -m training.build_benchmark --dataset gonzaloa --n 500 \
    --out data/eval/gonzaloa_test_500.jsonl
```

Other dataset shorthands: `pulk17`, `mrm8488`. For arbitrary CSVs:

```bash
python -m training.build_benchmark --csv my_data.csv \
    --text-col body --label-col verdict \
    --label-map '{"true":"reliable","false":"unreliable"}' \
    --n 500 --out data/eval/my_data_500.jsonl
```

### Baseline numbers on GonzaloA/fake_news

These articles ship without URLs, so the domain-reputation channel
contributes no signal — both pipelines are measured on content + xref
alone. The differences between pipelines are therefore informative about
their behavior on raw text, which is also the worst-case for the
production API.

| run                            | n   | accuracy | macro-F1 | ROC-AUC | Brier |
| ------------------------------ | --- | -------- | -------- | ------- | ----- |
| Offline (no LLM, no Tavily)    | 500 | 0.506    | 0.347    | 0.525   | 0.252 |
| Default pipeline (LLM+Tavily)  | 50  | 0.540    | 0.480    | **0.776** | 0.219 |
| Per-claim pipeline (LLM+Tavily)| 50  | 0.520    | 0.429    | 0.603   | 0.249 |

The default pipeline is the strongest current variant on this benchmark.
Per-claim is *more conservative*: its scores cluster in a narrow band
(min 42.8, max 64.6 vs default's 28.7-76.3) because when retrieval can't
verify a claim, the score is pulled toward the 50 prior. On benchmarks
where ground truth reflects *source classification* (ISOT/GonzaloA), the
default pipeline's article-level LLM judgment picks up holistic style
cues that line up with the dataset construction. On benchmarks where
ground truth reflects *claim verifiability* (FEVER, AVeriTeC), per-claim
is expected to win — that's the next experiment to run.

Reproduce:

```bash
# Offline floor.
MISTRAL_API_KEY= TAVILY_API_KEY= python -m backend.eval \
    --dataset jsonl --path data/eval/gonzaloa_test_500.jsonl \
    --pipeline default --skip-cross-reference \
    --concurrency 16 --output baseline_gonzaloa_500_offline.json

# Default pipeline, 50 examples, with API keys.
python -m backend.eval \
    --dataset jsonl --path data/eval/gonzaloa_test_500.jsonl \
    --limit 50 --pipeline default --concurrency 8 \
    --output baseline_gonzaloa_50_default.json

# Per-claim pipeline, 50 examples, with API keys.
python -m backend.eval \
    --dataset jsonl --path data/eval/gonzaloa_test_500.jsonl \
    --limit 50 --pipeline per-claim --concurrency 6 \
    --output baseline_gonzaloa_50_perclaim.json
```

## Per-claim verification pipeline

A second pipeline (`backend/reliability/pipeline_per_claim.py`) replaces the
default's single whole-article LLM judgment with **atomic-claim
decomposition + per-claim retrieval**:

```
article
  └─ analyze_content      (legacy axes: objectivity, transparency, sensationalism)
  └─ decompose_claims     (Mistral → ≤6 atomic, individually verifiable claims)
  └─ for each claim concurrently:
       ├─ Tavily search   (claim-specific query, original domain excluded)
       └─ LLM NLI         (label each hit: supports / contradicts / unclear)
  └─ aggregate
       ├─ factuality      = coverage · mean(claim_score) + (1-coverage)·50
       └─ xref            = union of per-claim evidence (deduped)
  └─ scorer.aggregate     (unchanged) → ReliabilityReport
```

`ReliabilityReport.content_analysis` now exposes `claim_verifications` and
`coverage` so the per-claim breakdown can be surfaced in the frontend or
analyzed downstream. The schema is backwards-compatible — both fields
default to `None` for the default pipeline.

**Cost (per article, N decomposed claims):** 2 + N LLM calls, N Tavily
searches. With N=6 that's ~4× the default pipeline's LLM cost and ~6×
its search cost. Use it where signal quality matters; the eval harness
makes the tradeoff measurable.

**Wiring into the API.** `POST /api/assess` defaults to the per-claim
pipeline; pass `?mode=default` to fall back to the legacy single-shot
pipeline, or `?mode=per-claim-reflective` to run the self-reflective
variant described below. The frontend exposes all three as a Pipeline
toggle.

## Self-reflective verification (per-claim-reflective mode)

The per-claim pipeline labels and aggregates in a single shot — every
choice (which claim to extract, which evidence to retrieve, how to
label each hit) commits in the first pass. Live debugging surfaced
three failure modes that single-shot can't catch:

1. **Mislabeled hits.** NLI labelers (both Mistral and the fine-tuned
   DeBERTa) sometimes mark obviously-supporting evidence as `unclear`
   due to hedging language, domain mismatch, or partial overlap. The
   canonical case: an article claims "SB 17 bars DEI activities", four
   Tavily hits literally restate that, all four labeled `unclear` →
   claim status `unverified`.
2. **Weak retrieval queries.** Generic claim phrasings return zero
   evidence or only tangentially-related hits.
3. **Compound claims that slipped past decomposition.** A single
   sentence packing two assertions gets one `unverified` verdict
   instead of two separable ones.

`per-claim-reflective` wraps the per-claim pipeline in a **Claude
critique loop** (Anthropic Claude Sonnet 4.6 via OpenRouter by default)
that audits the initial verifications and emits structured fixes the
system executes:

```
... (steps 1-3 of per-claim) ...
  └─ run_reflection_loop (≤2 rounds)
       └─ Claude critique  → STRICT JSON {issues, actions:[...]}
       └─ execute actions:
            ├─ relabel_hit(claim, hit_url, new_label)   no LLM cost
            ├─ research_claim(claim, alternative_query) +1 Tavily + 1 NLI
            └─ split_claim(claim, subclaims)            +N Tavily + N NLI
       └─ diff scores → ReflectionRound appended to trace
       └─ converge when actions=[] or no |delta|≥5 or critique fails
  └─ aggregate (unchanged) → ReliabilityReport with reflection_trace
```

The full trace is attached to `ReliabilityReport.content_analysis.reflection_trace`
as a typed `list[ReflectionRound]` so the frontend renders the agent's
critique, issue tags, structured actions, and resulting score deltas
inline above the per-claim breakdown.

**Cost (per article):** 1 extra Claude critique call per round + ≤4
action executions per round, capped at 2 rounds. Worst case ~12 extra
calls per article; typical ~3-5. With OpenRouter promo credits the
incremental cost is negligible.

**When to use it:** demo / writeup / borderline-mixed articles where a
single-shot verdict is unsatisfying. The standard `per-claim` mode is
still the default API mode for speed.

**Tests.** Loop logic is covered by `tests/test_reflection.py` with a
monkeypatched critique LLM — all action types, convergence paths, and
the noop-on-failure path are exercised offline.

## Score calibration

Raw `overall_score` from either pipeline is informative but not
*calibrated* — a score of 70 doesn't reliably mean "70% chance the
article is reliable." `backend/reliability/calibration.py` implements
isotonic regression via Pool-Adjacent-Violators (no sklearn at runtime)
and is wired into the scorer as an optional post-processor.

### Fit a calibrator from any eval result

```bash
python -m training.fit_calibration \
    --predictions baseline_gonzaloa_50_default.json \
    --out models/calibration/default_gonzaloa.json \
    --train-frac 0.7
```

The CLI splits the predictions train/test, fits PAV on train, and prints
before/after metrics on test. Calibrators are saved as small JSON files
(thresholds + monotone-fitted scores).

### Apply at inference

```bash
export CALIBRATION_PATH=$PWD/models/calibration/default_gonzaloa.json
python -m backend.main          # API now serves calibrated scores
```

Or for A/B in the eval harness:

```bash
python -m backend.eval --dataset jsonl --path data/eval/gonzaloa_test_500.jsonl \
    --pipeline default --calibration models/calibration/default_gonzaloa.json
```

### Measured impact on the test split (15 held-out from GonzaloA/fake_news, 50 with-LLM examples)

| pipeline + calibration | accuracy | macro-F1 | Brier  | ROC-AUC |
| ---------------------- | -------- | -------- | ------ | ------- |
| Default, raw           | 0.467    | 0.400    | 0.229  | 0.824   |
| Default, calibrated    | **0.800**| **0.762**| **0.167** | 0.778   |
| Per-claim, raw         | 0.467    | 0.400    | 0.259  | 0.639   |
| Per-claim, calibrated  | 0.600    | 0.489    | 0.262  | 0.537   |

Two things to read off this:

* **Default is badly threshold-miscalibrated, then sharply fixed.** Raw
  scores cluster around 50-65; threshold-at-50 metrics underperform
  ranking metrics (AUC 0.82). Calibration lines up the threshold with
  actual reliability and accuracy jumps 33 points without sacrificing
  ranking.
* **Per-claim is already well-calibrated** — Brier barely moves (0.259 →
  0.262) because there isn't much miscalibration to fix. The pipeline is
  intentionally conservative on raw text without URLs (claims hard to
  directly verify → scores pulled to the 50 prior). Calibration sharpens
  thresholds modestly but cannot manufacture ranking signal.

The headline takeaway for the writeup: **per-claim trades raw
classification performance on ISOT-style benchmarks for calibrated
uncertainty — appropriate where false confidence is costly.** Default +
calibration is the strongest *measured* configuration on this benchmark.
Per-claim's expected wins are on claim-verifiability benchmarks (FEVER
article-level, AVeriTeC); that's the next experiment.

Calibrators are pipeline-specific — a calibrator fit on one pipeline's
outputs is not portable to another (or to a sufficiently different
score range like the offline-floor heuristic). Fit one per pipeline.

## Local NLI stance labeler (fine-tuned)

The per-claim pipeline labels each retrieved hit as
`supports / contradicts / unclear`. By default this goes through Mistral
(one LLM call per claim). Set `NLI_BACKEND=local` to route those labels
through a local NLI model instead — much cheaper at inference time and,
once fine-tuned on FEVER + VitaminC, more accurate on the long-tail
news-fact-checking cases that drive most of the remaining error.

### Setup

Install the heavier ML stack (optional — only needed for NLI training or
inference):

```bash
pip install -r requirements-nli.txt
```

Out of the box, with no extra setup, the inference module loads a public
DeBERTa-v3-large NLI checkpoint:

```bash
export NLI_BACKEND=local
python -m backend.eval --dataset dev --pipeline per-claim
```

### Fine-tuning

Build the unified FEVER + VitaminC dataset (one-time, ~5 minutes for the
download):

```bash
python -m training.datasets --out training/data
```

Fine-tune on one GPU:

```bash
python -m training.train_nli \
    --train  training/data/train.jsonl \
    --val    training/data/val.jsonl \
    --model  microsoft/deberta-v3-large \
    --output models/nli/deberta-v3-large-fever-vitc \
    --epochs 2 --batch-size 16 --lr 1e-5 --bf16
```

Expected runtime: ≈3-4 h on one A100 80GB. Use
`--model microsoft/deberta-v3-base` for ~5× speedup at ~2-3% accuracy
cost. After training, point the inference module at the checkpoint:

```bash
export NLI_BACKEND=local
export NLI_MODEL=$PWD/models/nli/deberta-v3-large-fever-vitc
```

### A/B-compare against the LLM labeler

The eval CLI accepts an `--nli` flag that sets the backend for the
duration of the run, so the harness produces side-by-side numbers:

```bash
python -m backend.eval --dataset jsonl --path eval.jsonl \
    --pipeline per-claim --nli llm   --output eval_llm.json
python -m backend.eval --dataset jsonl --path eval.jsonl \
    --pipeline per-claim --nli local --output eval_local.json
```

### Cost model

| backend | per-(claim,hit) cost     | latency (4 hits, 1 claim) |
| ------- | ------------------------ | ------------------------- |
| `llm`   | 1 Mistral call per claim | ~2-5 s                    |
| `local` | 1 forward pass per hit   | ~50 ms on A100, ~250 ms on CPU |

For a typical article with 5 claims × 4 evidence each, `local` reduces
the per-article LLM call count from ≈ 5 stance calls to 0, leaving only
the 1 decomposition call to the API.

## Compute requirements

* **Serving needs no GPU.** With the default LLM stance backend, all heavy
  lifting runs on hosted APIs (OpenRouter/Claude, Mistral, Tavily) and the local
  server is I/O-bound. Running the local NLI labeler (`NLI_BACKEND=local`) works
  on CPU (~250 ms/hit) and is much faster on GPU (~50 ms/hit).
* **Fine-tuning the NLI model uses a GPU** — the shipped checkpoint is a
  DeBERTa-v3 fine-tuned on FEVER + VitaminC (see *Local NLI stance labeler*).
* **Memory:** the FastAPI process uses well under 200 MB (LLM backend).
* **Latency:** a typical assessment takes ~10 s (Quick) to ~30 s (Deep)
  end-to-end.

## Limitations

* The curated domain database is small and US-news-leaning. Adding entries
  is straightforward — see `data/domain_reputation.json`.
* The LLM can be wrong about whether a hit "supports" or "contradicts" a
  claim, especially for nuanced policy/scientific claims. Use the
  per-source labels as hints, not ground truth.
* No author or publication-date verification is performed.

## AI usage disclosure

Per the course AI policy, this section documents where and how AI tools were
used in this project.

**AI as a runtime component (the product itself).** VeriSource's analysis
pipeline calls large language models at inference time: claim decomposition,
content analysis, stance labeling (in the default `llm` backend), opinion
extraction, adversarial-query generation, and the self-critique agent all run
on **Anthropic Claude** (via OpenRouter; `anthropic/claude-opus-4.8` by
default) with **Mistral** as a fallback. Web evidence retrieval uses the
**Tavily** search API. These are core to how the system works and are
documented throughout this README.

**AI as a development tool.** AI coding assistants (**Claude Code** /
Anthropic Claude) were used during development to help implement and refactor
parts of the backend pipeline, the frontend UI, the evaluation harness, and
this documentation. All architectural decisions, the scoring/fusion design,
the experimental setup, and the final review of generated code were directed
and verified by the author. <!-- TODO(author): adjust this paragraph to
accurately reflect your own usage — e.g. which parts you wrote by hand vs.
with assistance. -->

No part of the evaluation numbers reported here was generated or fabricated by
an LLM — all metrics come from running the eval harness (`backend/eval/`) over
labeled data.

## Citations, data sources & acknowledgements

This project builds on the following external data, models, and libraries.

**Domain-reputation data** (`data/domain_reputation.json`) is curated from:
* Media Bias/Fact Check — <https://mediabiasfactcheck.com/>
* Ad Fontes Media (Media Bias Chart) — <https://adfontesmedia.com/>
* AllSides Media Bias Ratings — <https://www.allsides.com/media-bias>

**Models:**
* `microsoft/deberta-v3-base` — He et al., *DeBERTaV3* (2021), the base model
  fine-tuned for the local NLI stance labeler.
* `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` — the off-the-shelf
  NLI checkpoint used when no fine-tuned `NLI_MODEL` is set.
* Anthropic Claude (Opus/Sonnet) and Mistral — hosted LLMs used at inference.

**Datasets** (NLI fine-tuning + evaluation):
* **FEVER** — Thorne et al., *FEVER: a Large-scale Dataset for Fact Extraction
  and VERification* (NAACL 2018).
* **VitaminC** — Schuster et al., *Get Your Vitamin C! Robust Fact Verification
  with Contrastive Evidence* (NAACL 2021).
* **GonzaloA/fake_news** (HuggingFace; ISOT-derived) — used as a public
  source-classification benchmark.

**Key libraries:** FastAPI, Hugging Face `transformers`, `trafilatura`,
scikit-learn (isotonic regression reference), Tavily SDK. See
`requirements.txt` / `requirements-nli.txt` for the full list.

This project was built from scratch for CS 153; it does not fork or
substantially borrow code from an existing repository. <!-- TODO(author): if
you did start from any template/repo, cite it here with a description of your
changes. -->
