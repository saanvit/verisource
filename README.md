# Source Reliability Assessor

An LLM-powered tool that evaluates how reliable a news article or written
source is. Given a URL or pasted text, it produces a 0-100 reliability score,
a verdict (`highly-reliable` → `unreliable`), and a structured breakdown of
the evidence.

## How reliability is assessed

Three independent signals are computed and combined:

1. **Domain reputation prior** (`backend/reliability/domain_reputation.py`).
   A curated database (`data/domain_reputation.json`) of well-known domains
   with scores, bias, and source-type metadata sourced from Media Bias/Fact
   Check, Ad Fontes, and AllSides. Falls back to TLD priors (`.gov`, `.edu`,
   `.ac.uk`) and finally a neutral prior.
2. **Content analysis** (`backend/reliability/analyzer.py`). The article is
   sent to **Mistral** with a strict-JSON system prompt that scores
   factuality, objectivity, transparency, and sensationalism-restraint, and
   extracts main claims and red flags. If the LLM is unavailable, a
   deterministic lexical heuristic is used as a fallback.
3. **Cross-reference** (`backend/reliability/cross_reference.py`). The
   article's main claim (or a user-supplied claim) is searched on the open
   web via **Tavily**, excluding the original domain. The LLM labels each
   independent hit as *supports / contradicts / unclear*; hits are weighted
   by their own domain reputation to produce a corroboration score and a
   consensus label (`strong-support`, `weak-support`, `mixed`,
   `contradicts`, `no-data`).

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
│   │   ├── analyzer.py               Mistral content analysis
│   │   ├── cross_reference.py        Tavily + LLM corroboration
│   │   └── scorer.py                 weighted fusion → final report
│   └── search/
│       └── web_search.py             Tavily client
├── frontend/
│   ├── index.html                    single-page UI
│   ├── styles.css
│   └── app.js
├── data/
│   └── domain_reputation.json        curated reputation DB
├── tests/                            pytest unit tests
├── requirements.txt
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
# Edit .env and add MISTRAL_API_KEY and TAVILY_API_KEY
```

Both keys are optional — without them the app still runs:
* No `MISTRAL_API_KEY`: content analysis falls back to a lexical heuristic.
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

Either `url` or `text` is required. Returns a `ReliabilityReport`:

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

## Compute requirements

* **No GPU required.** All heavy lifting runs on hosted APIs (Mistral,
  Tavily). The local server is I/O-bound.
* **Memory:** the FastAPI process uses well under 200 MB.
* **Latency:** a typical assessment takes 10-30 seconds end-to-end (one
  article fetch + 1-2 LLM calls + one search call).

## Limitations

* The curated domain database is small and US-news-leaning. Adding entries
  is straightforward — see `data/domain_reputation.json`.
* The LLM can be wrong about whether a hit "supports" or "contradicts" a
  claim, especially for nuanced policy/scientific claims. Use the
  per-source labels as hints, not ground truth.
* No author or publication-date verification is performed.
