"""Per-claim verification.

The article is decomposed into atomic, individually verifiable claims; each
claim is then independently retrieved-against and labeled. This gives a
finer-grained signal than the legacy whole-article LLM judgment: a single
unverified rumor inside an otherwise factual piece drags the score down
*for that claim* rather than getting averaged out, and the per-claim
breakdown can be surfaced to the user.

Public entry points:

* ``decompose_claims(content)`` — LLM extracts ≤ MAX_CLAIMS atomic claims.
* ``verify_claim(claim, exclude_domain)`` — Tavily search + LLM stance
  labeling against the retrieved hits.
* ``verify_all_claims(claims, exclude_domain)`` — concurrent fan-out over
  the claim list.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any
from urllib.parse import urlparse

from mistralai import Mistral
from tenacity import retry, stop_after_attempt, wait_exponential

from backend.config import settings
from backend.models import ClaimEvidence, ClaimVerification
from backend.reliability.content_extractor import ExtractedContent
from backend.reliability.domain_reputation import lookup as lookup_domain
from backend.search.web_search import SearchHit, web_search

MAX_CLAIMS = 8
MAX_INPUT_CHARS = 12_000
EVIDENCE_PER_CLAIM = 4

# Backend used for per-hit stance labeling. "llm" calls Mistral; "local"
# runs the NLI model from backend.reliability.nli (requires the heavy ML
# extras — see requirements-nli.txt). Default is "llm" so existing
# deployments are unchanged.
def _nli_backend() -> str:
    return (os.getenv("NLI_BACKEND") or "llm").strip().lower()


DECOMPOSE_SYSTEM_PROMPT = """You are a fact-checking assistant.
Decompose the input article into a small set of ATOMIC, INDEPENDENTLY
VERIFIABLE factual claims. Each claim must:
- contain EXACTLY ONE factual assertion. A retrieval search should be able
  to verify each claim with a single targeted query.
- be self-contained (not a question or opinion),
- name the specific actor, action, object, and any quantities/dates verbatim
  where present,
- be checkable against external sources without referring back to the article.

CRITICAL — atomicity over faithfulness:

EVEN IF the article presents multiple facts in a single sentence connected
by "and", "where", "while", "who also", commas, or any similar construction,
you MUST split them into separate claims when each fact could be
independently true or false. Do NOT preserve the article's sentence
structure when doing so would combine independent facts. Faithfulness to
the article's phrasing is NOT a goal — atomicity is.

Worked examples of compound-splitting:

  Input: "Liliana Garces serves as VP at the Alliance, where Jackie Pedota
          also serves as a fellow."
  Output (TWO claims, not one):
    - "Liliana Garces serves as vice president at the Alliance for
       Higher Education."
    - "Jackie Pedota serves as a fellow at the Alliance for Higher
       Education."

  Input: "The bill passed the Senate 52-48 and was signed by the President
          on March 3."
  Output (TWO claims):
    - "The bill passed the Senate by a 52-48 vote."
    - "The President signed the bill on March 3."

Counter-example — do NOT over-split single events:

  Input: "Trump met Putin in Helsinki in 2018."
  Output (ONE claim — single event, can't be partially true):
    - "Trump met Putin in Helsinki in 2018."

DO NOT include rhetorical statements, value judgments, summary statements,
or speculation framed with "may", "could", "is expected to" — those are not
atomic claims. Skip anything that is purely opinion.

If the piece is satire/parody/fabricated, extract the FACTUAL-SOUNDING claims
the piece makes (as if they were stated as fact) so they can be checked.
Return STRICT JSON only:

{"claims": ["claim 1", "claim 2", ...]}

Return AT MOST 6 claims. Prefer the most concrete and high-signal ones.
"""


STANCE_SYSTEM_PROMPT = """You are evaluating whether independent sources
corroborate a target claim. For each search hit below, label its agreement
with the TARGET CLAIM as exactly one of:
"supports", "contradicts", or "unclear".

Rules:
- "supports": the hit explicitly affirms the same factual content as the claim.
- "contradicts": the hit explicitly asserts the opposite, or asserts a fact
  that is incompatible with the claim.
- "unclear": the hit is on-topic but does not assert one way or the other,
  OR the hit is off-topic, OR the snippet is too short to judge.

Output STRICT JSON only, in the SAME ORDER as the input hits:

{"labels": ["supports" | "contradicts" | "unclear", ...]}
"""


# ---------- LLM plumbing ----------


def _parse_json(raw: str) -> dict[str, Any]:
    raw = raw.strip().strip("`")
    if raw.startswith("json"):
        raw = raw[4:].lstrip()
    match = re.search(r"\{.*\}", raw, re.S)
    if not match:
        raise ValueError("model did not return JSON")
    return json.loads(match.group(0))


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
def _call_decompose(client: Mistral, prompt: str) -> str:
    response = client.chat.complete(
        model=settings.mistral_model,
        messages=[
            {"role": "system", "content": DECOMPOSE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content or ""


@retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=4))
def _call_stance(client: Mistral, prompt: str) -> str:
    response = client.chat.complete(
        model=settings.mistral_model,
        messages=[
            {"role": "system", "content": STANCE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content or ""


# ---------- Public surface ----------


def _clean_claim(s: str) -> str:
    # Strip trailing punctuation noise and quotes; cap length.
    s = re.sub(r"\s+", " ", s).strip().strip("\"' .")
    return s[:280]


async def decompose_claims(content: ExtractedContent) -> list[str]:
    """Use Mistral to extract atomic factual claims from the article.

    If the LLM is not configured, returns an empty list — callers should
    fall back to ``ContentAnalysis.main_claims`` from the legacy analyzer.
    """
    if not settings.has_llm or not (content.text or "").strip():
        return []

    snippet = content.text[:MAX_INPUT_CHARS]
    prompt = (
        f"TITLE: {content.title or '(none)'}\n"
        f"URL:   {content.url or '(none)'}\n\n"
        f"--- ARTICLE TEXT ---\n{snippet}\n--- END ---\n\n"
        "Return the JSON object now."
    )

    client = Mistral(api_key=settings.mistral_api_key)
    try:
        raw = await asyncio.to_thread(_call_decompose, client, prompt)
        data = _parse_json(raw)
    except Exception:  # pragma: no cover - network/parse failures are expected
        return []

    raw_claims = data.get("claims") or []
    if not isinstance(raw_claims, list):
        return []
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in raw_claims:
        if not isinstance(item, str):
            continue
        c = _clean_claim(item)
        if len(c) < 12:
            continue
        key = c.lower()[:120]
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(c)
        if len(cleaned) >= MAX_CLAIMS:
            break
    return cleaned


async def _label_hits_llm(claim: str, hits: list[SearchHit]) -> list[str]:
    if not hits or not settings.has_llm:
        return ["unclear"] * len(hits)

    serialized = "\n\n".join(
        f"[{i}] {h.title}\nURL: {h.url}\nSNIPPET: {h.snippet}"
        for i, h in enumerate(hits)
    )
    prompt = f"TARGET CLAIM:\n{claim}\n\nHITS:\n{serialized}\n\nReturn JSON only."

    client = Mistral(api_key=settings.mistral_api_key)
    try:
        raw = await asyncio.to_thread(_call_stance, client, prompt)
        data = _parse_json(raw)
    except Exception:  # pragma: no cover
        return ["unclear"] * len(hits)

    labels = data.get("labels") or []
    out: list[str] = []
    for i in range(len(hits)):
        label = labels[i] if i < len(labels) and isinstance(labels[i], str) else "unclear"
        out.append(label if label in ("supports", "contradicts", "unclear") else "unclear")
    return out


async def _label_hits_local(claim: str, hits: list[SearchHit]) -> list[str]:
    """Run a local NLI model. The heavy import is deferred to call time."""
    if not hits:
        return []
    try:
        from backend.reliability import nli  # local import — heavy deps
        return await asyncio.to_thread(nli.label_hits_nli, claim, hits)
    except Exception:  # pragma: no cover - fall back if model load fails
        return ["unclear"] * len(hits)


async def _label_hits(claim: str, hits: list[SearchHit]) -> list[str]:
    """Stance-label each search hit against a claim.

    Backend selected by the ``NLI_BACKEND`` env var: ``llm`` (default,
    Mistral) or ``local`` (fine-tuned NLI model from
    ``backend.reliability.nli``).
    """
    if not hits:
        return []
    backend = _nli_backend()
    if backend == "local":
        return await _label_hits_local(claim, hits)
    return await _label_hits_llm(claim, hits)


def _claim_status(support_ratio: float, contradict_ratio: float, n_evidence: int) -> str:
    # Support threshold (0.35) is calibrated below the previous 0.5 to
    # account for the fine-tuned NLI labeler's empirical conservatism on
    # real-news evidence: ~30% of obviously-supporting hits get labeled
    # "unclear" because of hedging language, partial overlap, or framing
    # differences. Without this calibration, supported facts kept landing
    # in "mixed". See README A/B section + the SB17 demo trace for the
    # specific cases that motivated this.
    if n_evidence == 0:
        return "unverified"
    if contradict_ratio >= 0.4:
        return "contradicted"
    if support_ratio >= 0.35 and contradict_ratio < 0.2:
        return "supported"
    if support_ratio > 0 or contradict_ratio > 0:
        return "mixed"
    return "unverified"


def _claim_score(
    support_ratio: float, contradict_ratio: float, n_evidence: int, n_high_quality: int
) -> float:
    """Map per-claim retrieval outcomes to a 0-100 reliability score.

    Anchored at 50 (no info), pushed up by reputation-weighted support and
    down by reputation-weighted contradiction, with bonuses for source
    quality and for the "system found relevant high-rep coverage and
    nothing contradicts" pattern (which our conservative NLI labeler tends
    to mark as "unclear" even when the evidence is clearly on-topic).
    """
    if n_evidence == 0:
        return 50.0
    base = 50.0
    base += 45.0 * support_ratio
    # Slightly softer than the original -60 — a single mislabeled
    # "contradicts" from a real news evidence pair shouldn't single-handedly
    # crater an otherwise well-supported claim.
    base -= 55.0 * contradict_ratio
    # Quality bonus: scales with high-quality source count, larger ceiling
    # than the old +10 so source diversity gets more credit.
    base += min(15.0, n_high_quality * 4.0)
    # On-topic coverage bonus: when retrieval pulled 2+ high-quality sources
    # and none of them actively contradict, that's real corroborative
    # signal even if the NLI labeler couldn't commit to "supports". This
    # specifically addresses the SB17/Garces failure mode where all the
    # evidence was clearly on-topic but labeled "unclear".
    if n_high_quality >= 2 and contradict_ratio < 0.2:
        base += 5.0
    return round(max(0.0, min(100.0, base)), 1)


async def verify_claim(claim: str, *, exclude_domain: str | None = None) -> ClaimVerification:
    """Retrieve evidence for a claim and label it.

    No LLM / no search keys → returns an ``unverified`` result with empty
    evidence so the caller can still aggregate.
    """
    hits = await web_search(claim, k=EVIDENCE_PER_CLAIM, exclude_domain=exclude_domain)
    if not hits:
        return ClaimVerification(
            claim=claim,
            score=50.0,
            status="unverified",
            support_ratio=0.0,
            contradict_ratio=0.0,
            n_evidence=0,
            n_high_quality=0,
            evidence=[],
        )

    labels = await _label_hits(claim, hits)

    evidence: list[ClaimEvidence] = []
    weighted_support = 0.0
    weighted_contradict = 0.0
    weight_total = 0.0
    n_high_quality = 0
    for hit, label in zip(hits, labels):
        rep = lookup_domain(hit.url)
        if rep.score >= 75:
            n_high_quality += 1
        # Floor reputation at 10 so unknown-domain hits still register some
        # signal (we don't want a single 0-rep page to fully dominate either
        # direction, but it shouldn't be silently dropped).
        weight = max(rep.score, 10.0)
        weight_total += weight
        if label == "supports":
            weighted_support += weight
        elif label == "contradicts":
            weighted_contradict += weight
        evidence.append(
            ClaimEvidence(
                title=hit.title or hit.url,
                url=hit.url,
                domain=rep.domain,
                domain_score=rep.score,
                snippet=hit.snippet,
                agreement=label,  # type: ignore[arg-type]
            )
        )

    support_ratio = weighted_support / weight_total if weight_total else 0.0
    contradict_ratio = weighted_contradict / weight_total if weight_total else 0.0
    return ClaimVerification(
        claim=claim,
        score=_claim_score(support_ratio, contradict_ratio, len(evidence), n_high_quality),
        status=_claim_status(support_ratio, contradict_ratio, len(evidence)),  # type: ignore[arg-type]
        support_ratio=round(support_ratio, 3),
        contradict_ratio=round(contradict_ratio, 3),
        n_evidence=len(evidence),
        n_high_quality=n_high_quality,
        evidence=evidence,
    )


async def verify_all_claims(
    claims: list[str], *, exclude_domain: str | None = None, concurrency: int = 4
) -> list[ClaimVerification]:
    """Verify each claim concurrently, bounded by ``concurrency``."""
    if not claims:
        return []
    sem = asyncio.Semaphore(max(1, concurrency))

    async def _one(c: str) -> ClaimVerification:
        async with sem:
            return await verify_claim(c, exclude_domain=exclude_domain)

    return await asyncio.gather(*[_one(c) for c in claims])


def domain_of(url: str | None) -> str | None:
    if not url:
        return None
    host = urlparse(url).hostname or None
    return host.lower() if host else None
