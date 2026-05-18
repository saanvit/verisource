"""Cross-reference an article's claims against independent sources.

Pipeline:
1. Extract a search query — either the user-supplied claim, the article title,
   or an LLM-derived headline-style summary.
2. Search the web (Tavily), excluding the original domain.
3. Score each hit by curated domain reputation.
4. Optionally ask the LLM to label each hit as supports/contradicts/unclear.
5. Aggregate into a 0-100 corroboration score and a consensus label.
"""

from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import urlparse

from mistralai import Mistral
from tenacity import retry, stop_after_attempt, wait_exponential

from backend.config import settings
from backend.models import ContentAnalysis, CorroboratingSource, CrossReferenceResult
from backend.reliability.content_extractor import ExtractedContent
from backend.reliability.domain_reputation import lookup as lookup_domain
from backend.search.web_search import SearchHit, web_search

CONSENSUS_PROMPT = """You are evaluating whether independent sources corroborate a target claim.
For each search hit, label its agreement with the target claim as one of:
"supports", "contradicts", or "unclear". Output STRICT JSON only:

{"labels": ["supports" | "contradicts" | "unclear", ...]}

The list must be the SAME LENGTH and ORDER as the input hits.
"""


def _domain_of(url: str) -> str:
    host = urlparse(url).hostname or ""
    return host.lower()


def _build_query(
    content: ExtractedContent, analysis: ContentAnalysis, claim: str | None
) -> str:
    if claim and claim.strip():
        return claim.strip()
    if analysis.main_claims:
        return analysis.main_claims[0]
    if content.title:
        return content.title
    snippet = re.sub(r"\s+", " ", content.text[:400]).strip()
    return snippet or ""


@retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=4))
def _label_hits_with_llm(claim: str, hits: list[SearchHit]) -> list[str]:
    if not settings.has_llm or not hits:
        return ["unclear"] * len(hits)

    serialized = "\n".join(
        f"[{i}] {h.title}\nURL: {h.url}\nSNIPPET: {h.snippet}"
        for i, h in enumerate(hits)
    )
    prompt = f"TARGET CLAIM:\n{claim}\n\nHITS:\n{serialized}\n\nReturn JSON only."

    client = Mistral(api_key=settings.mistral_api_key)
    response = client.chat.complete(
        model=settings.mistral_model,
        messages=[
            {"role": "system", "content": CONSENSUS_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content or ""
    raw = raw.strip().strip("`")
    match = re.search(r"\{.*\}", raw, re.S)
    if not match:
        return ["unclear"] * len(hits)
    data: dict[str, Any] = json.loads(match.group(0))
    labels = data.get("labels") or []
    out: list[str] = []
    for i in range(len(hits)):
        label = labels[i] if i < len(labels) else "unclear"
        out.append(label if label in ("supports", "contradicts", "unclear") else "unclear")
    return out


async def cross_reference(
    content: ExtractedContent,
    analysis: ContentAnalysis,
    claim: str | None,
) -> CrossReferenceResult:
    query = _build_query(content, analysis, claim)
    if not query:
        return CrossReferenceResult(
            score=50.0,
            n_sources=0,
            n_high_quality=0,
            sources=[],
            consensus="no-data",
        )

    exclude = _domain_of(content.url) if content.url else None
    hits = await web_search(query, k=settings.cross_reference_k, exclude_domain=exclude)
    if not hits:
        return CrossReferenceResult(
            score=50.0,
            n_sources=0,
            n_high_quality=0,
            sources=[],
            consensus="no-data",
        )

    try:
        labels = _label_hits_with_llm(query, hits)
    except Exception:  # pragma: no cover
        labels = ["unclear"] * len(hits)

    sources: list[CorroboratingSource] = []
    n_high_quality = 0
    weighted_support = 0.0
    weighted_contradict = 0.0
    weight_total = 0.0

    for hit, label in zip(hits, labels, strict=False):
        rep = lookup_domain(hit.url)
        if rep.score >= 75:
            n_high_quality += 1
        sources.append(
            CorroboratingSource(
                title=hit.title or hit.url,
                url=hit.url,
                domain=rep.domain,
                domain_score=rep.score,
                snippet=hit.snippet,
                agreement=label,  # type: ignore[arg-type]
            )
        )
        weight = max(rep.score, 10.0)
        weight_total += weight
        if label == "supports":
            weighted_support += weight
        elif label == "contradicts":
            weighted_contradict += weight

    if weight_total == 0:
        consensus = "no-data"
        score = 50.0
    else:
        support_ratio = weighted_support / weight_total
        contradict_ratio = weighted_contradict / weight_total

        if support_ratio >= 0.6 and contradict_ratio < 0.1:
            consensus = "strong-support"
        elif support_ratio >= 0.3 and contradict_ratio < support_ratio:
            consensus = "weak-support"
        elif contradict_ratio >= 0.4:
            consensus = "contradicts"
        elif support_ratio > 0 or contradict_ratio > 0:
            consensus = "mixed"
        else:
            consensus = "no-data"

        base = 50.0
        base += 40.0 * support_ratio
        base -= 50.0 * contradict_ratio
        base += min(15.0, n_high_quality * 4.0)
        score = max(0.0, min(100.0, round(base, 1)))

    return CrossReferenceResult(
        score=score,
        n_sources=len(sources),
        n_high_quality=n_high_quality,
        sources=sources,
        consensus=consensus,  # type: ignore[arg-type]
    )
