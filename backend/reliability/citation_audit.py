"""Citation Audit.

Given an article, this checks whether each *cited source* actually supports
the sentence that cites it. It targets a real failure mode that source-level
reliability scoring misses entirely: an article from a reputable outlet can
still attach a link to a claim the linked page does not support (stale links,
misread studies, citation laundering, hallucinated references in AI-written
copy).

Pipeline:
    1. Parse the article HTML for in-prose hyperlinks (<a> inside p/li/
       blockquote), capturing the citing sentence + the external target URL.
    2. Fetch each cited source and extract its readable text.
    3. NLI-label the citing sentence against the cited source's text
       (supports / contradicts / unclear) using the same stance backend as
       the per-claim pipeline.
    4. Aggregate into a citation-integrity score = share of citations whose
       source actually supports the citing sentence.

Reuses ``_build_verification`` + ``_label_hits`` so the per-citation cards
carry the same evidence/scoring shape as per-claim verification, and renders
through the shared ReliabilityReport schema.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from backend.models import (
    ClaimVerification,
    ContentAnalysis,
    CorroboratingSource,
    CrossReferenceResult,
    DomainReputation,
    ReliabilityReport,
)
from backend.config import settings
from backend.reliability.claims import (
    _build_verification,
    _label_hits,
    _label_hits_llm,
    domain_of,
)
from backend.reliability.content_extractor import extract_from_url, fetch_url
from backend.reliability.scorer import _verdict
from backend.search.web_search import SearchHit

MAX_CITATIONS = 8
SOURCE_SNIPPET_CHARS = 2000
MAX_SENTENCE_CHARS = 400

# Social/sharing/junk hosts that are links but not substantive citations.
_SKIP_HOST_SUBSTR = (
    "twitter.com", "x.com", "facebook.com", "instagram.com", "youtube.com",
    "youtu.be", "linkedin.com", "t.co", "reddit.com", "tiktok.com",
    "pinterest.com", "whatsapp.com",
)
_NEUTRAL_DOMAIN = DomainReputation(
    domain="(article)", score=50.0, bias="n/a", type="unknown",
    rationale="Citation audit — the article's own reliability is not scored here; "
    "each cited source is checked against the sentence that cites it.",
)


@dataclass
class Citation:
    citing_sentence: str
    anchor_text: str
    url: str


def _is_external(href: str, base_host: str | None) -> bool:
    p = urlparse(href)
    if p.scheme not in ("http", "https"):
        return False
    host = (p.hostname or "").lower()
    if not host:
        return False
    if base_host and (host == base_host or host.endswith("." + base_host)):
        return False
    return not any(s in host for s in _SKIP_HOST_SUBSTR)


def _citing_sentence(anchor) -> str:
    block = anchor.find_parent(["p", "li", "blockquote"]) or anchor.parent
    text = block.get_text(" ", strip=True) if block else anchor.get_text(" ", strip=True)
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) <= MAX_SENTENCE_CHARS:
        return text
    # Narrow to the sentence span containing the anchor text when the block
    # is long (e.g. a whole paragraph with several links).
    anchor_text = re.sub(r"\s+", " ", anchor.get_text(" ", strip=True) or "").strip()
    if anchor_text and anchor_text in text:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        for s in sentences:
            if anchor_text in s:
                return s[:MAX_SENTENCE_CHARS]
    return text[:MAX_SENTENCE_CHARS]


def extract_citations(
    html: str, base_url: str | None, *, max_citations: int = MAX_CITATIONS
) -> list[Citation]:
    """Pull in-prose external hyperlinks + their citing sentence from HTML."""
    soup = BeautifulSoup(html or "", "html.parser")
    container = soup.find("article") or soup.find("main") or soup.body or soup
    base_host = domain_of(base_url) if base_url else None

    seen: set[str] = set()
    out: list[Citation] = []
    for a in container.find_all("a", href=True):
        # Restrict to body prose to avoid nav bars, menus, and footers.
        if not a.find_parent(["p", "li", "blockquote"]):
            continue
        href = urljoin(base_url or "", a["href"])
        if not _is_external(href, base_host) or href in seen:
            continue
        sentence = _citing_sentence(a)
        if len(sentence) < 20:
            continue
        seen.add(href)
        out.append(
            Citation(
                citing_sentence=sentence,
                anchor_text=(a.get_text(" ", strip=True) or "")[:120],
                url=href,
            )
        )
        if len(out) >= max_citations:
            break
    return out


async def _verify_citation(c: Citation) -> ClaimVerification:
    """Fetch the cited source and label whether it supports the citing sentence."""
    snippet = ""
    title = c.url
    try:
        src = await extract_from_url(c.url)
        snippet = (src.text or "")[:SOURCE_SNIPPET_CHARS]
        title = src.title or c.url
    except (httpx.HTTPError, ValueError, RuntimeError):
        snippet = ""

    if not snippet:
        # Source unreachable / empty — can't confirm support.
        v = _build_verification(c.citing_sentence, [], [])
        return v

    hit = SearchHit(title=title, url=c.url, snippet=snippet)
    # Prefer the LLM stance labeler here: citation audit compares a short
    # citing sentence against a long source document, where the small local
    # NLI model is overly conservative (labels clear support "unclear"). Fall
    # back to the configured backend if no LLM is available.
    if settings.has_llm:
        labels = await _label_hits_llm(c.citing_sentence, [hit])
    else:
        labels = await _label_hits(c.citing_sentence, [hit])
    return _build_verification(c.citing_sentence, [hit], labels)


def _looks_like_html(s: str) -> bool:
    return bool(re.search(r"<a\s+[^>]*href=", s or "", re.I))


async def assess_citation_audit(
    *,
    url: str | None = None,
    text: str | None = None,
    max_citations: int = MAX_CITATIONS,
) -> ReliabilityReport:
    """Audit an article's citations and wrap the result as a ReliabilityReport.

    Provide ``url`` (fetched + parsed for links) or ``text`` (used directly
    if it contains HTML anchors; plain text yields no citations).
    """
    if url:
        html = await fetch_url(url)
        base_url = url
    elif text and _looks_like_html(text):
        html = text
        base_url = None
    else:
        html = ""
        base_url = None

    citations = extract_citations(html, base_url, max_citations=max_citations)
    verifications = await asyncio.gather(*[_verify_citation(c) for c in citations]) if citations else []

    n = len(verifications)
    n_supported = sum(1 for v in verifications if v.status == "supported")
    n_unsupported = n - n_supported
    integrity = round(100.0 * n_supported / n, 1) if n else 50.0

    red_flags: list[str] = []
    for v in verifications:
        if v.status != "supported":
            label = {
                "contradicted": "Cited source contradicts the claim",
                "mixed": "Cited source only partially supports the claim",
                "unverified": "Cited source does not clearly support the claim (or was unreachable)",
            }.get(v.status, "Cited source does not support the claim")
            red_flags.append(f'{label}: "{v.claim[:140]}"')

    if n == 0:
        summary = (
            "No external in-text citations were found to audit. Provide an article "
            "URL (or HTML) that links out to its sources."
        )
    else:
        summary = (
            f"{n_supported}/{n} cited sources actually support the sentence that cites them"
            f"{'; ' + str(n_unsupported) + ' do not.' if n_unsupported else '.'}"
        )

    sources = [
        CorroboratingSource(
            title=e.title, url=e.url, domain=e.domain, domain_score=e.domain_score,
            snippet=e.snippet, agreement=e.agreement,
        )
        for v in verifications
        for e in v.evidence
    ]
    xref = CrossReferenceResult(
        score=integrity,
        n_sources=len(sources),
        n_high_quality=sum(1 for s in sources if s.domain_score >= 75),
        sources=sources,
        consensus=("strong-support" if integrity >= 70 else "mixed" if n_supported else "no-data"),  # type: ignore[arg-type]
    )

    content = ContentAnalysis(
        score=integrity, factuality=integrity, objectivity=50.0, transparency=50.0,
        sensationalism=50.0,
        main_claims=[c.citing_sentence for c in citations],
        red_flags=red_flags, citations_present=n > 0, summary=summary,
        claim_verifications=verifications, coverage=1.0 if n else 0.0,
    )

    return ReliabilityReport(
        overall_score=integrity,
        verdict=_verdict(integrity),  # type: ignore[arg-type]
        confidence=round(min(1.0, 0.3 + 0.1 * n), 2),
        domain_reputation=_NEUTRAL_DOMAIN,
        content_analysis=content,
        cross_reference=xref,
        explanation=summary,
        weights={"citation_integrity": 1.0},
    )
