"""Tests for the citation-audit pipeline.

Link extraction is tested as a pure function; the end-to-end audit is
tested with the source fetch + stance labeler monkeypatched, so it runs
offline and deterministically (no network / no API keys).
"""

from __future__ import annotations

import asyncio

import httpx

from backend.reliability import citation_audit as ca
from backend.reliability.content_extractor import ExtractedContent

ARTICLE_HTML = """
<html><body>
<nav><a href="https://othersite.com/menu">Menu</a></nav>
<article>
  <p>The vaccine was 95% effective in the trial,
     <a href="https://www.nejm.org/study">according to NEJM</a>.</p>
  <p>Cases fell sharply last year,
     <a href="https://www.cdc.gov/data">per CDC figures</a>.</p>
  <p>Share this on <a href="https://twitter.com/x">Twitter</a> or read
     our <a href="https://mysite.com/about">about page</a>.</p>
</article>
</body></html>
"""

# Two external in-prose citations only — used for the text-input audit path,
# which has no base URL (so same-domain filtering can't apply).
AUDIT_HTML = """
<article>
  <p>The vaccine was 95% effective in the trial,
     <a href="https://www.nejm.org/study">according to NEJM</a>.</p>
  <p>Cases fell sharply last year,
     <a href="https://www.cdc.gov/data">per CDC figures</a>.</p>
</article>
"""


def test_extract_citations_keeps_only_in_prose_external_links():
    cits = ca.extract_citations(ARTICLE_HTML, "https://mysite.com/post")
    urls = {c.url for c in cits}
    assert urls == {"https://www.nejm.org/study", "https://www.cdc.gov/data"}
    # nav link, twitter (social), and same-domain about page are excluded.
    for c in cits:
        assert "twitter.com" not in c.url
        assert "mysite.com" not in c.url


def test_extract_citations_captures_citing_sentence():
    cits = ca.extract_citations(ARTICLE_HTML, "https://mysite.com/post")
    by_url = {c.url: c for c in cits}
    assert "95% effective" in by_url["https://www.nejm.org/study"].citing_sentence


def _patch_audit(monkeypatch, *, sources: dict[str, str], labels: dict[str, str]):
    async def fake_extract(url: str) -> ExtractedContent:
        if url not in sources:
            raise httpx.HTTPStatusError("404", request=None, response=httpx.Response(404))
        return ExtractedContent(url=url, title="src", text=sources[url], n_links=0, has_citations=False)

    async def fake_label(claim: str, hits):
        # Label by the (single) hit's URL via the labels map.
        url = hits[0].url if hits else ""
        return [labels.get(url, "unclear")]

    monkeypatch.setattr(ca, "extract_from_url", fake_extract)
    monkeypatch.setattr(ca, "_label_hits_llm", fake_label)
    monkeypatch.setattr(ca, "_label_hits", fake_label)


def test_assess_citation_audit_mixed(monkeypatch):
    _patch_audit(
        monkeypatch,
        sources={
            "https://www.nejm.org/study": "The trial reported 95% efficacy.",
            "https://www.cdc.gov/data": "Unrelated text about budget policy.",
        },
        labels={
            "https://www.nejm.org/study": "supports",
            "https://www.cdc.gov/data": "unclear",
        },
    )
    report = asyncio.run(ca.assess_citation_audit(text=AUDIT_HTML))
    cits = report.content_analysis.claim_verifications
    assert len(cits) == 2
    statuses = {c.status for c in cits}
    assert "supported" in statuses
    # 1 of 2 supported → integrity 50.
    assert report.overall_score == 50.0
    # The unsupported citation produces a red flag.
    assert len(report.content_analysis.red_flags) == 1


def test_assess_citation_audit_unreachable_source_is_unverified(monkeypatch):
    _patch_audit(
        monkeypatch,
        sources={},  # every fetch 404s
        labels={},
    )
    report = asyncio.run(ca.assess_citation_audit(text=AUDIT_HTML))
    cits = report.content_analysis.claim_verifications
    assert len(cits) == 2
    assert all(c.status == "unverified" for c in cits)
    assert report.overall_score == 0.0


def test_assess_citation_audit_no_citations_plain_text(monkeypatch):
    report = asyncio.run(ca.assess_citation_audit(text="Just some prose with no links at all."))
    assert report.content_analysis.claim_verifications == []
    assert report.overall_score == 50.0
    assert "No external in-text citations" in report.explanation
