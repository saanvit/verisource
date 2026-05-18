"""Fetch a URL and extract its main textual content.

Uses trafilatura for clean article extraction, falling back to BeautifulSoup
if trafilatura returns nothing useful (e.g. for very short pages).
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx
import trafilatura
from bs4 import BeautifulSoup

USER_AGENT = (
    "Mozilla/5.0 (compatible; SourceReliabilityBot/1.0; "
    "+https://example.com/source-reliability)"
)
DEFAULT_TIMEOUT = 20.0
MAX_BYTES = 2_000_000


@dataclass
class ExtractedContent:
    url: str
    title: str
    text: str
    n_links: int
    has_citations: bool


async def fetch_url(url: str) -> str:
    async with httpx.AsyncClient(
        follow_redirects=True,
        timeout=DEFAULT_TIMEOUT,
        headers={"User-Agent": USER_AGENT},
    ) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.text[:MAX_BYTES]


def _extract_with_trafilatura(html: str, url: str) -> tuple[str, str] | None:
    text = trafilatura.extract(
        html,
        url=url,
        include_comments=False,
        include_tables=False,
        favor_recall=True,
    )
    if not text or len(text.strip()) < 200:
        return None
    metadata = trafilatura.extract_metadata(html)
    title = metadata.title if metadata and metadata.title else ""
    return title, text


def _extract_with_bs(html: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.string.strip() if soup.title and soup.title.string else ""
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    text = "\n\n".join(p for p in paragraphs if len(p) > 40)
    return title, text


def _citation_signals(html: str) -> tuple[int, bool]:
    soup = BeautifulSoup(html, "html.parser")
    article = soup.find("article") or soup
    links = article.find_all("a", href=True)
    n_links = len(links)
    has_citations = any(
        any(token in (a.get("href") or "").lower() for token in ("doi.org", "pubmed", "arxiv"))
        or (a.get_text() or "").strip().startswith("[")
        for a in links
    )
    return n_links, has_citations


async def extract_from_url(url: str) -> ExtractedContent:
    html = await fetch_url(url)

    extracted = _extract_with_trafilatura(html, url)
    if extracted is None:
        title, text = _extract_with_bs(html)
    else:
        title, text = extracted

    n_links, has_citations = _citation_signals(html)
    return ExtractedContent(
        url=url,
        title=title or "",
        text=text or "",
        n_links=n_links,
        has_citations=has_citations,
    )


def extract_from_text(text: str) -> ExtractedContent:
    return ExtractedContent(
        url="",
        title="",
        text=text,
        n_links=0,
        has_citations=False,
    )
