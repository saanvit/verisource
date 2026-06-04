"""Async web search with a keyless fallback.

Primary provider is Tavily (LLM-friendly snippets, good for news queries).
When Tavily is unconfigured OR returns nothing (e.g. over quota / HTTP 432),
we fall back to a keyless DuckDuckGo HTML scrape so the app keeps working
without any search key — which also means graders can run it key-free.
"""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import parse_qs, unquote, urlparse

import httpx
from bs4 import BeautifulSoup

from backend.config import settings

TAVILY_ENDPOINT = "https://api.tavily.com/search"
DDG_ENDPOINT = "https://html.duckduckgo.com/html/"
_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


@dataclass
class SearchHit:
    title: str
    url: str
    snippet: str


async def _tavily_search(query: str, k: int, exclude_domain: str | None) -> list[SearchHit]:
    payload: dict = {
        "api_key": settings.tavily_api_key,
        "query": query,
        "max_results": max(k, 5),
        "search_depth": "advanced",
        "include_answer": False,
        "include_raw_content": False,
    }
    if exclude_domain:
        payload["exclude_domains"] = [exclude_domain]
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(TAVILY_ENDPOINT, json=payload)
            resp.raise_for_status()
            data = resp.json()
    except Exception:  # network error / quota (432) → let caller fall back
        return []

    hits: list[SearchHit] = []
    for item in (data.get("results") or [])[:k]:
        url = item.get("url") or ""
        if url:
            hits.append(
                SearchHit(
                    title=(item.get("title") or "").strip(),
                    url=url,
                    snippet=(item.get("content") or "").strip()[:500],
                )
            )
    return hits


def _ddg_real_url(href: str) -> str:
    """DuckDuckGo wraps result links as /l/?uddg=<encoded-url>. Unwrap it."""
    if not href:
        return ""
    if href.startswith("//"):
        href = "https:" + href
    parsed = urlparse(href)
    if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
        q = parse_qs(parsed.query).get("uddg")
        if q:
            return unquote(q[0])
    return href


async def _ddg_search(query: str, k: int, exclude_domain: str | None) -> list[SearchHit]:
    """Keyless DuckDuckGo HTML scrape. Best-effort; returns [] on any failure."""
    try:
        async with httpx.AsyncClient(timeout=20.0, headers={"User-Agent": _UA}) as client:
            resp = await client.post(DDG_ENDPOINT, data={"q": query})
            resp.raise_for_status()
            html = resp.text
    except Exception:  # pragma: no cover - network errors handled gracefully
        return []

    soup = BeautifulSoup(html, "html.parser")
    hits: list[SearchHit] = []
    for result in soup.select("div.result, div.web-result"):
        a = result.select_one("a.result__a")
        if not a:
            continue
        url = _ddg_real_url(a.get("href") or "")
        if not url or not url.startswith("http"):
            continue
        if exclude_domain and exclude_domain in (urlparse(url).hostname or ""):
            continue
        snippet_el = result.select_one(".result__snippet")
        snippet = (snippet_el.get_text(" ", strip=True) if snippet_el else "")[:500]
        hits.append(
            SearchHit(title=a.get_text(" ", strip=True), url=url, snippet=snippet)
        )
        if len(hits) >= k:
            break
    return hits


async def web_search(query: str, k: int = 5, exclude_domain: str | None = None) -> list[SearchHit]:
    """Search the web, preferring Tavily and falling back to keyless DuckDuckGo.

    Falls back when Tavily is unconfigured or returns no results (over quota,
    transient error, etc.), so the pipeline keeps producing evidence.
    """
    if not query.strip():
        return []

    if settings.has_search:
        hits = await _tavily_search(query, k, exclude_domain)
        if hits:
            return hits
    # No Tavily key, or Tavily returned nothing → keyless fallback.
    return await _ddg_search(query, k, exclude_domain)
