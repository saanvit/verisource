"""Thin async wrapper around the Tavily search API.

We use Tavily because it returns LLM-friendly snippets and works well for
news-style queries. If no API key is configured, returns [].
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx

from backend.config import settings

TAVILY_ENDPOINT = "https://api.tavily.com/search"


@dataclass
class SearchHit:
    title: str
    url: str
    snippet: str


async def web_search(query: str, k: int = 5, exclude_domain: str | None = None) -> list[SearchHit]:
    if not settings.has_search or not query.strip():
        return []

    payload = {
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
    except Exception:  # pragma: no cover - network errors handled gracefully
        return []

    hits: list[SearchHit] = []
    for item in (data.get("results") or [])[:k]:
        url = item.get("url") or ""
        if not url:
            continue
        hits.append(
            SearchHit(
                title=(item.get("title") or "").strip(),
                url=url,
                snippet=(item.get("content") or "").strip()[:500],
            )
        )
    return hits
