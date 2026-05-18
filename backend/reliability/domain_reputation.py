"""Look up a domain's prior reliability.

Strategy (in priority order):
1. Exact match on registered domain (e.g. ``nytimes.com``).
2. Exact match on the host (e.g. ``bbc.co.uk``, ``science.org``).
3. TLD-based prior (``.gov``, ``.edu``, ``.ac.uk``, etc.).
4. Generic ``unknown`` prior.
"""

from __future__ import annotations

import json
from functools import lru_cache
from urllib.parse import urlparse

import tldextract

from backend.config import DATA_DIR, PROJECT_ROOT
from backend.models import DomainReputation

# Use a workspace-local cache so the lookup works in sandboxed/read-only-home
# environments (CI, containers, etc.) without touching ~/.cache.
_TLD_CACHE_DIR = PROJECT_ROOT / ".cache" / "tldextract"
_TLD_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_extractor = tldextract.TLDExtract(
    cache_dir=str(_TLD_CACHE_DIR),
    suffix_list_urls=(),  # use the bundled snapshot; no network needed
)


@lru_cache(maxsize=1)
def _load_db() -> dict:
    with open(DATA_DIR / "domain_reputation.json", encoding="utf-8") as f:
        return json.load(f)


def _registered_domain(host: str) -> str:
    extracted = _extractor(host)
    if extracted.registered_domain:
        return extracted.registered_domain.lower()
    return host.lower()


def _host_from(value: str) -> str:
    if "://" in value:
        return urlparse(value).hostname or ""
    return value.lower()


def lookup(url_or_text: str | None) -> DomainReputation:
    """Return reputation for a URL. For raw-text inputs, returns the unknown prior."""

    if not url_or_text or "://" not in url_or_text:
        return DomainReputation(
            domain="unknown",
            score=50.0,
            bias="unknown",
            type="unknown",
            rationale=(
                "No URL provided; treating source as unknown. Reliability is being judged "
                "purely on content and external corroboration."
            ),
        )

    db = _load_db()
    host = _host_from(url_or_text)
    if not host:
        return DomainReputation(
            domain="unknown",
            score=50.0,
            bias="unknown",
            type="unknown",
            rationale="Could not parse host from URL.",
        )

    registered = _registered_domain(host)

    def _rationale(domain: str, rec: dict) -> str:
        if rec["type"] == "satire":
            return (
                f"{domain} is a known satire/parody publication. Its content is comedic and "
                "should not be treated as factual reporting."
            )
        return f"Curated reputation for {domain} (type={rec['type']}, bias={rec['bias']})."

    domains = db.get("domains", {})
    if host in domains:
        rec = domains[host]
        return DomainReputation(
            domain=host,
            score=float(rec["score"]),
            bias=rec["bias"],
            type=rec["type"],
            rationale=_rationale(host, rec),
        )
    if registered in domains:
        rec = domains[registered]
        return DomainReputation(
            domain=registered,
            score=float(rec["score"]),
            bias=rec["bias"],
            type=rec["type"],
            rationale=_rationale(registered, rec),
        )

    tld_priors = db.get("tld_priors", {})
    for tld_key, prior in tld_priors.items():
        if registered.endswith("." + tld_key) or host.endswith("." + tld_key):
            return DomainReputation(
                domain=registered or host,
                score=float(prior["score"]),
                bias=prior["bias"],
                type=prior["type"],
                rationale=(
                    f"No curated entry for {registered or host}; using .{tld_key} prior "
                    f"(type={prior['type']})."
                ),
            )

    return DomainReputation(
        domain=registered or host,
        score=50.0,
        bias="unknown",
        type="unknown",
        rationale=(
            f"No curated entry or TLD prior for {registered or host}. Using neutral prior; "
            "weighting more on content analysis and corroboration."
        ),
    )
