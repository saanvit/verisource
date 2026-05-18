"""LLM-driven content reliability analysis using Mistral.

Asks the model to produce a structured JSON judgment over several reliability
sub-signals (factuality, objectivity, transparency, sensationalism), the main
factual claims, and any red flags. We sanitize/clip the output and fall back
to a deterministic heuristic if the LLM is unavailable.
"""

from __future__ import annotations

import json
import re
from typing import Any

from mistralai import Mistral
from tenacity import retry, stop_after_attempt, wait_exponential

from backend.config import settings
from backend.models import ContentAnalysis
from backend.reliability.content_extractor import ExtractedContent

MAX_INPUT_CHARS = 12_000

SYSTEM_PROMPT = """You are an expert media-literacy analyst.
Given a news article or piece of writing, return a STRICT JSON object that scores
the source's reliability on multiple axes. Be conservative: when unsure, score
toward the middle (50). Output JSON ONLY, no prose, no markdown fences.

ALWAYS populate "main_claims" with the article's purported factual claims, even
if you suspect the article is satire, opinion, or fabricated — list what the
article asserts AS IF it were factual. Use "red_flags" to record that it is
satire/opinion/etc.

If the piece is satire or parody (e.g. The Onion, Babylon Bee, Clickhole, or
similar comedic content), set factuality near 0 and add an explicit red flag:
"This is satire/parody, not factual reporting."

Schema:
{
  "factuality":      0-100,  // verifiability and accuracy of factual claims
  "objectivity":     0-100,  // neutral framing, low loaded language
  "transparency":    0-100,  // named author, dateline, sources/links cited
  "sensationalism":  0-100,  // 0 = pure clickbait, 100 = restrained tone
  "citations_present": true | false,
  "main_claims":     [string, ...]   // up to 5 short factual claims being made
  "red_flags":       [string, ...]   // up to 5 specific reliability concerns
  "summary":         string          // 2-3 sentence overall assessment
}
"""


def _extract_claims_heuristic(text: str, limit: int = 5) -> list[str]:
    """Pull the first few declarative-looking sentences as candidate claims.

    This is a crude fallback for when the LLM is unavailable. We split on
    sentence punctuation, keep sentences that have a verb-like token and a
    reasonable length, and drop questions/exclamations.
    """

    if not text:
        return []
    # Normalize whitespace and split into sentences.
    cleaned = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z\"'])", cleaned)

    claims: list[str] = []
    seen: set[str] = set()
    for raw in sentences:
        s = raw.strip().rstrip("\"'")
        if len(s) < 40 or len(s) > 280:
            continue
        if s.endswith("?"):
            continue
        # Require at least one common verb-like token.
        if not re.search(r"\b(is|are|was|were|has|have|had|said|reported|"
                         r"announced|claimed|confirmed|found|will|would|did)\b",
                         s, re.I):
            continue
        key = s.lower()[:80]
        if key in seen:
            continue
        seen.add(key)
        claims.append(s)
        if len(claims) >= limit:
            break
    return claims


def _heuristic_analysis(content: ExtractedContent) -> ContentAnalysis:
    text = content.text or ""
    word_count = len(text.split())

    sensational_tokens = re.findall(
        r"\b(shocking|outrageous|destroyed|slammed|bombshell|you won't believe|insane)\b",
        text,
        flags=re.I,
    )
    hedging_tokens = re.findall(r"\b(may|might|could|reportedly|allegedly|suggests)\b", text, re.I)
    quoted = len(re.findall(r"\"[^\"]{20,}\"", text))

    sensationalism = max(0, 80 - min(60, len(sensational_tokens) * 12))
    objectivity = max(20, 70 - min(50, len(sensational_tokens) * 8) + min(15, len(hedging_tokens)))
    transparency = 40 + min(30, content.n_links // 2) + (15 if content.has_citations else 0)
    factuality = 50 + min(20, quoted * 4) - min(20, len(sensational_tokens) * 5)

    factuality = max(0, min(100, factuality))
    objectivity = max(0, min(100, objectivity))
    transparency = max(0, min(100, transparency))
    sensationalism = max(0, min(100, sensationalism))

    score = round(
        0.40 * factuality + 0.25 * objectivity + 0.20 * transparency + 0.15 * sensationalism,
        1,
    )

    red_flags: list[str] = []
    if sensational_tokens:
        red_flags.append(
            f"Sensationalist phrases detected: {', '.join(set(sensational_tokens[:3]))}."
        )
    if not content.has_citations and content.n_links < 3:
        red_flags.append("Few or no outgoing citations/links.")
    if word_count < 200:
        red_flags.append("Very short article — limited evidence available.")

    return ContentAnalysis(
        score=score,
        factuality=factuality,
        objectivity=objectivity,
        transparency=transparency,
        sensationalism=sensationalism,
        main_claims=_extract_claims_heuristic(text),
        red_flags=red_flags,
        citations_present=content.has_citations,
        summary=(
            "Heuristic-only analysis (LLM not configured). Scoring is based on simple "
            "lexical and structural signals; main claims are extracted by sentence pattern."
        ),
    )


def _coerce_score(value: Any, default: float = 50.0) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(100.0, f))


def _coerce_str_list(value: Any, limit: int = 5) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value[:limit]:
        if isinstance(item, str):
            s = item.strip()
            if s:
                out.append(s[:280])
    return out


def _parse_llm_json(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?", "", raw).strip()
        raw = re.sub(r"```$", "", raw).strip()
    match = re.search(r"\{.*\}", raw, re.S)
    if not match:
        raise ValueError("Model did not return JSON")
    return json.loads(match.group(0))


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
def _call_mistral(client: Mistral, prompt: str) -> str:
    response = client.chat.complete(
        model=settings.mistral_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content or ""


async def analyze_content(content: ExtractedContent) -> ContentAnalysis:
    if not settings.has_llm or not (content.text or "").strip():
        return _heuristic_analysis(content)

    snippet = content.text[:MAX_INPUT_CHARS]
    prompt = (
        f"TITLE: {content.title or '(none)'}\n"
        f"URL:   {content.url or '(none)'}\n"
        f"CITATION SIGNALS: n_links={content.n_links}, has_citations={content.has_citations}\n\n"
        f"--- ARTICLE TEXT ---\n{snippet}\n--- END ---\n\n"
        "Return the JSON object now."
    )

    client = Mistral(api_key=settings.mistral_api_key)
    try:
        raw = _call_mistral(client, prompt)
        data = _parse_llm_json(raw)
    except Exception:  # pragma: no cover - network/parse failures are expected in dev
        return _heuristic_analysis(content)

    factuality = _coerce_score(data.get("factuality"))
    objectivity = _coerce_score(data.get("objectivity"))
    transparency = _coerce_score(data.get("transparency"))
    sensationalism = _coerce_score(data.get("sensationalism"))

    score = round(
        0.40 * factuality + 0.25 * objectivity + 0.20 * transparency + 0.15 * sensationalism,
        1,
    )

    return ContentAnalysis(
        score=score,
        factuality=factuality,
        objectivity=objectivity,
        transparency=transparency,
        sensationalism=sensationalism,
        main_claims=_coerce_str_list(data.get("main_claims")),
        red_flags=_coerce_str_list(data.get("red_flags")),
        citations_present=bool(data.get("citations_present", content.has_citations)),
        summary=(data.get("summary") or "").strip()[:1000]
        or "LLM did not provide a summary.",
    )
