"""Opinion detection + fact-grounding.

The verifiable-facts track (``claims.decompose_claims`` →
``verify_all_claims``) deliberately filters OUT opinions, so opinion
pieces produce thin output even when they're full of contestable
interpretive assertions. This module adds a complementary track:

1. Extract the article's 2-4 central OPINIONS / interpretations (1 LLM
   call). For each, the model lists the 1-3 atomic, web-verifiable
   factual premises the opinion rests on.
2. Verify all premises with the EXISTING verification machinery
   (``claims.verify_all_claims``).
3. Map verifications back to their opinions and assign each opinion a
   grounding verdict + 0-100 score: does this opinion actually rest on
   facts that hold up to independent scrutiny?

This is genuinely useful for op-eds: instead of "2 verifiable claims,
not much to say", the system can report "the article's central thesis
(X) rests on premises that independent sources only weakly support."

Premise verifications are held *inside* the returned OpinionGrounding
objects and never join the article-level verification list, so they
don't inflate factuality/coverage.
"""

from __future__ import annotations

import asyncio
from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential

from backend.config import settings
from backend.models import ClaimVerification, OpinionGrounding, OpinionPremise
from backend.reliability.claims import _parse_json, verify_all_claims
from backend.reliability.content_extractor import ExtractedContent
from backend.reliability.llm_client import chat_json

MAX_OPINIONS = 4
MAX_PREMISES_PER_OPINION = 3
MAX_INPUT_CHARS = 12_000


OPINION_SYSTEM_PROMPT = """You are a media-analysis assistant. Identify
the central OPINIONS the article advances — NOT its verifiable facts
(those are handled separately).

An OPINION is a contestable judgment, prediction, recommendation, or
value-laden interpretation the author is arguing for. Examples:
"the policy will backfire", "the party is overconfident", "this ruling
threatens democracy". These cannot be directly fact-checked, but they
REST ON factual premises that can.

For each opinion, list the 1-3 FACTUAL PREMISES it depends on, phrased
as atomic, independently web-verifiable claims (name the specific actor,
quantity, date verbatim where present). Only include premises that could
plausibly be confirmed or denied by public web evidence — skip premises
that reference unpublished studies, private conversations, or the
author's own subjective experience.

Extract AT MOST 4 opinions, the ones most central to the article's
argument. If the article is straight news reporting with no real
opinions, return an empty list.

Return STRICT JSON only:

{
  "opinions": [
    {
      "opinion": "<the contestable assertion, in the author's framing>",
      "stance_summary": "<one neutral sentence paraphrasing what the author claims>",
      "premises": ["<atomic verifiable premise>", "..."]
    }
  ]
}
"""


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
def _call_extract_opinions(prompt: str) -> str:
    return chat_json(OPINION_SYSTEM_PROMPT, prompt, temperature=0.2, json_mode=True)


async def _extract_opinions(content: ExtractedContent) -> list[dict[str, Any]]:
    """One LLM call → a list of {opinion, stance_summary, premises[]} dicts.

    Returns [] when the LLM is unavailable, the article has no text, or
    extraction fails / yields nothing usable.
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
    try:
        raw = await asyncio.to_thread(_call_extract_opinions, prompt)
        data = _parse_json(raw)
    except Exception:  # pragma: no cover - network/parse failures expected in dev
        return []

    raw_opinions = data.get("opinions") or []
    if not isinstance(raw_opinions, list):
        return []

    cleaned: list[dict[str, Any]] = []
    for item in raw_opinions[:MAX_OPINIONS]:
        if not isinstance(item, dict):
            continue
        opinion = str(item.get("opinion") or "").strip()
        if len(opinion) < 8:
            continue
        stance = str(item.get("stance_summary") or "").strip()[:280]
        raw_premises = item.get("premises") or []
        premises: list[str] = []
        if isinstance(raw_premises, list):
            for p in raw_premises[:MAX_PREMISES_PER_OPINION]:
                if isinstance(p, str) and len(p.strip()) >= 12:
                    premises.append(p.strip()[:280])
        cleaned.append(
            {"opinion": opinion[:400], "stance_summary": stance, "premises": premises}
        )
    return cleaned


def _grounding_verdict(premise_vers: list[ClaimVerification]) -> tuple[str, float]:
    """Map an opinion's premise verifications to a (verdict, score) pair.

    Score = mean premise score (or neutral 50 when no premises). Verdict
    thresholds are calibrated to the same conservative-NLI regime as the
    per-claim status thresholds:

    * contradicted  → any premise verified as "contradicted"
    * well-grounded → score >= 70 and at least one premise "supported"
    * weakly-grounded → 45 <= score < 70
    * unsupported   → score < 45, or every premise unverified
    """
    if not premise_vers:
        return "unsupported", 50.0

    if any(v.status == "contradicted" for v in premise_vers):
        score = sum(v.score for v in premise_vers) / len(premise_vers)
        return "contradicted", round(score, 1)

    score = sum(v.score for v in premise_vers) / len(premise_vers)
    n_supported = sum(1 for v in premise_vers if v.status == "supported")
    all_unverified = all(v.status == "unverified" for v in premise_vers)

    if all_unverified:
        verdict = "unsupported"
    elif score >= 70 and n_supported >= 1:
        verdict = "well-grounded"
    elif score >= 45:
        verdict = "weakly-grounded"
    else:
        verdict = "unsupported"
    return verdict, round(score, 1)


async def ground_opinions(
    content: ExtractedContent, *, exclude_domain: str | None = None
) -> list[OpinionGrounding]:
    """Extract the article's opinions and assess each one's grounding.

    All premises across all opinions are verified in a SINGLE
    ``verify_all_claims`` batch (concurrent, bounded) and then mapped
    back to their owning opinion. Premise verifications live inside the
    returned objects — they are never returned to the article-level
    aggregation, so they cannot inflate factuality/coverage.
    """
    extracted = await _extract_opinions(content)
    if not extracted:
        return []

    # Flatten premises, remembering which opinion each belongs to.
    flat_premises: list[str] = []
    owner: list[int] = []  # index into `extracted` for each flat premise
    for i, op in enumerate(extracted):
        for premise in op["premises"]:
            flat_premises.append(premise)
            owner.append(i)

    # Verify all premises at once (skips cleanly to [] if none).
    verifications: list[ClaimVerification] = (
        await verify_all_claims(flat_premises, exclude_domain=exclude_domain)
        if flat_premises
        else []
    )

    # Bucket verifications back to their opinion.
    by_opinion: list[list[ClaimVerification]] = [[] for _ in extracted]
    for v, idx in zip(verifications, owner):
        by_opinion[idx].append(v)

    groundings: list[OpinionGrounding] = []
    for i, op in enumerate(extracted):
        premise_vers = by_opinion[i]
        verdict, score = _grounding_verdict(premise_vers)
        groundings.append(
            OpinionGrounding(
                opinion=op["opinion"],
                stance_summary=op["stance_summary"],
                premises=[
                    OpinionPremise(text=v.claim, verification=v) for v in premise_vers
                ],
                grounding_score=score,
                verdict=verdict,  # type: ignore[arg-type]
            )
        )
    return groundings


def opinion_red_flags(groundings: list[OpinionGrounding]) -> list[str]:
    """Red flags for opinions whose factual premises don't hold up.

    Surfaced as article-level red flags so an op-ed whose central
    argument rests on unsupported/contradicted premises is flagged even
    though it produced few verifiable *facts*.
    """
    flags: list[str] = []
    for g in groundings:
        if g.verdict == "contradicted":
            flags.append(
                f"The article's argument that \"{g.opinion[:120]}\" rests on a premise "
                "that independent sources contradict."
            )
        elif g.verdict == "unsupported":
            flags.append(
                f"The article's argument that \"{g.opinion[:120]}\" is not supported by "
                "verifiable evidence — its factual premises could not be corroborated."
            )
    return flags
