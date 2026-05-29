"""Self-reflective verification agent.

Wraps the single-shot per-claim pipeline in a reflection loop where the
LLM (Claude Sonnet 4.6 via OpenRouter, by default) audits its own
intermediate output and emits structured fix-actions that the system
executes — then re-aggregates.

This addresses three failure modes the single-shot pipeline can't catch
on its own:

1. **Mislabeled hits** — NLI labelers sometimes mark obviously-supporting
   evidence as ``unclear`` due to hedging language, domain mismatch, or
   partial overlap (the SB17 / Garces failure case is the canonical
   example).
2. **Weak retrieval queries** — generic claim phrasings return either
   zero evidence or only tangentially-related hits.
3. **Compound claims that slipped past decomposition** — a single
   sentence packing two assertions gets one ``unverified`` verdict
   instead of two separable ones.

The loop is bounded (default 2 rounds), structured (critique returns
JSON-typed actions, never free-form), and idempotent (each action takes
the current ``verifications`` list and returns a new one — no in-place
mutation). The full trace is returned alongside the updated
verifications so callers can persist it on the ``ReliabilityReport``
for the UI and for downstream analysis.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from backend.models import (
    ClaimEvidence,
    ClaimVerification,
    ReflectionAction,
    ReflectionRound,
)
from backend.reliability.claims import (
    _build_verification,
    _label_hits,
    recompute_verification_after_relabel,
    verify_all_claims,
)
from backend.reliability.content_extractor import ExtractedContent
from backend.reliability.llm_client import chat_json
from backend.search.web_search import web_search

# ---------- critique prompt ----------

CRITIQUE_SYSTEM_PROMPT = """You audit fact-check verification results.
You will see a JSON array of CLAIMS, each with retrieved EVIDENCE and the
per-hit stance label that an automated NLI model assigned. Your job is to
identify cases where the labeling or retrieval likely went wrong and
propose targeted, structured fixes.

Look specifically for these failure patterns (in priority order):

1. CLAIMS with status="unverified" or "mixed" DESPITE having 2 or more
   high-quality hits (domain_score >= 75) whose snippets clearly affirm
   the claim. The NLI labeler is being too conservative.
     → emit a "relabel_hit" action with new_label="supports" for each
       hit whose snippet directly affirms the claim.

2. HITS labeled "contradicts" whose snippet does NOT actually contradict
   the claim — it just discusses a related but different aspect or uses
   hedging language. (Distinguish "different aspect" from "incompatible
   fact": only the latter is a real contradiction.)
     → "relabel_hit" with new_label="unclear" (or "supports" if it
       actually supports).

3. CLAIMS with n_evidence=0 due to overly-generic queries.
     → "research_claim" with a more specific alternative_query that
       includes named entities, dates, or institution names.

4. CLAIMS that contain "and"/"while"/"as well as"/multiple verb phrases
   joining two independent factual assertions.
     → "split_claim" with subclaims listing each atom.

5. Inconsistent labels across two hits whose snippets say essentially
   the same thing.
     → "relabel_hit" on the inconsistent one to match the consensus.

Output STRICT JSON ONLY (no markdown fences, no prose before or after):

{
  "critique": "1-2 sentence overall assessment of the verification quality",
  "issues": ["short tag per problem you found", ...],
  "actions": [
    {
      "type": "relabel_hit",
      "claim": "exact claim text from input",
      "hit_url": "exact URL from input",
      "new_label": "supports" | "contradicts" | "unclear",
      "reason": "1 sentence why"
    },
    {
      "type": "research_claim",
      "claim": "exact claim text from input",
      "alternative_query": "<more targeted Tavily query>",
      "reason": "1 sentence why"
    },
    {
      "type": "split_claim",
      "claim": "exact claim text from input",
      "subclaims": ["atomic claim A", "atomic claim B"],
      "reason": "1 sentence why"
    }
  ]
}

Return AT MOST 4 actions per round. If the verifications look fine,
return actions=[]. Do not invent claims or URLs that aren't in the
input.
"""


# ---------- LLM plumbing (mirrors claims._parse_json) ----------


def _parse_critique_json(raw: str) -> dict[str, Any]:
    raw = raw.strip().strip("`")
    if raw.startswith("json"):
        raw = raw[4:].lstrip()
    match = re.search(r"\{.*\}", raw, re.S)
    if not match:
        raise ValueError("critique LLM did not return JSON")
    return json.loads(match.group(0))


def _serialize_verifications(verifications: list[ClaimVerification]) -> str:
    """Compact serialization of the verifications array for the critique
    LLM. Includes everything the critic needs to act on: claim text,
    status, score, and each evidence hit with URL + snippet + current
    label + domain reputation."""
    payload = []
    for v in verifications:
        payload.append(
            {
                "claim": v.claim,
                "status": v.status,
                "score": v.score,
                "support_ratio": v.support_ratio,
                "contradict_ratio": v.contradict_ratio,
                "n_evidence": v.n_evidence,
                "n_high_quality": v.n_high_quality,
                "evidence": [
                    {
                        "url": e.url,
                        "title": e.title[:200],
                        "domain": e.domain,
                        "domain_score": e.domain_score,
                        "snippet": e.snippet[:600],
                        "current_label": e.agreement,
                    }
                    for e in v.evidence
                ],
            }
        )
    return json.dumps(payload, ensure_ascii=False)


def _call_critique_llm(verifications_payload: str) -> str:
    """Thin wrapper so tests can monkeypatch this single function."""
    return chat_json(
        CRITIQUE_SYSTEM_PROMPT,
        f"VERIFICATIONS:\n{verifications_payload}\n\nReturn JSON only.",
        temperature=0.1,
        json_mode=True,
    )


async def _get_critique(verifications: list[ClaimVerification]) -> dict[str, Any]:
    """Run one critique pass. Returns the parsed JSON dict or raises."""
    payload = _serialize_verifications(verifications)
    raw = await asyncio.to_thread(_call_critique_llm, payload)
    return _parse_critique_json(raw)


# ---------- action executors ----------


def _find_verification_index(
    verifications: list[ClaimVerification], claim_text: str
) -> int | None:
    """Find a verification by claim text. Tolerant of minor whitespace
    differences so the LLM doesn't have to round-trip the claim verbatim
    to the byte."""
    if not claim_text:
        return None
    target = " ".join(claim_text.strip().split()).lower()
    for i, v in enumerate(verifications):
        candidate = " ".join(v.claim.strip().split()).lower()
        if candidate == target:
            return i
    # Fall back: prefix match (handles trailing punctuation drift)
    for i, v in enumerate(verifications):
        candidate = " ".join(v.claim.strip().split()).lower()
        if candidate.startswith(target[:60]) or target.startswith(candidate[:60]):
            return i
    return None


def _action_relabel_hit(
    verifications: list[ClaimVerification], action: ReflectionAction
) -> tuple[list[ClaimVerification], bool]:
    """Apply a single ``relabel_hit`` action. Returns (new_list, changed)."""
    if not action.claim or not action.hit_url or not action.new_label:
        return verifications, False
    idx = _find_verification_index(verifications, action.claim)
    if idx is None:
        return verifications, False
    v = verifications[idx]
    new_evidence: list[ClaimEvidence] = []
    found = False
    for e in v.evidence:
        if e.url == action.hit_url and e.agreement != action.new_label:
            new_evidence.append(e.model_copy(update={"agreement": action.new_label}))
            found = True
        else:
            new_evidence.append(e)
    if not found:
        return verifications, False
    updated = recompute_verification_after_relabel(
        v.model_copy(update={"evidence": new_evidence})
    )
    new_list = list(verifications)
    new_list[idx] = updated
    return new_list, True


async def _action_research_claim(
    verifications: list[ClaimVerification],
    action: ReflectionAction,
    *,
    exclude_domain: str | None,
) -> tuple[list[ClaimVerification], bool]:
    """Apply a single ``research_claim`` action — re-search with the
    suggested alt query, then replace the existing verification."""
    if not action.claim or not action.alternative_query:
        return verifications, False
    idx = _find_verification_index(verifications, action.claim)
    if idx is None:
        return verifications, False
    try:
        hits = await web_search(
            action.alternative_query, k=4, exclude_domain=exclude_domain
        )
    except Exception:
        return verifications, False
    if not hits:
        # No hits → keep the existing verification untouched. We didn't
        # improve anything, but the trace records that we tried.
        return verifications, False
    labels = await _label_hits(action.claim, list(hits))
    new_v = _build_verification(action.claim, list(hits), list(labels))
    new_list = list(verifications)
    new_list[idx] = new_v
    return new_list, True


async def _action_split_claim(
    verifications: list[ClaimVerification],
    action: ReflectionAction,
    *,
    exclude_domain: str | None,
) -> tuple[list[ClaimVerification], bool]:
    """Apply a single ``split_claim`` action — verify each subclaim
    independently, remove the original, append the new ones."""
    if not action.claim or not action.subclaims or len(action.subclaims) < 2:
        return verifications, False
    idx = _find_verification_index(verifications, action.claim)
    if idx is None:
        return verifications, False
    # Cap at 3 splits to bound cost.
    subclaims = [s.strip() for s in action.subclaims if s and s.strip()][:3]
    if len(subclaims) < 2:
        return verifications, False
    try:
        new_verifications = await verify_all_claims(
            subclaims, exclude_domain=exclude_domain
        )
    except Exception:
        return verifications, False
    new_list = list(verifications)
    del new_list[idx]
    new_list.extend(new_verifications)
    return new_list, True


# ---------- main loop ----------


async def _execute_actions(
    verifications: list[ClaimVerification],
    actions: list[ReflectionAction],
    *,
    exclude_domain: str | None,
) -> list[ClaimVerification]:
    """Apply actions sequentially. Order matters because later actions
    may target verifications whose claim text was changed by earlier
    actions (especially split_claim). We accept that and just apply in
    the order the critic returned them."""
    current = verifications
    for action in actions:
        if action.type == "relabel_hit":
            current, _ = _action_relabel_hit(current, action)
        elif action.type == "research_claim":
            current, _ = await _action_research_claim(
                current, action, exclude_domain=exclude_domain
            )
        elif action.type == "split_claim":
            current, _ = await _action_split_claim(
                current, action, exclude_domain=exclude_domain
            )
        # "noop" intentionally does nothing
    return current


def _coerce_actions(raw_actions: Any) -> list[ReflectionAction]:
    """Parse the LLM's ``actions`` list into typed models. Skip
    individual actions that fail validation rather than aborting the
    whole round."""
    out: list[ReflectionAction] = []
    if not isinstance(raw_actions, list):
        return out
    for item in raw_actions[:4]:  # belt-and-suspenders cap matching the prompt
        if not isinstance(item, dict):
            continue
        try:
            out.append(ReflectionAction(**item))
        except Exception:  # pydantic ValidationError, etc.
            continue
    return out


def _score_deltas(
    before: list[ClaimVerification], after: list[ClaimVerification]
) -> dict[str, float]:
    """Per-claim score change after a round. Uses claim text as the key;
    when split_claim adds new claims, their delta is recorded as the
    new claim's score minus 50 (the neutral prior they replaced)."""
    before_by_claim = {v.claim: v.score for v in before}
    deltas: dict[str, float] = {}
    after_claims = {v.claim for v in after}
    for v in after:
        old = before_by_claim.get(v.claim)
        if old is None:
            # Newly-added (split_claim) — compare against neutral prior.
            delta = v.score - 50.0
        else:
            delta = v.score - old
        if abs(delta) >= 0.05:
            deltas[v.claim] = round(delta, 1)
    # Removed claims (split_claim originals) show up as their old score
    # being "erased". Record as 0-old to flag the disappearance.
    for claim, old in before_by_claim.items():
        if claim not in after_claims:
            deltas[claim] = round(-old, 1)
    return deltas


async def run_reflection_loop(
    verifications: list[ClaimVerification],
    *,
    content: ExtractedContent,  # noqa: ARG001 - reserved for richer critique
    exclude_domain: str | None,
    max_rounds: int = 2,
) -> tuple[list[ClaimVerification], list[ReflectionRound]]:
    """Run up to ``max_rounds`` of critique → execute → re-aggregate.

    Returns the (possibly-updated) verifications list and the full
    reflection trace. The loop terminates early when the critic returns
    no actions, when no score moved by more than 5 points in a round, or
    when the critique call fails (we record a noop round and exit).
    """
    trace: list[ReflectionRound] = []
    current = list(verifications)

    for round_index in range(max_rounds):
        # 1. Get critique. If the LLM call fails, record a noop and exit.
        try:
            critique_json = await _get_critique(current)
        except Exception as exc:
            trace.append(
                ReflectionRound(
                    round_index=round_index,
                    critique=f"Critique call failed: {type(exc).__name__}",
                    issues=["critique_call_failed"],
                    actions=[],
                    score_deltas={},
                )
            )
            break

        critique_text = str(critique_json.get("critique") or "").strip()[:1000]
        issues_raw = critique_json.get("issues") or []
        issues = [
            str(i).strip()[:120]
            for i in (issues_raw if isinstance(issues_raw, list) else [])
            if i
        ][:8]
        actions = _coerce_actions(critique_json.get("actions"))

        # 2. Execute actions (if any).
        before = list(current)
        current = await _execute_actions(
            current, actions, exclude_domain=exclude_domain
        )
        deltas = _score_deltas(before, current)

        trace.append(
            ReflectionRound(
                round_index=round_index,
                critique=critique_text or "(no critique text)",
                issues=issues,
                actions=actions,
                score_deltas=deltas,
            )
        )

        # 3. Convergence checks.
        if not actions:
            break
        if not deltas:
            # Actions executed but nothing actually changed — stop.
            break
        # Significant-change check: if every delta is < 5 points, stop.
        if all(abs(d) < 5.0 for d in deltas.values()):
            break

    return current, trace
