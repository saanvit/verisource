"""Unified LLM client supporting OpenRouter and Mistral.

The rest of the codebase calls ``chat_json(system, user)`` and doesn't
know or care which provider it's talking to. Provider selection is
driven by env vars in ``backend.config.settings``:

* ``OPENROUTER_API_KEY`` set → route through OpenRouter (preferred when
  configured because OpenRouter gives access to frontier models like
  ``anthropic/claude-sonnet-4.6`` and ``anthropic/claude-opus-4.8``).
* Otherwise, falls back to direct Mistral if ``MISTRAL_API_KEY`` set.
* Otherwise, raises ``RuntimeError`` — callers should check
  ``settings.has_llm`` before calling.

Both backends speak the OpenAI-style chat-completions schema, so this
module is intentionally thin: just request shaping + JSON-string
extraction. Parsing the JSON content is the caller's job (and existing
sites already handle markdown fences and regex extraction).
"""

from __future__ import annotations

import httpx

from backend.config import settings

OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"


def _call_openrouter(
    system: str, user: str, temperature: float, json_mode: bool, max_tokens: int
) -> str:
    """One synchronous request to OpenRouter. Returns the model's text content.

    Raises ``httpx.HTTPError`` on transport failures and ``ValueError`` if
    the response shape is unexpected — callers wrap this in their existing
    retry + fallback logic.
    """
    headers = {
        "Authorization": f"Bearer {settings.openrouter_api_key}",
        "Content-Type": "application/json",
        # OpenRouter requests these for app attribution; not auth-critical
        # but improves dashboard analytics + helps with rate limit fairness.
        "HTTP-Referer": "https://github.com/saanvit/verisource",
        "X-Title": "VeriSource (CS153)",
    }
    body: dict = {
        "model": settings.openrouter_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        # Cap output length. Without this, OpenRouter assumes the model's full
        # context as max_tokens and its credit-affordability check rejects the
        # request (HTTP 402) even though these JSON-extraction tasks only emit
        # a few hundred tokens. Also keeps per-call cost predictable.
        "max_tokens": max_tokens,
    }
    if json_mode:
        # Many but not all OpenRouter-served models support this. For models
        # that don't, the field is silently ignored and the model is still
        # JSON-conditioned by the system prompt — callers tolerate either
        # path via _parse_llm_json's markdown-fence stripping.
        body["response_format"] = {"type": "json_object"}

    resp = httpx.post(OPENROUTER_ENDPOINT, json=body, headers=headers, timeout=60.0)
    resp.raise_for_status()
    data = resp.json()
    try:
        content = data["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"OpenRouter returned unexpected payload: {data!r}") from exc
    return content


def _call_mistral(
    system: str, user: str, temperature: float, json_mode: bool, max_tokens: int
) -> str:
    """One synchronous request to Mistral via the official SDK."""
    from mistralai import Mistral  # local import — avoids importing when unused

    client = Mistral(api_key=settings.mistral_api_key)
    kwargs: dict = {
        "model": settings.mistral_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    response = client.chat.complete(**kwargs)
    return response.choices[0].message.content or ""


def chat_json(
    system: str,
    user: str,
    *,
    temperature: float = 0.1,
    json_mode: bool = True,
    max_tokens: int = 4096,
) -> str:
    """Send a system+user prompt to whichever LLM is configured.

    Returns the raw model output as a string. Callers are responsible
    for parsing JSON (existing call sites use ``_parse_llm_json`` which
    handles markdown fences and stray prose).

    Routing: OpenRouter is preferred when ``OPENROUTER_API_KEY`` is set,
    because that's where the CS153 credits live and it gives access to
    Claude Sonnet/Opus, which are noticeably better than Mistral
    on structured-extraction tasks. Falls back to Mistral otherwise.
    """
    provider = settings.llm_provider
    if provider == "openrouter":
        return _call_openrouter(system, user, temperature, json_mode, max_tokens)
    if provider == "mistral":
        return _call_mistral(system, user, temperature, json_mode, max_tokens)
    raise RuntimeError(
        "No LLM configured. Set OPENROUTER_API_KEY or MISTRAL_API_KEY in your .env."
    )


def active_model_name() -> str:
    """For logging/UX: which model is actually being called right now."""
    if settings.llm_provider == "openrouter":
        return f"openrouter:{settings.openrouter_model}"
    if settings.llm_provider == "mistral":
        return f"mistral:{settings.mistral_model}"
    return "none"
