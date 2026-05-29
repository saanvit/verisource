"""Centralized configuration loaded from environment."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


@dataclass(frozen=True)
class Settings:
    mistral_api_key: str | None
    mistral_model: str
    openrouter_api_key: str | None
    openrouter_model: str
    tavily_api_key: str | None
    cross_reference_k: int
    host: str
    port: int

    @property
    def has_llm(self) -> bool:
        return bool(self.openrouter_api_key) or bool(self.mistral_api_key)

    @property
    def llm_provider(self) -> str | None:
        """Which LLM backend ``llm_client.chat_json`` will route to.

        OpenRouter takes priority when configured because the CS153
        credits live there and it gives access to frontier models
        (Claude 3.5 Sonnet, GPT-4o, etc.). Falls back to Mistral.
        """
        if self.openrouter_api_key:
            return "openrouter"
        if self.mistral_api_key:
            return "mistral"
        return None

    @property
    def has_search(self) -> bool:
        return bool(self.tavily_api_key)


def load_settings() -> Settings:
    return Settings(
        mistral_api_key=os.getenv("MISTRAL_API_KEY") or None,
        mistral_model=os.getenv("MISTRAL_MODEL", "mistral-large-latest"),
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY") or None,
        openrouter_model=os.getenv("OPENROUTER_MODEL", "anthropic/claude-3.5-sonnet"),
        tavily_api_key=os.getenv("TAVILY_API_KEY") or None,
        cross_reference_k=int(os.getenv("CROSS_REFERENCE_K", "5")),
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
    )


settings = load_settings()
