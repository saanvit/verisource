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
    tavily_api_key: str | None
    cross_reference_k: int
    host: str
    port: int

    @property
    def has_llm(self) -> bool:
        return bool(self.mistral_api_key)

    @property
    def has_search(self) -> bool:
        return bool(self.tavily_api_key)


def load_settings() -> Settings:
    return Settings(
        mistral_api_key=os.getenv("MISTRAL_API_KEY") or None,
        mistral_model=os.getenv("MISTRAL_MODEL", "mistral-large-latest"),
        tavily_api_key=os.getenv("TAVILY_API_KEY") or None,
        cross_reference_k=int(os.getenv("CROSS_REFERENCE_K", "5")),
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
    )


settings = load_settings()
