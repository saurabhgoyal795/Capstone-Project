"""Application configuration loaded from environment variables (.env supported)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SUPPORTED_PROVIDERS = ("openai", "gemini", "ollama")

DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "gemini": "gemini-2.0-flash",
    "ollama": "llama3.1",
}


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


@dataclass(frozen=True)
class Settings:
    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "openai").lower())
    model_name: str = field(default_factory=lambda: _env("MODEL_NAME", ""))
    temperature: float = field(default_factory=lambda: float(_env("LLM_TEMPERATURE", "0.1")))
    request_timeout: int = field(default_factory=lambda: int(_env("LLM_TIMEOUT_SECONDS", "120")))
    max_retries: int = field(default_factory=lambda: int(_env("LLM_MAX_RETRIES", "2")))
    ollama_base_url: str = field(default_factory=lambda: _env("OLLAMA_BASE_URL", "http://localhost:11434"))
    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / _env("DATA_DIR", "data"))
    output_dir: Path = field(default_factory=lambda: PROJECT_ROOT / _env("OUTPUT_DIR", "output"))
    max_workers: int = field(default_factory=lambda: int(_env("MAX_WORKERS", "3")))
    max_document_chars: int = field(default_factory=lambda: int(_env("MAX_DOCUMENT_CHARS", "12000")))
    log_level: str = field(default_factory=lambda: _env("LOG_LEVEL", "INFO").upper())
    supported_extensions: tuple[str, ...] = (".txt", ".pdf", ".docx")

    @property
    def resolved_model(self) -> str:
        return self.model_name or DEFAULT_MODELS.get(self.llm_provider, "")

    def validate(self) -> None:
        if self.llm_provider not in SUPPORTED_PROVIDERS:
            raise ValueError(
                f"Unsupported LLM_PROVIDER '{self.llm_provider}'. Choose one of {SUPPORTED_PROVIDERS}."
            )
        if self.llm_provider == "openai" and not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY is not set. Add it to .env or switch LLM_PROVIDER.")
        if self.llm_provider == "gemini" and not os.getenv("GOOGLE_API_KEY"):
            raise ValueError("GOOGLE_API_KEY is not set. Add it to .env or switch LLM_PROVIDER.")
        if self.max_workers < 1:
            raise ValueError("MAX_WORKERS must be >= 1.")


def get_settings() -> Settings:
    return Settings()
