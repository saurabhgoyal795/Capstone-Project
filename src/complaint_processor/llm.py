"""LLM factory: returns a LangChain chat model for the configured provider.

Provider packages are imported lazily so that only the one actually in use needs to be
installed/configured (e.g. running fully local with Ollama needs no API keys).
"""

from __future__ import annotations

import logging

from langchain_core.language_models import BaseChatModel

from .config import Settings

logger = logging.getLogger(__name__)


def get_llm(settings: Settings) -> BaseChatModel:
    """Build a chat model for ``settings.llm_provider`` (openai | gemini | ollama | bedrock)."""
    settings.validate()
    provider = settings.llm_provider
    model = settings.resolved_model

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        llm: BaseChatModel = ChatOpenAI(
            model=model,
            temperature=settings.temperature,
            timeout=settings.request_timeout,
            max_retries=settings.max_retries,
        )
    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=settings.temperature,
            timeout=settings.request_timeout,
            max_retries=settings.max_retries,
        )
    elif provider == "ollama":
        from langchain_ollama import ChatOllama

        llm = ChatOllama(
            model=model,
            base_url=settings.ollama_base_url,
            temperature=settings.temperature,
            # Disable "thinking" output on reasoning models (qwen3, glm-4.x, deepseek-r1 ...):
            # it is much faster and keeps <think> blocks out of the JSON we parse.
            reasoning=False,
            num_ctx=8192,
            client_kwargs={"timeout": settings.request_timeout},
        )
    elif provider == "bedrock":
        from botocore.config import Config
        from langchain_aws import ChatBedrockConverse

        # Credentials come from the standard AWS chain (profile locally, IAM role on EC2).
        llm = ChatBedrockConverse(
            model=model,
            region_name=settings.aws_region,
            temperature=settings.temperature,
            config=Config(read_timeout=settings.request_timeout,
                          retries={"max_attempts": settings.max_retries + 1, "mode": "standard"}),
        )
    else:  # pragma: no cover - validate() already guards this
        raise ValueError(f"Unsupported provider: {provider}")

    logger.info("LLM initialised: provider=%s model=%s temperature=%s", provider, model, settings.temperature)
    return llm
