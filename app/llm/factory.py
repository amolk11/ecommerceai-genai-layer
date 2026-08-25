"""Construction for concrete LLM providers."""

import os

from app.errors import LLMConfigurationError
from app.llm.config import OpenAISettings, OpenRouterSettings
from app.llm.protocols import BusinessLLM


def create_business_llm(
    settings: OpenAISettings | OpenRouterSettings | None = None,
    provider: str | None = None,
) -> BusinessLLM:
    """Create the selected provider only when application composition requires it."""
    selected_provider = _provider_name(settings, provider)

    if selected_provider == "openrouter":
        from app.llm.openrouter import OpenRouterBusinessLLM

        configured = settings if isinstance(settings, OpenRouterSettings) else OpenRouterSettings.from_environment()
        return OpenRouterBusinessLLM(configured)

    if selected_provider == "openai":
        from app.llm.openai import OpenAIResponsesBusinessLLM

        configured = settings if isinstance(settings, OpenAISettings) else OpenAISettings.from_environment()
        return OpenAIResponsesBusinessLLM(configured)

    raise LLMConfigurationError(f"Unsupported LLM_PROVIDER: {selected_provider}.")


def _provider_name(settings: OpenAISettings | OpenRouterSettings | None, provider: str | None) -> str:
    """Resolve an explicit provider while retaining the original OpenAI settings API."""
    if isinstance(settings, OpenAISettings):
        return "openai"
    if isinstance(settings, OpenRouterSettings):
        return "openrouter"
    return (provider or os.getenv("LLM_PROVIDER", "openrouter")).strip().lower()
