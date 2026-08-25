"""Minimal environment configuration for the LLM provider."""

import os
from dataclasses import dataclass

from app.errors import LLMConfigurationError


@dataclass(frozen=True)
class OpenAISettings:
    """Settings required by the OpenAI Responses API adapter."""

    api_key: str
    model: str = "gpt-4o-mini"

    @classmethod
    def from_environment(cls) -> "OpenAISettings":
        """Load provider settings without exposing credentials in code."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY must be set to use the OpenAI LLM.")
        return cls(api_key=api_key, model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))


@dataclass(frozen=True)
class OpenRouterSettings:
    """Settings required by the OpenRouter OpenAI-compatible adapter."""

    api_key: str
    model: str

    @classmethod
    def from_environment(cls) -> "OpenRouterSettings":
        """Load the selected OpenRouter configuration without revealing its secret."""
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key or not api_key.strip():
            raise LLMConfigurationError("OPENROUTER_API_KEY must be set to use the OpenRouter LLM.")

        model = os.getenv("OPENROUTER_MODEL")
        if not model or not model.strip():
            raise LLMConfigurationError("OPENROUTER_MODEL must be set to use the OpenRouter LLM.")

        return cls(api_key=api_key, model=model.strip())
