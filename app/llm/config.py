"""Minimal environment configuration for the LLM provider."""

import os
from dataclasses import dataclass


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
