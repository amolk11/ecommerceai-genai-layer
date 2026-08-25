"""Construction for concrete LLM providers."""

from app.llm.config import OpenAISettings
from app.llm.openai import OpenAIResponsesBusinessLLM
from app.llm.protocols import BusinessLLM


def create_business_llm(settings: OpenAISettings | None = None) -> BusinessLLM:
    """Create the configured business LLM provider."""
    return OpenAIResponsesBusinessLLM(settings or OpenAISettings.from_environment())
