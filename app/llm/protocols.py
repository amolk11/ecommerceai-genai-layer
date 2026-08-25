"""Contracts for language model capabilities."""

from typing import Protocol, runtime_checkable

from app.models.business_insight import BusinessInsight


@runtime_checkable
class BusinessLLM(Protocol):
    """Generates a structured insight from a business prompt."""

    def generate_business_insight(self, prompt: str) -> BusinessInsight:
        """Return one validated business insight."""
        ...
