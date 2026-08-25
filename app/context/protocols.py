"""Contracts for workflow context construction."""

from typing import Protocol

from app.models.business_context import BusinessContext


class BusinessContextProvider(Protocol):
    """Build a bounded business context without calling an LLM."""

    def build(self, user_id: str) -> BusinessContext:
        """Return data relevant to the requesting business user."""
        ...
