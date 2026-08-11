"""Customer workspace state."""

from typing import Any

from pydantic import Field

from app.state.base import BaseGenAIState


class CustomerState(BaseGenAIState):
    """State for customer-facing workflows."""

    shopping_context: dict[str, Any] = Field(default_factory=dict)
    recommendation_context: dict[str, Any] = Field(default_factory=dict)