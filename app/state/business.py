"""Business workspace state."""

from typing import Any

from pydantic import Field

from app.models.business_insight import BusinessInsight
from app.state.base import BaseGenAIState


class BusinessState(BaseGenAIState):
    """State for business-user workflows."""

    business_context: dict[str, Any] = Field(default_factory=dict)
    insight_context: dict[str, Any] = Field(default_factory=dict)
    insight: BusinessInsight | None = None
