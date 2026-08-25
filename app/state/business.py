"""Business workspace state."""

from app.models.business_context import BusinessContext
from app.models.business_insight import BusinessInsight
from app.state.base import BaseGenAIState


class BusinessState(BaseGenAIState):
    """State for business-user workflows."""

    business_context: BusinessContext | None = None
    insight: BusinessInsight | None = None
