"""Internal request context for GenAI workflows."""

from pydantic import BaseModel, ConfigDict

from app.models.request import GenAIRequest
from app.models.user_context import UserContext


class RequestContext(BaseModel):
    """Combines user identity and the incoming GenAI request."""

    model_config = ConfigDict(frozen=True)

    user: UserContext
    request: GenAIRequest
