"""Safe request and response models exposed by the GenAI HTTP API."""

from pydantic import BaseModel, ConfigDict, Field

from app.models.business_insight import BusinessInsight
from app.routing.persona import Persona
from app.routing.workspace import Workspace


class ApplicationGenAIRequest(BaseModel):
    """The minimal authenticated request accepted by the application boundary."""

    model_config = ConfigDict(frozen=True)

    user_id: str = Field(min_length=1)
    persona: Persona
    message: str = Field(min_length=1)


class BusinessApplicationResponse(BaseModel):
    """Safe, structured result for the currently supported Business capability."""

    model_config = ConfigDict(frozen=True)

    persona: Persona
    workspace: Workspace
    insight: BusinessInsight


class ApplicationErrorDetail(BaseModel):
    """Safe public error information correlated to one HTTP request."""

    code: str
    message: str
    request_id: str


class ApplicationErrorResponse(BaseModel):
    """Typed envelope used for every handled application error."""

    error: ApplicationErrorDetail
