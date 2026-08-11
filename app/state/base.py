"""Base state definitions for GenAI workflows."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.models.request_context import RequestContext
from app.routing.workspace import Workspace


class BaseGenAIState(BaseModel):
    """Shared state carried by all GenAI workflows."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    context: RequestContext
    workspace: Workspace

    messages: list[dict[str, Any]] = Field(default_factory=list)

    intent: str | None = None

    current_step: str | None = None

    tool_results: dict[str, Any] = Field(default_factory=dict)

    final_response: str | None = None

    error: str | None = None