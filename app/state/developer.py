"""Developer workspace state."""

from typing import Any

from pydantic import Field

from app.state.base import BaseGenAIState


class DeveloperState(BaseGenAIState):
    """State for developer-facing workflows."""

    codebase_context: dict[str, Any] = Field(default_factory=dict)
    diagnostic_context: dict[str, Any] = Field(default_factory=dict)