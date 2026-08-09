"""Request models for the GenAI Layer."""

from pydantic import BaseModel, ConfigDict, Field


class GenAIRequest(BaseModel):
    """Represents a request entering the GenAI Layer."""

    model_config = ConfigDict(frozen=True)

    message: str = Field(min_length=1)
    