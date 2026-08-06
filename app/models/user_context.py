"""User context model."""

from pydantic import BaseModel, ConfigDict

from app.routing.persona import Persona


class UserContext(BaseModel):
    """
    Represents the authenticated user context.

    This object is passed through the GenAI workflow and provides
    information about the current user and their persona.
    """

    model_config = ConfigDict(frozen=True)

    user_id: str
    persona: Persona
    