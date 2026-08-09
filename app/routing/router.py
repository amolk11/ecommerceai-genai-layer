"""Persona-based request routing."""

from app.models.user_context import UserContext
from app.routing.persona import Persona
from app.routing.workspace import Workspace


class PersonaRouter:
    """Routes users to the appropriate GenAI workspace."""

    _WORKSPACE_MAP = {
        Persona.BUSINESS: Workspace.BUSINESS,
        Persona.CUSTOMER: Workspace.CUSTOMER,
        Persona.DEVELOPER: Workspace.DEVELOPER,
    }

    def route(self, context: UserContext) -> Workspace:
        """
        Return the workspace associated with the user's persona.

        Args:
            context: Authenticated user context.

        Returns:
            Workspace assigned to the user's persona.

        Raises:
            ValueError: If no workspace is configured for the persona.
        """
        try:
            return self._WORKSPACE_MAP[context.persona]
        except KeyError as exc:
            raise ValueError(
                f"No workspace configured for persona: {context.persona}"
            ) from exc