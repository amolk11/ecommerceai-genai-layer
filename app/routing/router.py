"""Persona-based request routing."""

from app.models.user_context import UserContext
from app.routing.persona import Persona


class PersonaRouter:
    """Routes requests to the appropriate user workspace."""

    _WORKSPACE_MAP = {
        Persona.BUSINESS: "business",
        Persona.CUSTOMER: "customer",
        Persona.DEVELOPER: "developer",
    }

    def route(self, context: UserContext) -> str:
        """
        Return the workspace associated with the user's persona.

        Args:
            context: Authenticated user context.

        Returns:
            Workspace identifier.

        Raises:
            ValueError: If no workspace is configured for the persona.
        """
        try:
            return self._WORKSPACE_MAP[context.persona]
        except KeyError as exc:
            raise ValueError(
                f"No workspace configured for persona: {context.persona}"
            ) from exc