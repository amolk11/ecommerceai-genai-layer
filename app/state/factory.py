"""Factory for creating persona-specific GenAI workflow state."""

from app.models.request_context import RequestContext
from app.routing.workspace import Workspace
from app.state.base import BaseGenAIState
from app.state.business import BusinessState
from app.state.customer import CustomerState
from app.state.developer import DeveloperState


class StateFactory:
    """Creates the appropriate workflow state for a workspace."""

    _STATE_MAP = {
        Workspace.BUSINESS: BusinessState,
        Workspace.CUSTOMER: CustomerState,
        Workspace.DEVELOPER: DeveloperState,
    }

    def create(self, context: RequestContext) -> BaseGenAIState:
        """
        Create workflow state for the user's workspace.

        Args:
            context: Internal request context.

        Returns:
            Persona-specific GenAI workflow state.

        Raises:
            ValueError: If no state is configured for the workspace.
        """
        state_class = self._STATE_MAP.get(context.user.persona)

        if state_class is None:
            raise ValueError(
                f"No state configured for persona: {context.user.persona}"
            )

        return state_class(
            context=context,
            workspace=self._workspace_for(context),
        )

    @staticmethod
    def _workspace_for(context: RequestContext) -> Workspace:
        """Resolve the workspace associated with the user's persona."""
        workspace_map = {
            "business": Workspace.BUSINESS,
            "customer": Workspace.CUSTOMER,
            "developer": Workspace.DEVELOPER,
        }

        try:
            return workspace_map[context.user.persona.value]
        except KeyError as exc:
            raise ValueError(
                f"No workspace configured for persona: {context.user.persona}"
            ) from exc