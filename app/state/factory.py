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

    def create(
        self,
        context: RequestContext,
        workspace: Workspace,
    ) -> BaseGenAIState:
        """
        Create workflow state for the resolved workspace.

        Args:
            context: Internal request context.
            workspace: Workspace resolved by the persona router.

        Returns:
            Persona-specific GenAI workflow state.

        Raises:
            ValueError: If no state is configured for the workspace.
        """
        state_class = self._STATE_MAP.get(workspace)

        if state_class is None:
            raise ValueError(
                f"No state configured for workspace: {workspace}"
            )

        return state_class(
            context=context,
            workspace=workspace,
        )
        