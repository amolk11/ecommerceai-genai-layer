"""Application orchestration for persona-aware GenAI requests."""

from typing import Any

from pydantic import BaseModel, ConfigDict

from app.models.request_context import RequestContext
from app.errors import AuthorizationError, WorkflowNotFoundError
from app.routing.router import PersonaRouter
from app.routing.workspace import Workspace
from app.state.base import BaseGenAIState
from app.state.factory import StateFactory
from app.workflows.registry import WorkflowRegistry
from app.workspaces.authorization import WorkspaceAuthorizer


class WorkflowResult(BaseModel):
    """The public result of a workflow execution."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    workspace: Workspace
    state: BaseGenAIState


class GenAIService:
    """Executes a request through its persona-specific workflow."""

    def __init__(
        self,
        registry: WorkflowRegistry,
        router: PersonaRouter | None = None,
        authorizer: WorkspaceAuthorizer | None = None,
        state_factory: StateFactory | None = None,
    ) -> None:
        self._registry = registry
        self._router = router or PersonaRouter()
        self._authorizer = authorizer or WorkspaceAuthorizer()
        self._state_factory = state_factory or StateFactory()

    def handle(self, context: RequestContext) -> WorkflowResult:
        """Route, authorize, execute, and return the request's workflow state."""
        workspace = self._router.route(context.user)
        try:
            self._authorizer.authorize_persona(context.user.persona, workspace)
        except PermissionError as exc:
            raise AuthorizationError(exc) from exc

        state = self._state_factory.create(context, workspace)
        try:
            workflow = self._registry.get(workspace)
        except ValueError as exc:
            raise WorkflowNotFoundError(exc) from exc
        output = workflow.invoke(state)

        return WorkflowResult(
            workspace=workspace,
            state=self._coerce_state(state, output),
        )

    @staticmethod
    def _coerce_state(
        expected_state: BaseGenAIState,
        output: BaseGenAIState | dict[str, Any],
    ) -> BaseGenAIState:
        """Restore the concrete state model after LangGraph returns a mapping."""
        if isinstance(output, BaseGenAIState):
            return output
        return type(expected_state).model_validate(output)
