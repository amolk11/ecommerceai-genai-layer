"""End-to-end tests for the GenAI application service."""

import pytest

from app.bootstrap import create_genai_service
from app.models.business_insight import BusinessInsight
from app.models.business_context import (
    BusinessContext,
    CustomerBehavioralIntelligence,
    CustomerBusinessOverview,
    DataAvailabilityMetadata,
)
from app.models.request import GenAIRequest
from app.models.request_context import RequestContext
from app.models.user_context import UserContext
from app.routing.persona import Persona
from app.routing.workspace import Workspace
from app.services.genai import GenAIService
from app.state.business import BusinessState
from app.state.customer import CustomerState
from app.state.developer import DeveloperState
from app.workflows.registry import WorkflowRegistry


class FakeBusinessLLM:
    """Business LLM double used to keep service tests offline."""

    def generate_business_insight(self, prompt: str) -> BusinessInsight:
        return BusinessInsight(
            summary="Test insight",
            key_points=["Test point"],
            recommended_actions=["Test action"],
        )


class FakeBusinessContextProvider:
    """Context-provider double used to keep service tests database-free."""

    def build(self, user_id: str) -> BusinessContext:
        return BusinessContext(
            customer_overview=CustomerBusinessOverview(user_id=user_id),
            behavioral_intelligence=CustomerBehavioralIntelligence(),
            data_availability=DataAvailabilityMetadata(
                product_limit=5, recommendation_limit=5
            ),
        )


def make_context(persona: Persona) -> RequestContext:
    """Create a request context for a persona."""
    return RequestContext(
        user=UserContext(user_id="test-user", persona=persona),
        request=GenAIRequest(message="Execute this request."),
    )


@pytest.mark.parametrize(
    ("persona", "workspace", "state_type"),
    [
        (Persona.BUSINESS, Workspace.BUSINESS, BusinessState),
        (Persona.CUSTOMER, Workspace.CUSTOMER, CustomerState),
        (Persona.DEVELOPER, Workspace.DEVELOPER, DeveloperState),
    ],
)
def test_service_executes_the_persona_specific_workflow(
    persona: Persona,
    workspace: Workspace,
    state_type: type[BusinessState | CustomerState | DeveloperState],
) -> None:
    """A request reaches its mapped workspace graph with the correct state."""
    result = create_genai_service(
        FakeBusinessLLM(), FakeBusinessContextProvider()
    ).handle(make_context(persona))

    assert result.workspace is workspace
    assert isinstance(result.state, state_type)
    assert result.state.workspace is workspace
    assert result.state.context.user.persona is persona


class IncorrectRouter:
    """Router double that deliberately returns an unauthorized workspace."""

    def route(self, context: UserContext) -> Workspace:
        return Workspace.DEVELOPER


class RecordingWorkflow:
    """Workflow double that records whether execution was attempted."""

    def __init__(self) -> None:
        self.was_invoked = False

    def invoke(self, state: BusinessState) -> BusinessState:
        self.was_invoked = True
        return state


def test_unauthorized_persona_workspace_stops_before_workflow() -> None:
    """Authorization failures must prevent workflow execution."""
    workflow = RecordingWorkflow()
    registry = WorkflowRegistry()
    registry.register(Workspace.DEVELOPER, workflow)
    service = GenAIService(registry=registry, router=IncorrectRouter())

    with pytest.raises(PermissionError, match="not authorized"):
        service.handle(make_context(Persona.BUSINESS))

    assert not workflow.was_invoked


def test_missing_workflow_fails_clearly() -> None:
    """A routed request reports an absent workspace workflow."""
    service = GenAIService(registry=WorkflowRegistry())

    with pytest.raises(ValueError, match="No workflow registered"):
        service.handle(make_context(Persona.CUSTOMER))
