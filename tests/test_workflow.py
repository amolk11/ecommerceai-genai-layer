"""Tests for the base LangGraph workflow."""

from app.models.request import GenAIRequest
from app.models.request_context import RequestContext
from app.models.user_context import UserContext
from app.routing.persona import Persona
from app.routing.workspace import Workspace
from app.state.business import BusinessState
from app.workflows.base import build_base_graph


def test_base_graph_executes() -> None:
    """Verify that the base graph accepts and returns workflow state."""
    context = RequestContext(
        user=UserContext(
            user_id="test-user",
            persona=Persona.BUSINESS,
        ),
        request=GenAIRequest(
            message="Give me business insights.",
        ),
    )

    state = BusinessState(
        context=context,
        workspace=Workspace.BUSINESS,
    )

    graph = build_base_graph()

    result = graph.invoke(state)

    assert result["workspace"] == Workspace.BUSINESS
    assert result["context"].user.user_id == "test-user"