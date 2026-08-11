"""Tests for the GenAI state factory."""

import pytest

from app.models.request import GenAIRequest
from app.models.request_context import RequestContext
from app.models.user_context import UserContext
from app.routing.persona import Persona
from app.routing.workspace import Workspace
from app.state.business import BusinessState
from app.state.customer import CustomerState
from app.state.developer import DeveloperState
from app.state.factory import StateFactory


@pytest.fixture
def factory() -> StateFactory:
    """Return a state factory."""
    return StateFactory()


@pytest.mark.parametrize(
    ("persona", "workspace", "expected_state"),
    [
        (Persona.BUSINESS, Workspace.BUSINESS, BusinessState),
        (Persona.CUSTOMER, Workspace.CUSTOMER, CustomerState),
        (Persona.DEVELOPER, Workspace.DEVELOPER, DeveloperState),
    ],
)
def test_creates_persona_specific_state(
    factory: StateFactory,
    persona: Persona,
    workspace: Workspace,
    expected_state: type,
) -> None:
    """Verify that each workspace creates the correct state."""
    context = RequestContext(
        user=UserContext(
            user_id="test-user",
            persona=persona,
        ),
        request=GenAIRequest(
            message="test request",
        ),
    )

    state = factory.create(context, workspace)

    assert isinstance(state, expected_state)
    assert state.workspace == workspace
    assert state.context == context