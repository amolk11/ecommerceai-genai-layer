"""Tests for persona-based routing."""

import pytest

from app.models.user_context import UserContext
from app.routing.persona import Persona
from app.routing.router import PersonaRouter


@pytest.fixture
def router() -> PersonaRouter:
    """Return a persona router."""
    return PersonaRouter()


@pytest.mark.parametrize(
    ("persona", "expected_workspace"),
    [
        (Persona.BUSINESS, "business"),
        (Persona.CUSTOMER, "customer"),
        (Persona.DEVELOPER, "developer"),
    ],
)
def test_routes_persona_to_workspace(
    router: PersonaRouter,
    persona: Persona,
    expected_workspace: str,
) -> None:
    """Verify that each persona maps to the correct workspace."""
    context = UserContext(
        user_id="test-user",
        persona=persona,
    )

    assert router.route(context) == expected_workspace
    