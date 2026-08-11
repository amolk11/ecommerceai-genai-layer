"""Tests for the GenAI workflow registry."""

import pytest

from app.routing.workspace import Workspace
from app.workflows.registry import WorkflowRegistry
from app.workflows.base import build_base_graph

class FakeWorkflow:
    """Minimal workflow implementation for testing."""

    def invoke(self, state):
        """Return the provided state."""
        return state


@pytest.fixture
def registry() -> WorkflowRegistry:
    """Return an empty workflow registry."""
    return WorkflowRegistry()


def test_register_and_get_workflow(
    registry: WorkflowRegistry,
) -> None:
    """Verify that a registered workflow can be retrieved."""
    workflow = FakeWorkflow()

    registry.register(Workspace.BUSINESS, workflow)

    assert registry.get(Workspace.BUSINESS) is workflow


def test_get_unregistered_workflow_raises(
    registry: WorkflowRegistry,
) -> None:
    """Verify that missing workflows raise an error."""
    with pytest.raises(ValueError, match="No workflow registered"):
        registry.get(Workspace.DEVELOPER)
        
def test_register_and_execute_langgraph_workflow(
    registry: WorkflowRegistry,
) -> None:
    """Verify that a compiled LangGraph workflow can be registered."""
    workflow = build_base_graph()

    registry.register(Workspace.BUSINESS, workflow)

    registered_workflow = registry.get(Workspace.BUSINESS)

    assert registered_workflow is workflow
    assert callable(registered_workflow.invoke)
    