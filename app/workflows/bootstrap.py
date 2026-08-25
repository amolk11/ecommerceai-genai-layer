"""Composition root for the built-in persona workflows."""

from app.routing.workspace import Workspace
from app.workflows.business import build_business_graph
from app.workflows.customer import build_customer_graph
from app.workflows.developer import build_developer_graph
from app.workflows.registry import WorkflowRegistry


def create_workflow_registry() -> WorkflowRegistry:
    """Create a registry containing every built-in workspace workflow."""
    registry = WorkflowRegistry()
    registry.register(Workspace.BUSINESS, build_business_graph())
    registry.register(Workspace.CUSTOMER, build_customer_graph())
    registry.register(Workspace.DEVELOPER, build_developer_graph())
    return registry
