"""Registry for GenAI workflows."""

from app.workflows.protocols import Workflow
from app.routing.workspace import Workspace


class WorkflowRegistry:
    """Maps workspaces to their workflow implementations."""

    def __init__(self) -> None:
        self._workflows: dict[Workspace, Workflow] = {}

    def register(self, workspace: Workspace, workflow: Workflow) -> None:
        """Register a workflow for a workspace."""
        self._workflows[workspace] = workflow

    def get(self, workspace: Workspace) -> Workflow:
        """Return the workflow registered for a workspace."""
        try:
            return self._workflows[workspace]
        except KeyError as exc:
            raise ValueError(
                f"No workflow registered for workspace: {workspace}"
            ) from exc
