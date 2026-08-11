"""Workspace capability authorization."""

from app.routing.workspace import Workspace
from app.workspaces.capabilities import Capability
from app.workspaces.policy import WORKSPACE_CAPABILITIES


class WorkspaceAuthorizer:
    """Checks whether a workspace has a requested capability."""

    def has_capability(
        self,
        workspace: Workspace,
        capability: Capability,
    ) -> bool:
        """Return whether the workspace has the requested capability."""
        return capability in WORKSPACE_CAPABILITIES.get(workspace, frozenset())

    def require(
        self,
        workspace: Workspace,
        capability: Capability,
    ) -> None:
        """
        Require a workspace to have a capability.

        Raises:
            PermissionError: If the capability is not allowed.
        """
        if not self.has_capability(workspace, capability):
            raise PermissionError(
                f"Workspace '{workspace.value}' does not have "
                f"capability '{capability.value}'."
            )