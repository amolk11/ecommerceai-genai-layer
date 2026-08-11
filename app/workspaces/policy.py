"""Capability policies for GenAI workspaces."""

from app.routing.workspace import Workspace
from app.workspaces.capabilities import Capability


WORKSPACE_CAPABILITIES: dict[Workspace, frozenset[Capability]] = {
    Workspace.BUSINESS: frozenset(
        {
            Capability.CUSTOMER_INSIGHTS,
            Capability.PRODUCT_INSIGHTS,
            Capability.RECOMMENDATIONS,
        }
    ),
    Workspace.CUSTOMER: frozenset(
        {
            Capability.CUSTOMER_PROFILE,
            Capability.PRODUCT_INFORMATION,
            Capability.RECOMMENDATIONS,
        }
    ),
    Workspace.DEVELOPER: frozenset(
        {
            Capability.CODEBASE_ACCESS,
            Capability.DOCUMENTATION_ACCESS,
            Capability.DEBUGGING,
        }
    ),
}