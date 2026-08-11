"""Tests for workspace capability authorization."""

import pytest

from app.routing.workspace import Workspace
from app.workspaces.authorization import WorkspaceAuthorizer
from app.workspaces.capabilities import Capability


@pytest.fixture
def authorizer() -> WorkspaceAuthorizer:
    """Return a workspace authorizer."""
    return WorkspaceAuthorizer()


@pytest.mark.parametrize(
    ("workspace", "capability"),
    [
        (Workspace.BUSINESS, Capability.CUSTOMER_INSIGHTS),
        (Workspace.BUSINESS, Capability.PRODUCT_INSIGHTS),
        (Workspace.CUSTOMER, Capability.CUSTOMER_PROFILE),
        (Workspace.CUSTOMER, Capability.RECOMMENDATIONS),
        (Workspace.DEVELOPER, Capability.CODEBASE_ACCESS),
        (Workspace.DEVELOPER, Capability.DEBUGGING),
    ],
)
def test_allowed_capabilities(
    authorizer: WorkspaceAuthorizer,
    workspace: Workspace,
    capability: Capability,
) -> None:
    """Verify allowed workspace capabilities."""
    assert authorizer.has_capability(workspace, capability)


@pytest.mark.parametrize(
    ("workspace", "capability"),
    [
        (Workspace.BUSINESS, Capability.CODEBASE_ACCESS),
        (Workspace.CUSTOMER, Capability.DEBUGGING),
        (Workspace.CUSTOMER, Capability.CODEBASE_ACCESS),
        (Workspace.DEVELOPER, Capability.CUSTOMER_PROFILE),
    ],
)
def test_denied_capabilities(
    authorizer: WorkspaceAuthorizer,
    workspace: Workspace,
    capability: Capability,
) -> None:
    """Verify denied workspace capabilities."""
    assert not authorizer.has_capability(workspace, capability)


def test_require_raises_for_denied_capability(
    authorizer: WorkspaceAuthorizer,
) -> None:
    """Verify require raises PermissionError when access is denied."""
    with pytest.raises(PermissionError):
        authorizer.require(
            Workspace.CUSTOMER,
            Capability.CODEBASE_ACCESS,
        )