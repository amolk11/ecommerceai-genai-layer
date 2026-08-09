"""Workspace definitions for the GenAI Layer."""

from enum import Enum


class Workspace(str, Enum):
    """Supported GenAI workspaces."""

    BUSINESS = "business"
    CUSTOMER = "customer"
    DEVELOPER = "developer"
    