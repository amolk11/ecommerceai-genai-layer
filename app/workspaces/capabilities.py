"""Workspace capability definitions."""

from enum import Enum


class Capability(str, Enum):
    """Capabilities available to GenAI workspaces."""

    CUSTOMER_INSIGHTS = "customer_insights"
    PRODUCT_INSIGHTS = "product_insights"
    RECOMMENDATIONS = "recommendations"

    CUSTOMER_PROFILE = "customer_profile"
    PRODUCT_INFORMATION = "product_information"

    CODEBASE_ACCESS = "codebase_access"
    DOCUMENTATION_ACCESS = "documentation_access"
    DEBUGGING = "debugging"
    