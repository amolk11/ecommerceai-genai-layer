"""Reusable prompt for business intelligence."""

from app.models.business_context import BusinessContext


def build_business_intelligence_prompt(message: str, context: BusinessContext) -> str:
    """Build the concise prompt used for a business intelligence request."""
    return f"""You are a business intelligence assistant.
Understand the user's business question, provide a concise summary, identify
important points, and suggest practical next actions. Return only the required
structured business insight.

Business question: {message}

Business context (bounded PostgreSQL records): {context.model_dump_json()}"""
