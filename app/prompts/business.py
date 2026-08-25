"""Reusable prompt for business intelligence."""


def build_business_intelligence_prompt(message: str) -> str:
    """Build the concise prompt used for a business intelligence request."""
    return f"""You are a business intelligence assistant.
Understand the user's business question, provide a concise summary, identify
important points, and suggest practical next actions. Return only the required
structured business insight.

Business question: {message}"""
