"""Structured output for business intelligence requests."""

from pydantic import BaseModel, Field


class BusinessInsight(BaseModel):
    """A concise, actionable business intelligence response."""

    summary: str = Field(min_length=1)
    key_points: list[str] = Field(min_length=1)
    recommended_actions: list[str] = Field(min_length=1)
