"""Bounded PostgreSQL-derived context for Business intelligence."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class CustomerBusinessOverview(BaseModel):
    """Observed customer profile values from the serving layer."""

    user_id: str
    observed_profile: dict[str, Any] = Field(default_factory=dict)
    favorite_aisles: list[dict[str, Any]] = Field(default_factory=list)
    favorite_departments: list[dict[str, Any]] = Field(default_factory=list)


class CustomerBehavioralIntelligence(BaseModel):
    """Separate observed behavior from model-derived values and segments."""

    observed_behavior: dict[str, Any] = Field(default_factory=dict)
    model_derived_scores: dict[str, Any] = Field(default_factory=dict)
    segments: dict[str, Any] = Field(default_factory=dict)


class ProductIntelligenceRecord(BaseModel):
    """One bounded product intelligence record from the serving layer."""

    model_config = ConfigDict(extra="allow")

    product_id: int | None = None
    product_name: str | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)


class RecommendationRecord(BaseModel):
    """One bounded recommendation record, including any derived score as supplied."""

    model_config = ConfigDict(extra="allow")

    product_id: int | None = None
    product_name: str | None = None
    recommendation_attributes: dict[str, Any] = Field(default_factory=dict)


class DataAvailabilityMetadata(BaseModel):
    """Describes which bounded data sources supplied the context."""

    serving_sources: list[str] = Field(default_factory=list)
    analytics_sources: list[str] = Field(default_factory=list)
    product_limit: int = Field(ge=1)
    recommendation_limit: int = Field(ge=1)


class BusinessContext(BaseModel):
    """Compact context passed to the Business workflow and its LLM prompt."""

    customer_overview: CustomerBusinessOverview
    behavioral_intelligence: CustomerBehavioralIntelligence
    products: list[ProductIntelligenceRecord] = Field(default_factory=list)
    recommendations: list[RecommendationRecord] = Field(default_factory=list)
    data_availability: DataAvailabilityMetadata
