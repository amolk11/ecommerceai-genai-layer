"""PostgreSQL-backed composition of compact Business context."""

from collections.abc import Mapping, Sequence
from typing import Any

from app.data.protocols import BusinessDataRepository, Row
from app.models.business_context import (
    BusinessContext,
    CustomerBehavioralIntelligence,
    CustomerBusinessOverview,
    DataAvailabilityMetadata,
    ProductIntelligenceRecord,
    RecommendationRecord,
)


class PostgresBusinessContextProvider:
    """Compose bounded serving and analytics records for a Business request."""

    def __init__(
        self,
        repository: BusinessDataRepository,
        product_limit: int = 5,
        recommendation_limit: int = 5,
    ) -> None:
        if product_limit < 1 or recommendation_limit < 1:
            raise ValueError("Business context limits must be at least one.")
        self._repository = repository
        self._product_limit = product_limit
        self._recommendation_limit = recommendation_limit

    def build(self, user_id: str) -> BusinessContext:
        """Load bounded records and preserve their observed/derived distinction."""
        profile = self._repository.get_customer_profile(user_id)
        behavior = self._repository.get_customer_behavior(user_id)
        scores = self._repository.get_customer_scores(user_id)
        segments = self._repository.get_customer_segment(user_id)
        favorite_products = self._repository.get_favorite_products(
            user_id, self._product_limit
        )
        favorite_aisles = self._repository.get_favorite_aisles(
            user_id, self._product_limit
        )
        favorite_departments = self._repository.get_favorite_departments(
            user_id, self._product_limit
        )
        product_ids = self._product_ids(favorite_products[: self._product_limit])
        products = self._repository.get_product_intelligence(
            product_ids, self._product_limit
        )
        recommendations = self._repository.get_recommendations(
            product_ids, self._recommendation_limit
        )

        return BusinessContext(
            customer_overview=CustomerBusinessOverview(
                user_id=user_id,
                observed_profile=self._attributes(profile, {"user_id"}),
                favorite_aisles=[dict(row) for row in favorite_aisles[: self._product_limit]],
                favorite_departments=[
                    dict(row) for row in favorite_departments[: self._product_limit]
                ],
            ),
            behavioral_intelligence=CustomerBehavioralIntelligence(
                observed_behavior=self._attributes(behavior, {"user_id"}),
                model_derived_scores=self._attributes(scores, {"user_id"}),
                segments=self._attributes(segments, {"user_id"}),
            ),
            products=[self._product_record(row) for row in products[: self._product_limit]],
            recommendations=[
                self._recommendation_record(row)
                for row in recommendations[: self._recommendation_limit]
            ],
            data_availability=DataAvailabilityMetadata(
                serving_sources=self._present_sources(
                    {
                        "serving.customer_profile": profile,
                        "serving.customer_favorite_products": favorite_products,
                        "serving.customer_favorite_aisles": favorite_aisles,
                        "serving.customer_favorite_departments": favorite_departments,
                        "serving.product_intelligence": products,
                        "serving.product_recommendations_top20": recommendations,
                    }
                ),
                analytics_sources=self._present_sources(
                    {
                        "analytics.customer_behavior": behavior,
                        "analytics.customer_business_scores": scores,
                        "analytics.customer_segments": segments,
                    }
                ),
                product_limit=self._product_limit,
                recommendation_limit=self._recommendation_limit,
            ),
        )

    @staticmethod
    def _product_ids(rows: Sequence[Row]) -> list[int]:
        return [int(row["product_id"]) for row in rows if row.get("product_id") is not None]

    @staticmethod
    def _attributes(row: Row | None, excluded: set[str]) -> dict[str, Any]:
        return {key: value for key, value in (row or {}).items() if key not in excluded}

    @staticmethod
    def _product_record(row: Mapping[str, Any]) -> ProductIntelligenceRecord:
        return ProductIntelligenceRecord(
            product_id=row.get("product_id"),
            product_name=row.get("product_name"),
            attributes=PostgresBusinessContextProvider._attributes(
                row, {"product_id", "product_name"}
            ),
        )

    @staticmethod
    def _recommendation_record(row: Mapping[str, Any]) -> RecommendationRecord:
        return RecommendationRecord(
            source_product_id=row.get("source_product_id"),
            product_id=row.get("product_id"),
            product_name=row.get("product_name"),
            recommendation_attributes=PostgresBusinessContextProvider._attributes(
                row,
                {"user_id", "source_product_id", "product_id", "product_name"},
            ),
        )

    @staticmethod
    def _present_sources(sources: Mapping[str, object]) -> list[str]:
        return [name for name, data in sources.items() if data]
