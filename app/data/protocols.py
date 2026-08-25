"""Database-facing contracts with no workflow or LLM concerns."""

from collections.abc import Sequence
from typing import Any, Mapping, Protocol


Row = Mapping[str, Any]


class BusinessDataRepository(Protocol):
    """Read the bounded records required for a business context."""

    def get_customer_profile(self, user_id: str) -> Row | None: ...

    def get_customer_behavior(self, user_id: str) -> Row | None: ...

    def get_customer_scores(self, user_id: str) -> Row | None: ...

    def get_customer_segment(self, user_id: str) -> Row | None: ...

    def get_favorite_products(self, user_id: str, limit: int) -> Sequence[Row]: ...

    def get_favorite_aisles(self, user_id: str, limit: int) -> Sequence[Row]: ...

    def get_favorite_departments(self, user_id: str, limit: int) -> Sequence[Row]: ...

    def get_product_intelligence(
        self, product_ids: Sequence[int], limit: int
    ) -> Sequence[Row]: ...

    def get_recommendations(
        self, source_product_ids: Sequence[int], limit: int
    ) -> Sequence[Row]: ...
