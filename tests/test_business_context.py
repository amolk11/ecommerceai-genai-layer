"""Unit tests for PostgreSQL-backed Business context composition."""

from collections.abc import Sequence
from typing import Any

from app.context.business import PostgresBusinessContextProvider
from app.data.postgres import PostgresBusinessRepository
from app.data.protocols import Row
from app.data.schema import (
    BusinessContextSchemaContract,
    ColumnProjection,
    TableProjection,
)
from app.models.business_context import (
    BusinessContext,
    CustomerBehavioralIntelligence,
    CustomerBusinessOverview,
    DataAvailabilityMetadata,
)


class FakeBusinessRepository:
    """In-memory repository double; it never opens a database connection."""

    def __init__(self) -> None:
        self.product_limit: int | None = None
        self.recommendation_limit: int | None = None

    def get_customer_profile(self, user_id: str) -> Row | None:
        return {"user_id": user_id, "total_orders": 12, "reorder_rate": 0.4}

    def get_customer_behavior(self, user_id: str) -> Row | None:
        return {"user_id": user_id, "avg_basket_size": 8.2}

    def get_customer_scores(self, user_id: str) -> Row | None:
        return {"user_id": user_id, "retention_score": 0.83}

    def get_customer_segment(self, user_id: str) -> Row | None:
        return {"user_id": user_id, "segment": "loyal"}

    def get_favorite_products(self, user_id: str, limit: int) -> Sequence[Row]:
        self.product_limit = limit
        return [{"product_id": index} for index in range(1, 10)]

    def get_favorite_aisles(self, user_id: str, limit: int) -> Sequence[Row]:
        return [{"aisle": "produce"}]

    def get_favorite_departments(self, user_id: str, limit: int) -> Sequence[Row]:
        return [{"department": "fresh"}]

    def get_product_intelligence(
        self, product_ids: Sequence[int], limit: int
    ) -> Sequence[Row]:
        assert product_ids == [1, 2]
        return [
            {
                "product_id": index,
                "product_name": f"Product {index}",
                "reorder_rate": 0.2,
            }
            for index in range(1, 10)
        ]

    def get_recommendations(self, user_id: str, limit: int) -> Sequence[Row]:
        self.recommendation_limit = limit
        return [
            {
                "user_id": user_id,
                "product_id": index,
                "product_name": f"Recommended {index}",
                "recommendation_score": 0.9,
            }
            for index in range(1, 10)
        ]


def test_business_context_allows_missing_optional_data() -> None:
    """An available user can have no product or recommendation records."""
    context = BusinessContext(
        customer_overview=CustomerBusinessOverview(user_id="42"),
        behavioral_intelligence=CustomerBehavioralIntelligence(),
        data_availability=DataAvailabilityMetadata(
            product_limit=5, recommendation_limit=5
        ),
    )

    assert context.products == []
    assert context.recommendations == []
    assert context.customer_overview.observed_profile == {}


def test_provider_maps_serving_and_analytics_records_into_business_context() -> None:
    """Profile data stays observed while scores and segments stay distinguished."""
    repository = FakeBusinessRepository()
    context = PostgresBusinessContextProvider(
        repository, product_limit=2, recommendation_limit=3
    ).build("42")

    assert context.customer_overview.observed_profile == {
        "total_orders": 12,
        "reorder_rate": 0.4,
    }
    assert context.customer_overview.favorite_aisles == [{"aisle": "produce"}]
    assert context.customer_overview.favorite_departments == [{"department": "fresh"}]
    assert context.behavioral_intelligence.observed_behavior == {
        "avg_basket_size": 8.2
    }
    assert context.behavioral_intelligence.model_derived_scores == {
        "retention_score": 0.83
    }
    assert context.behavioral_intelligence.segments == {"segment": "loyal"}
    assert context.products[0].product_name == "Product 1"
    assert context.products[0].attributes == {"reorder_rate": 0.2}
    assert context.recommendations[0].recommendation_attributes == {
        "recommendation_score": 0.9
    }
    assert "serving.customer_profile" in context.data_availability.serving_sources
    assert "analytics.customer_business_scores" in context.data_availability.analytics_sources


def test_provider_applies_bounded_top_n_limits() -> None:
    """The provider limits both database requests and prompt-facing records."""
    repository = FakeBusinessRepository()
    context = PostgresBusinessContextProvider(
        repository, product_limit=2, recommendation_limit=3
    ).build("42")

    assert repository.product_limit == 2
    assert repository.recommendation_limit == 3
    assert len(context.products) == 2
    assert len(context.recommendations) == 3


class FakeCursor:
    """Cursor double for testing SQL shape without PostgreSQL."""

    def __init__(self) -> None:
        self.executions: list[tuple[str, tuple[Any, ...]]] = []

    def __enter__(self) -> "FakeCursor":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def execute(self, query: str, parameters: tuple[Any, ...]) -> None:
        self.executions.append((query, parameters))

    def fetchall(self) -> list[Row]:
        return [{"user_id": "42"}]


class FakeConnection:
    """Connection double that exposes the cursor's recorded SQL."""

    def __init__(self) -> None:
        self.cursor_instance = FakeCursor()

    def __enter__(self) -> "FakeConnection":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def cursor(self) -> FakeCursor:
        return self.cursor_instance


def schema_contract() -> BusinessContextSchemaContract:
    """A verified fixture contract; production values require DB inspection."""
    user_table = lambda columns: TableProjection(
        columns=columns, user_id_column="customer_key"
    )
    return BusinessContextSchemaContract(
        customer_profile=user_table((ColumnProjection("customer_key", "user_id"),)),
        customer_behavior=user_table((ColumnProjection("customer_key", "user_id"),)),
        customer_scores=user_table((ColumnProjection("customer_key", "user_id"),)),
        customer_segments=user_table((ColumnProjection("customer_key", "user_id"),)),
        favorite_products=user_table((ColumnProjection("product_key", "product_id"),)),
        favorite_aisles=user_table((ColumnProjection("aisle_name", "aisle"),)),
        favorite_departments=user_table(
            (ColumnProjection("department_name", "department"),)
        ),
        product_intelligence=TableProjection(
            columns=(
                ColumnProjection("product_key", "product_id"),
                ColumnProjection("display_name", "product_name"),
            ),
            product_id_column="product_key",
        ),
        recommendations=user_table(
            (
                ColumnProjection("product_key", "product_id"),
                ColumnProjection("display_name", "product_name"),
            )
        ),
    )


def test_postgres_repository_uses_parameterized_serving_queries() -> None:
    """Repository owns SQL and sends user input separately as parameters."""
    connection = FakeConnection()
    repository = PostgresBusinessRepository(lambda: connection, schema_contract())

    profile = repository.get_customer_profile("42")
    recommendations = repository.get_recommendations("42", 5)

    assert profile == {"user_id": "42"}
    assert recommendations == [{"user_id": "42"}]
    profile_query, profile_parameters = connection.cursor_instance.executions[0]
    recommendation_query, recommendation_parameters = connection.cursor_instance.executions[1]
    assert "serving.customer_profile" in profile_query
    assert profile_query == (
        "SELECT customer_key AS user_id FROM serving.customer_profile "
        "WHERE customer_key = %s LIMIT %s"
    )
    assert profile_parameters == ("42", 1)
    assert "serving.product_recommendations_top20" in recommendation_query
    assert "SELECT product_key AS product_id, display_name AS product_name" in recommendation_query
    assert "*" not in recommendation_query
    assert recommendation_parameters == ("42", 5)


def test_schema_contract_rejects_an_unsafe_identifier() -> None:
    """Schema mappings are identifiers, never a channel for arbitrary SQL."""
    projection = TableProjection(
        columns=(ColumnProjection("user_id; DROP TABLE users"),),
        user_id_column="user_id",
    )

    try:
        projection.select_list()
    except ValueError as exc:
        assert "Unsafe SQL identifier" in str(exc)
    else:
        raise AssertionError("Unsafe schema identifier was accepted")
