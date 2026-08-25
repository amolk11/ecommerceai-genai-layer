"""PostgreSQL implementation of business-context data access."""

import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from app.data.protocols import BusinessDataRepository, Row
from app.data.schema import BusinessContextSchemaContract, TableProjection


def normalize_postgres_dsn(dsn: str) -> str:
    """Normalize supported PostgreSQL URL schemes for psycopg without logging them."""
    sqlalchemy_prefix = "postgresql+psycopg://"
    if dsn.startswith(sqlalchemy_prefix):
        return "postgresql://" + dsn.removeprefix(sqlalchemy_prefix)
    if dsn.startswith(("postgresql://", "postgres://")):
        return dsn
    raise ValueError(
        "DATABASE_URL must use postgresql://, postgres://, or postgresql+psycopg://."
    )


@dataclass(frozen=True)
class PostgresSettings:
    """PostgreSQL connection settings loaded from the environment."""

    dsn: str

    def __post_init__(self) -> None:
        """Store the driver-compatible form while keeping credentials opaque."""
        object.__setattr__(self, "dsn", normalize_postgres_dsn(self.dsn))

    @classmethod
    def from_environment(cls) -> "PostgresSettings":
        """Load the connection string without hardcoding credentials."""
        dsn = os.getenv("DATABASE_URL")
        if not dsn:
            raise RuntimeError("DATABASE_URL must be set for PostgreSQL.")
        return cls(dsn=dsn)


ConnectionFactory = Callable[[], Any]


def create_connection_factory(settings: PostgresSettings) -> ConnectionFactory:
    """Create a lazy, non-global psycopg connection factory."""
    def connect() -> Any:
        try:
            import psycopg
            from psycopg.rows import dict_row
        except ImportError as exc:
            raise RuntimeError("psycopg is required for PostgreSQL data access.") from exc
        return psycopg.connect(settings.dsn, row_factory=dict_row)

    return connect


class PostgresBusinessRepository(BusinessDataRepository):
    """Execute parameterized, bounded queries against ecommerce PostgreSQL data."""

    def __init__(
        self,
        connection_factory: ConnectionFactory,
        schema_contract: BusinessContextSchemaContract,
    ) -> None:
        self._connection_factory = connection_factory
        self._schema = schema_contract

    @classmethod
    def from_settings(
        cls,
        settings: PostgresSettings,
        schema_contract: BusinessContextSchemaContract,
    ) -> "PostgresBusinessRepository":
        """Create a repository with a lazy psycopg connection factory."""
        return cls(create_connection_factory(settings), schema_contract)

    def get_customer_profile(self, user_id: str) -> Row | None:
        return self._fetch_one(
            self._user_query("serving.customer_profile", self._schema.customer_profile),
            (user_id, 1),
        )

    def get_customer_behavior(self, user_id: str) -> Row | None:
        return self._fetch_one(
            self._user_query("analytics.customer_behavior", self._schema.customer_behavior),
            (user_id, 1),
        )

    def get_customer_scores(self, user_id: str) -> Row | None:
        return self._fetch_one(
            self._user_query(
                "analytics.customer_business_scores", self._schema.customer_scores
            ),
            (user_id, 1),
        )

    def get_customer_segment(self, user_id: str) -> Row | None:
        return self._fetch_one(
            self._user_query("analytics.customer_segments", self._schema.customer_segments),
            (user_id, 1),
        )

    def get_favorite_products(self, user_id: str, limit: int) -> Sequence[Row]:
        return self._fetch_all(
            self._user_query(
                "serving.customer_favorite_products", self._schema.favorite_products
            ),
            (user_id, limit),
        )

    def get_favorite_aisles(self, user_id: str, limit: int) -> Sequence[Row]:
        return self._fetch_all(
            self._user_query(
                "serving.customer_favorite_aisles", self._schema.favorite_aisles
            ),
            (user_id, limit),
        )

    def get_favorite_departments(self, user_id: str, limit: int) -> Sequence[Row]:
        return self._fetch_all(
            self._user_query(
                "serving.customer_favorite_departments", self._schema.favorite_departments
            ),
            (user_id, limit),
        )

    def get_product_intelligence(
        self, product_ids: Sequence[int], limit: int
    ) -> Sequence[Row]:
        if not product_ids:
            return []
        return self._fetch_all(
            self._product_query(
                "serving.product_intelligence", self._schema.product_intelligence
            ),
            (list(product_ids), limit),
        )

    def get_recommendations(
        self, source_product_ids: Sequence[int], limit: int
    ) -> Sequence[Row]:
        if not source_product_ids:
            return []
        return self._fetch_all(
            self._product_query(
                "serving.product_recommendations_top20", self._schema.recommendations
            ),
            (list(source_product_ids), limit),
        )

    def _fetch_one(self, query: str, parameters: tuple[Any, ...]) -> Row | None:
        rows = self._fetch_all(query, parameters)
        return rows[0] if rows else None

    def _fetch_all(
        self, query: str, parameters: tuple[Any, ...]
    ) -> Sequence[Row]:
        with self._connection_factory() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, parameters)
                return list(cursor.fetchall())

    @staticmethod
    def _user_query(table: str, projection: TableProjection) -> str:
        """Build a bounded user-scoped query from a verified schema contract."""
        return (
            f"SELECT {projection.select_list()} FROM {table} "
            f"WHERE {projection.user_filter_column()} = %s LIMIT %s"
        )

    @staticmethod
    def _product_query(table: str, projection: TableProjection) -> str:
        """Build a bounded product-scoped query from a verified schema contract."""
        return (
            f"SELECT {projection.select_list()} FROM {table} "
            f"WHERE {projection.product_filter_column()} = ANY(%s) LIMIT %s"
        )
