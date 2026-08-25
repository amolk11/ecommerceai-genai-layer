"""Read-only validation of PostgreSQL BusinessContext schema dependencies."""

from collections.abc import Callable
from typing import Any

from app.data.schema import BusinessContextSchemaContract, TableProjection


class SchemaContractError(RuntimeError):
    """Raised when a required BusinessContext table or column is unavailable."""


class PostgresSchemaValidator:
    """Validate only the tables and columns consumed by the Business context layer."""

    _TABLES = {
        "customer_profile": ("serving", "customer_profile"),
        "customer_behavior": ("analytics", "customer_behavior"),
        "customer_scores": ("analytics", "customer_business_scores"),
        "customer_segments": ("analytics", "customer_segments"),
        "favorite_products": ("serving", "customer_favorite_products"),
        "favorite_aisles": ("serving", "customer_favorite_aisles"),
        "favorite_departments": ("serving", "customer_favorite_departments"),
        "product_intelligence": ("serving", "product_intelligence"),
        "recommendations": ("serving", "product_recommendations_top20"),
    }

    def __init__(
        self,
        connection_factory: Callable[[], Any],
        contract: BusinessContextSchemaContract,
    ) -> None:
        self._connection_factory = connection_factory
        self._contract = contract

    def validate(self) -> None:
        """Raise a clear error for every missing consumed table or column."""
        with self._connection_factory() as connection:
            with connection.cursor() as cursor:
                for name, (schema, table) in self._TABLES.items():
                    projection = getattr(self._contract, name)
                    self._validate_table(cursor, schema, table, projection)

    @staticmethod
    def _validate_table(
        cursor: Any, schema: str, table: str, projection: TableProjection
    ) -> None:
        cursor.execute(
            "SELECT 1 FROM information_schema.tables "
            "WHERE table_schema = %s AND table_name = %s",
            (schema, table),
        )
        if cursor.fetchone() is None:
            raise SchemaContractError(f"Missing required table: {schema}.{table}")

        required_columns = sorted(projection.required_columns())
        cursor.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = %s AND table_name = %s AND column_name = ANY(%s)",
            (schema, table, required_columns),
        )
        present_columns = {row["column_name"] for row in cursor.fetchall()}
        missing_columns = set(required_columns) - present_columns
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise SchemaContractError(
                f"Missing required column(s) in {schema}.{table}: {missing}"
            )
