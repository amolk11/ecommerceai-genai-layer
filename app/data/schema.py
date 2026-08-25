"""Verified column contracts required by PostgreSQL Business context queries."""

import re
from dataclasses import dataclass


_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _validate_identifier(value: str) -> str:
    if not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"Unsafe SQL identifier in schema contract: {value!r}")
    return value


@dataclass(frozen=True)
class ColumnProjection:
    """One verified source column, optionally normalized to an application alias."""

    column: str
    alias: str | None = None

    def sql(self) -> str:
        """Return the safely quoted-free SQL projection from verified identifiers."""
        column = _validate_identifier(self.column)
        if self.alias is None:
            return column
        return f"{column} AS {_validate_identifier(self.alias)}"


@dataclass(frozen=True)
class TableProjection:
    """Columns and filter columns verified for one fixed platform table."""

    columns: tuple[ColumnProjection, ...]
    user_id_column: str | None = None
    product_id_column: str | None = None

    def select_list(self) -> str:
        """Build a non-empty explicit SELECT list."""
        if not self.columns:
            raise ValueError("A BusinessContext table projection needs at least one column.")
        return ", ".join(column.sql() for column in self.columns)

    def user_filter_column(self) -> str:
        """Return the verified customer identifier column for this projection."""
        if self.user_id_column is None:
            raise ValueError("A verified user_id_column is required.")
        return _validate_identifier(self.user_id_column)

    def product_filter_column(self) -> str:
        """Return the verified product identifier column for this projection."""
        if self.product_id_column is None:
            raise ValueError("A verified product_id_column is required.")
        return _validate_identifier(self.product_id_column)

    def required_columns(self) -> set[str]:
        """Return every source column required by this projection and its filter."""
        columns = {column.column for column in self.columns}
        if self.user_id_column:
            columns.add(self.user_filter_column())
        if self.product_id_column:
            columns.add(self.product_filter_column())
        return columns


@dataclass(frozen=True)
class BusinessContextSchemaContract:
    """Verified immutable column mappings for the EcommerceAI PostgreSQL schema."""

    customer_profile: TableProjection
    customer_behavior: TableProjection
    customer_scores: TableProjection
    customer_segments: TableProjection
    favorite_products: TableProjection
    favorite_aisles: TableProjection
    favorite_departments: TableProjection
    product_intelligence: TableProjection
    recommendations: TableProjection


def ecommerce_business_context_schema() -> BusinessContextSchemaContract:
    """Return the explicit mapping verified against the EcommerceAI schema export."""
    return BusinessContextSchemaContract(
        customer_profile=TableProjection(
            columns=tuple(
                ColumnProjection(column)
                for column in (
                    "user_id", "total_orders", "customer_tenure", "total_items",
                    "avg_basket_size", "avg_days_between_orders", "reorder_rate",
                    "total_reorders", "unique_products", "unique_departments",
                    "unique_aisles",
                )
            ),
            user_id_column="user_id",
        ),
        customer_behavior=TableProjection(
            columns=tuple(
                ColumnProjection(column)
                for column in (
                    "user_id", "purchase_depth_score", "purchase_depth",
                    "purchase_regularity_score", "purchase_regularity",
                    "purchase_loyalty_score", "purchase_loyalty",
                    "purchase_exploration_score", "purchase_exploration",
                )
            ),
            user_id_column="user_id",
        ),
        customer_scores=TableProjection(
            columns=tuple(
                ColumnProjection(column)
                for column in (
                    "user_id", "loyalty_score", "engagement_score",
                    "consistency_score", "diversity_score", "customer_health_score",
                )
            ),
            user_id_column="user_id",
        ),
        customer_segments=TableProjection(
            columns=tuple(
                ColumnProjection(column)
                for column in (
                    "user_id", "lifecycle_segment", "value_segment",
                    "behavior_segment", "confidence",
                )
            ),
            user_id_column="user_id",
        ),
        favorite_products=TableProjection(
            columns=(ColumnProjection("product_id"),), user_id_column="user_id"
        ),
        favorite_aisles=TableProjection(
            columns=tuple(
                ColumnProjection(column)
                for column in ("aisle", "preference_score", "purchase_count", "aisle_share")
            ),
            user_id_column="user_id",
        ),
        favorite_departments=TableProjection(
            columns=tuple(
                ColumnProjection(column)
                for column in (
                    "department", "preference_score", "purchase_count", "department_share"
                )
            ),
            user_id_column="user_id",
        ),
        product_intelligence=TableProjection(
            columns=tuple(
                ColumnProjection(column)
                for column in (
                    "product_id", "product_name", "department", "aisle",
                    "purchase_count", "unique_customers", "unique_orders",
                    "global_health_score", "primary_strength", "primary_weakness",
                )
            ),
            product_id_column="product_id",
        ),
        recommendations=TableProjection(
            columns=(
                ColumnProjection("product_id_a", "source_product_id"),
                ColumnProjection("product_id_b", "product_id"),
                ColumnProjection("recommendation_score"),
                ColumnProjection("recommendation_rank"),
                ColumnProjection("lift"),
            ),
            product_id_column="product_id_a",
        ),
    )
