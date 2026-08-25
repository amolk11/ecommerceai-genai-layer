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


@dataclass(frozen=True)
class BusinessContextSchemaContract:
    """Verified column mappings supplied after inspecting the PostgreSQL schema."""

    customer_profile: TableProjection
    customer_behavior: TableProjection
    customer_scores: TableProjection
    customer_segments: TableProjection
    favorite_products: TableProjection
    favorite_aisles: TableProjection
    favorite_departments: TableProjection
    product_intelligence: TableProjection
    recommendations: TableProjection
