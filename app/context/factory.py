"""Composition helpers for PostgreSQL-backed context providers."""

from app.context.business import PostgresBusinessContextProvider
from app.context.protocols import BusinessContextProvider
from app.data.postgres import (
    PostgresBusinessRepository,
    PostgresSettings,
    create_connection_factory,
)
from app.data.schema import BusinessContextSchemaContract, ecommerce_business_context_schema
from app.data.validation import PostgresSchemaValidator


def create_business_context_provider(
    schema_contract: BusinessContextSchemaContract | None = None,
) -> BusinessContextProvider:
    """Create the configured PostgreSQL Business context provider."""
    schema_contract = schema_contract or ecommerce_business_context_schema()
    settings = PostgresSettings.from_environment()
    connection_factory = create_connection_factory(settings)
    PostgresSchemaValidator(connection_factory, schema_contract).validate()
    return PostgresBusinessContextProvider(
        PostgresBusinessRepository(connection_factory, schema_contract)
    )
