"""Composition helpers for PostgreSQL-backed context providers."""

from app.context.business import PostgresBusinessContextProvider
from app.context.protocols import BusinessContextProvider
from app.data.postgres import PostgresBusinessRepository, PostgresSettings
from app.data.schema import BusinessContextSchemaContract


def create_business_context_provider(
    schema_contract: BusinessContextSchemaContract | None = None,
) -> BusinessContextProvider:
    """Create the configured PostgreSQL Business context provider."""
    if schema_contract is None:
        raise RuntimeError(
            "PostgreSQL BusinessContext column mappings must be verified and supplied "
            "as a BusinessContextSchemaContract before production use."
        )
    settings = PostgresSettings.from_environment()
    return PostgresBusinessContextProvider(
        PostgresBusinessRepository.from_settings(settings, schema_contract)
    )
