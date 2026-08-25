"""Application composition root."""

from app.context.factory import create_business_context_provider
from collections.abc import Callable
from app.data.postgres import PostgresSettings, create_connection_factory
from app.context.protocols import BusinessContextProvider
from app.llm.factory import create_business_llm
from app.llm.protocols import BusinessLLM
from app.services.genai import GenAIService
from app.workflows.bootstrap import create_workflow_registry


def create_genai_service(
    business_llm: BusinessLLM | None = None,
    context_provider: BusinessContextProvider | None = None,
) -> GenAIService:
    """Create a fully wired GenAI application service."""
    return GenAIService(
        registry=create_workflow_registry(
            business_llm or create_business_llm(),
            context_provider or create_business_context_provider(),
        )
    )


def create_readiness_check() -> Callable[[], None]:
    """Create a lightweight PostgreSQL connectivity check without invoking an LLM."""
    def check() -> None:
        connection_factory = create_connection_factory(PostgresSettings.from_environment())
        with connection_factory() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1")

    return check
