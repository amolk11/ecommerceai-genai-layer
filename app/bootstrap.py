"""Application composition root."""

from app.context.factory import create_business_context_provider
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
