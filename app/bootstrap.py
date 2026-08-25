"""Application composition root."""

from app.services.genai import GenAIService
from app.workflows.bootstrap import create_workflow_registry


def create_genai_service() -> GenAIService:
    """Create a fully wired GenAI application service."""
    return GenAIService(registry=create_workflow_registry())
