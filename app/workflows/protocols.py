"""Contracts for GenAI workflows."""

from typing import Protocol

from app.state.base import BaseGenAIState


class Workflow(Protocol):
    """Contract that every GenAI workflow must satisfy."""

    def invoke(self, state: BaseGenAIState) -> BaseGenAIState:
        """Execute the workflow and return updated state."""
        ...