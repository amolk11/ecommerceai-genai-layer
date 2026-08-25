"""Application facade: HTTP models → GenAIService → safe Business response."""

from app.api.models import ApplicationGenAIRequest, BusinessApplicationResponse
from collections.abc import Callable

from app.errors import UnsupportedCapabilityError
from app.observability import elapsed_ms, log_event
from app.models.request import GenAIRequest
from app.models.request_context import RequestContext
from app.models.user_context import UserContext
from app.services.genai import GenAIService
from app.state.business import BusinessState
from time import perf_counter
from pydantic import ValidationError
from app.errors import StructuredOutputError


class GenAIApplication:
    """Thin application boundary over the existing orchestration service."""

    def __init__(
        self,
        service: GenAIService | None = None,
        service_factory: Callable[[], GenAIService] | None = None,
    ) -> None:
        if service is None and service_factory is None:
            raise ValueError("A service or service factory is required.")
        self._service = service
        self._service_factory = service_factory

    def _get_service(self) -> GenAIService:
        """Compose the concrete provider on first execution, never during module import."""
        if self._service is None:
            assert self._service_factory is not None
            self._service = self._service_factory()
        return self._service

    def execute(self, request: ApplicationGenAIRequest) -> BusinessApplicationResponse:
        """Execute an application request without exposing orchestration internals."""
        started = perf_counter()
        log_event("workflow_started", persona=request.persona.value)
        try:
            result = self._get_service().handle(
                RequestContext(
                    user=UserContext(user_id=request.user_id, persona=request.persona),
                    request=GenAIRequest(message=request.message),
                )
            )
        except ValidationError as exc:
            raise StructuredOutputError(exc) from exc
        if not isinstance(result.state, BusinessState) or result.state.insight is None:
            raise UnsupportedCapabilityError()
        log_event(
            "workflow_completed",
            persona=request.persona.value,
            workspace=result.workspace.value,
            duration_ms=elapsed_ms(started),
            success=True,
        )
        return BusinessApplicationResponse(
            persona=request.persona,
            workspace=result.workspace,
            insight=result.state.insight,
        )
