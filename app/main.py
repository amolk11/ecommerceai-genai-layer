"""FastAPI reliability boundary: request ID → service → safe response telemetry."""

import re
from collections.abc import Callable
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, Header, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api.models import (
    ApplicationErrorDetail,
    ApplicationErrorResponse,
    ApplicationGenAIRequest,
    BusinessApplicationResponse,
)
from app.api.service import GenAIApplication
from app.bootstrap import create_genai_service, create_readiness_check
from app.errors import ApplicationError
from app.observability import elapsed_ms, log_event, request_correlation
from app.services.genai import GenAIService


_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


def _request_id(value: str | None) -> str:
    """Use a safe client correlation ID or create a UUID without logging headers."""
    return value if value and _REQUEST_ID.fullmatch(value) else str(uuid4())


def _error_response(error: ApplicationError, request_id: str) -> JSONResponse:
    """Return a stable error envelope without revealing the internal cause."""
    response = ApplicationErrorResponse(
        error=ApplicationErrorDetail(
            code=error.code,
            message=error.public_message,
            request_id=request_id,
        )
    )
    return JSONResponse(
        status_code=error.status_code,
        content=response.model_dump(),
        headers={"X-Request-ID": request_id},
    )


def create_app(
    service: GenAIService | None = None,
    readiness_check: Callable[[], None] | None = None,
) -> FastAPI:
    """Create the API with dependencies composed outside handlers.

    HTTP delegates to GenAIApplication; authorization, workflows, context retrieval,
    and LLM execution remain in their existing layers. Readiness only checks PostgreSQL.
    """
    application = GenAIApplication(service or create_genai_service())
    check_readiness = readiness_check or create_readiness_check()
    app = FastAPI(title="EcommerceAI GenAI Layer", version="1.0.0")

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        """Correlate typed request-validation failures without serializing field internals."""
        request_id = _request_id(request.headers.get("X-Request-ID"))
        with request_correlation(request_id):
            log_event("request_failed", error_code="INVALID_REQUEST")
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=ApplicationErrorResponse(
                error=ApplicationErrorDetail(
                    code="INVALID_REQUEST",
                    message="The request is invalid.",
                    request_id=request_id,
                )
            ).model_dump(),
            headers={"X-Request-ID": request_id},
        )

    @app.get("/health")
    def health() -> dict[str, str]:
        """Return process liveness without contacting PostgreSQL or an LLM."""
        return {"status": "ok"}

    @app.get("/ready")
    def ready() -> JSONResponse:
        """Verify required connectivity safely; this endpoint never calls the LLM."""
        try:
            check_readiness()
        except Exception:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={"status": "not_ready"},
            )
        return JSONResponse(content={"status": "ready"})

    @app.post("/v1/genai", response_model=BusinessApplicationResponse)
    def execute_genai(
        request: ApplicationGenAIRequest,
        x_request_id: str | None = Header(default=None, alias="X-Request-ID"),
    ) -> BusinessApplicationResponse | JSONResponse:
        """Execute a persona-aware workflow with safe correlation and error handling."""
        request_id = _request_id(x_request_id)
        started = perf_counter()
        with request_correlation(request_id):
            log_event("request_received", persona=request.persona.value)
            try:
                response = application.execute(request)
            except ApplicationError as exc:
                log_event(
                    "request_failed",
                    error_code=exc.code,
                    duration_ms=elapsed_ms(started),
                    success=False,
                )
                return _error_response(exc, request_id)
            except Exception:
                error = ApplicationError()
                log_event(
                    "request_failed",
                    error_code=error.code,
                    duration_ms=elapsed_ms(started),
                    success=False,
                )
                return _error_response(error, request_id)
            log_event(
                "request_completed",
                workspace=response.workspace.value,
                duration_ms=elapsed_ms(started),
                success=True,
            )
        return JSONResponse(
            content=response.model_dump(mode="json"),
            headers={"X-Request-ID": request_id},
        )

    return app
