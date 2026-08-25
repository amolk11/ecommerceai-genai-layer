"""Application-boundary tests for the FastAPI GenAI entrypoint."""

import logging

from fastapi.testclient import TestClient
import pytest

from app.bootstrap import create_genai_service
from app.main import create_app
from app.models.business_context import (
    BusinessContext,
    CustomerBehavioralIntelligence,
    CustomerBusinessOverview,
    DataAvailabilityMetadata,
)
from app.models.business_insight import BusinessInsight
from app.models.user_context import UserContext
from app.routing.workspace import Workspace
from app.services.genai import GenAIService
from app.workflows.registry import WorkflowRegistry


class RecordingContextProvider:
    """Context double representing the workflow's injected context boundary."""

    def __init__(self) -> None:
        self.user_ids: list[str] = []

    def build(self, user_id: str) -> BusinessContext:
        self.user_ids.append(user_id)
        return BusinessContext(
            customer_overview=CustomerBusinessOverview(
                user_id=user_id,
                observed_profile={"total_orders": 12},
            ),
            behavioral_intelligence=CustomerBehavioralIntelligence(
                observed_behavior={"purchase_loyalty": "high"},
                model_derived_scores={"loyalty_score": 0.8},
                segments={"lifecycle_segment": "established"},
            ),
            data_availability=DataAvailabilityMetadata(
                serving_sources=["serving.customer_profile"],
                analytics_sources=["analytics.customer_behavior"],
                product_limit=5,
                recommendation_limit=5,
            ),
        )


class RecordingBusinessLLM:
    """LLM double that proves the workflow receives bounded context in its prompt."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate_business_insight(self, prompt: str) -> BusinessInsight:
        self.prompts.append(prompt)
        return BusinessInsight(
            summary="Customer loyalty is strong.",
            key_points=["Repeat behavior is established."],
            recommended_actions=["Maintain retention initiatives."],
        )


def client_with_business_dependencies() -> tuple[TestClient, RecordingContextProvider, RecordingBusinessLLM]:
    """Create an HTTP client backed by the actual application orchestration path."""
    provider = RecordingContextProvider()
    llm = RecordingBusinessLLM()
    service = create_genai_service(llm, provider)
    return TestClient(create_app(service)), provider, llm


def business_payload() -> dict[str, str]:
    """Return a valid application request payload."""
    return {
        "user_id": "42",
        "persona": "business",
        "message": "How should we retain valuable customers?",
    }


def test_health_is_live_without_workflow_execution() -> None:
    """Health is a liveness endpoint and does not invoke the Business workflow."""
    client, provider, llm = client_with_business_dependencies()

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert provider.user_ids == []
    assert llm.prompts == []


def test_default_app_import_and_health_do_not_require_an_openrouter_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The default composition defers provider configuration until a GenAI request."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)

    response = TestClient(create_app()).get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_missing_openrouter_configuration_is_a_safe_api_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Configuration is explicit for service callers but never exposes a key via HTTP."""
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)

    response = TestClient(create_app()).post("/v1/genai", json=business_payload())

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "LLM_CONFIGURATION_ERROR"
    assert response.json()["error"]["message"] == "The AI service is not configured."
    assert "OPENROUTER_API_KEY" not in response.text


def test_invalid_request_is_rejected_by_the_typed_boundary() -> None:
    """Pydantic request validation rejects malformed HTTP payloads before execution."""
    client, provider, llm = client_with_business_dependencies()

    response = client.post(
        "/v1/genai",
        json={"persona": "business"},
        headers={"X-Request-ID": "invalid-request"},
    )

    assert response.status_code == 422
    assert response.headers["X-Request-ID"] == "invalid-request"
    assert response.json() == {
        "error": {
            "code": "INVALID_REQUEST",
            "message": "The request is invalid.",
            "request_id": "invalid-request",
        }
    }
    assert provider.user_ids == []
    assert llm.prompts == []


def test_business_request_uses_service_context_and_returns_structured_insight() -> None:
    """HTTP delegates through GenAIService to the workflow and validated response."""
    client, provider, llm = client_with_business_dependencies()

    response = client.post("/v1/genai", json=business_payload())

    assert response.status_code == 200
    assert response.json() == {
        "persona": "business",
        "workspace": "business",
        "insight": {
            "summary": "Customer loyalty is strong.",
            "key_points": ["Repeat behavior is established."],
            "recommended_actions": ["Maintain retention initiatives."],
        },
    }
    assert provider.user_ids == ["42"]
    assert len(llm.prompts) == 1
    assert '"loyalty_score":0.8' in llm.prompts[0]


def test_request_id_is_propagated_or_generated() -> None:
    """The API preserves safe client IDs and generates one when absent."""
    client, _, _ = client_with_business_dependencies()

    supplied = client.post(
        "/v1/genai", json=business_payload(), headers={"X-Request-ID": "request-42"}
    )
    generated = client.post("/v1/genai", json=business_payload())

    assert supplied.headers["X-Request-ID"] == "request-42"
    assert generated.headers["X-Request-ID"]
    assert generated.headers["X-Request-ID"] != "request-42"


def test_missing_workflow_is_a_safe_service_error() -> None:
    """Registry failures never expose orchestration implementation details."""
    client = TestClient(create_app(GenAIService(registry=WorkflowRegistry())))

    response = client.post("/v1/genai", json=business_payload())

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "WORKFLOW_NOT_FOUND"
    assert response.json()["error"]["message"] == "The requested workflow is unavailable."
    assert response.headers["X-Request-ID"] == response.json()["error"]["request_id"]


class IncorrectRouter:
    """Router double that deliberately violates persona/workspace authorization."""

    def route(self, context: UserContext) -> Workspace:
        return Workspace.DEVELOPER


def test_unauthorized_workspace_is_rejected_before_workflow_execution() -> None:
    """Authorization failures become safe 403 application responses."""
    service = GenAIService(registry=WorkflowRegistry(), router=IncorrectRouter())

    response = TestClient(create_app(service)).post("/v1/genai", json=business_payload())

    assert response.status_code == 403
    assert response.json()["error"]["code"] == "AUTHORIZATION_ERROR"
    assert response.headers["X-Request-ID"] == response.json()["error"]["request_id"]


class FailingBusinessLLM:
    """LLM double that simulates a provider outage."""

    def generate_business_insight(self, prompt: str) -> BusinessInsight:
        raise RuntimeError("LLM provider unavailable")


def test_llm_failure_is_mapped_without_provider_details() -> None:
    """LLM failures become safe application errors."""
    service = create_genai_service(FailingBusinessLLM(), RecordingContextProvider())
    response = TestClient(create_app(service)).post("/v1/genai", json=business_payload())

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "LLM_PROVIDER_ERROR"


class InvalidBusinessLLM:
    """LLM double that bypasses its protocol with malformed structured data."""

    def generate_business_insight(self, prompt: str) -> object:
        return {"summary": "Missing list fields"}


def test_invalid_llm_output_is_rejected_at_the_application_boundary() -> None:
    """Pydantic validation failures never become successful API responses."""
    service = create_genai_service(InvalidBusinessLLM(), RecordingContextProvider())

    response = TestClient(create_app(service)).post("/v1/genai", json=business_payload())

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "STRUCTURED_OUTPUT_ERROR"


class FailingContextProvider:
    """Context double that includes deliberately sensitive-looking failure text."""

    def build(self, user_id: str) -> BusinessContext:
        raise RuntimeError("database failure for postgresql://user:secret@host/database")


def test_context_failure_does_not_leak_database_details() -> None:
    """Database/context errors are safe at the HTTP boundary."""
    service = create_genai_service(RecordingBusinessLLM(), FailingContextProvider())
    response = TestClient(create_app(service)).post("/v1/genai", json=business_payload())

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "CONTEXT_PROVIDER_ERROR"
    assert "postgresql" not in response.text
    assert "secret" not in response.text


def test_customer_request_cannot_execute_business_workflow() -> None:
    """Unsupported personas do not invoke the Business LLM or context provider."""
    client, provider, llm = client_with_business_dependencies()
    payload = business_payload() | {"persona": "customer"}

    response = client.post("/v1/genai", json=payload)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "UNSUPPORTED_CAPABILITY"
    assert provider.user_ids == []
    assert llm.prompts == []


def test_readiness_is_safe_and_never_requires_llm_or_context() -> None:
    """Readiness uses an injected dependency check, distinct from liveness."""
    client, provider, llm = client_with_business_dependencies()
    calls: list[bool] = []
    app = create_app(
        create_genai_service(llm, provider), readiness_check=lambda: calls.append(True)
    )

    response = TestClient(app).get("/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}
    assert calls == [True]
    assert provider.user_ids == []
    assert llm.prompts == []


def test_readiness_failure_is_safe() -> None:
    """Readiness failures do not disclose database configuration."""
    client, provider, llm = client_with_business_dependencies()
    app = create_app(
        create_genai_service(llm, provider),
        readiness_check=lambda: (_ for _ in ()).throw(
            RuntimeError("postgresql://user:secret@host/database")
        ),
    )

    response = TestClient(app).get("/ready")

    assert response.status_code == 503
    assert response.json() == {"status": "not_ready"}
    assert "secret" not in response.text


class ExplodingService:
    """Service double that raises an unexpected internal exception."""

    def handle(self, context: object) -> object:
        raise RuntimeError("unexpected internal failure")


def test_unexpected_error_is_generic_and_correlated() -> None:
    """Unhandled failures never leak internal exception text."""
    response = TestClient(create_app(ExplodingService())).post(
        "/v1/genai", json=business_payload(), headers={"X-Request-ID": "unexpected-1"}
    )

    assert response.status_code == 500
    assert response.json() == {
        "error": {
            "code": "APPLICATION_ERROR",
            "message": "The request could not be completed.",
            "request_id": "unexpected-1",
        }
    }
    assert "unexpected internal" not in response.text


def test_lifecycle_logs_are_correlated_and_do_not_include_context(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Telemetry emits safe lifecycle events without prompts or business payloads."""
    caplog.set_level(logging.INFO, logger="ecommerceai.genai")
    client, _, _ = client_with_business_dependencies()

    response = client.post(
        "/v1/genai", json=business_payload(), headers={"X-Request-ID": "log-42"}
    )

    assert response.status_code == 200
    messages = "\n".join(record.message for record in caplog.records)
    assert "request_received" in messages
    assert "workflow_started" in messages
    assert "context_loaded" in messages
    assert "llm_completed" in messages
    assert "request_completed" in messages
    assert "log-42" in messages
    assert "total_orders" not in messages
