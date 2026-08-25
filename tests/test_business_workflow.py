"""Tests for the LLM-backed Business workflow capability."""

import pytest
from pydantic import ValidationError

from app.bootstrap import create_genai_service
from app.llm.config import OpenAISettings
from app.models.business_insight import BusinessInsight
from app.models.business_context import (
    BusinessContext,
    CustomerBehavioralIntelligence,
    CustomerBusinessOverview,
    DataAvailabilityMetadata,
)
from app.models.request import GenAIRequest
from app.models.request_context import RequestContext
from app.models.user_context import UserContext
from app.routing.persona import Persona
from app.state.business import BusinessState
from app.state.customer import CustomerState
from app.state.developer import DeveloperState


def business_context() -> RequestContext:
    """Create the business request used by this vertical slice."""
    return RequestContext(
        user=UserContext(user_id="business-user", persona=Persona.BUSINESS),
        request=GenAIRequest(message="How can I improve customer retention?"),
    )


class RecordingBusinessLLM:
    """Fake LLM that records the prompt it was given."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate_business_insight(self, prompt: str) -> BusinessInsight:
        self.prompts.append(prompt)
        return BusinessInsight(
            summary="Retention is improved by increasing repeat engagement.",
            key_points=["Segment customers by engagement", "Measure churn reasons"],
            recommended_actions=["Launch a re-engagement campaign"],
        )


class InvalidBusinessLLM:
    """Fake LLM that returns a malformed structured response."""

    def generate_business_insight(self, prompt: str) -> object:
        return {"summary": "Missing required fields"}


class FakeBusinessContextProvider:
    """Database-free context provider for Business workflow tests."""

    def build(self, user_id: str) -> BusinessContext:
        return BusinessContext(
            customer_overview=CustomerBusinessOverview(
                user_id=user_id,
                observed_profile={"total_orders": 12},
            ),
            behavioral_intelligence=CustomerBehavioralIntelligence(
                observed_behavior={"reorder_rate": 0.4},
                model_derived_scores={"retention_score": 0.8},
                segments={"segment": "loyal"},
            ),
            data_availability=DataAvailabilityMetadata(
                serving_sources=["serving.customer_profile"],
                analytics_sources=["analytics.customer_behavior"],
                product_limit=5,
                recommendation_limit=5,
            ),
        )


def test_business_service_returns_a_structured_insight() -> None:
    """A Business request reaches the injected LLM and returns its insight."""
    llm = RecordingBusinessLLM()

    result = create_genai_service(llm, FakeBusinessContextProvider()).handle(
        business_context()
    )

    assert isinstance(result.state, BusinessState)
    assert result.state.insight is not None
    assert result.state.insight.summary.startswith("Retention")
    assert result.state.insight.key_points == [
        "Segment customers by engagement",
        "Measure churn reasons",
    ]
    assert result.state.insight.recommended_actions == [
        "Launch a re-engagement campaign"
    ]
    assert len(llm.prompts) == 1
    assert "How can I improve customer retention?" in llm.prompts[0]
    assert '"retention_score":0.8' in llm.prompts[0]


def test_business_workflow_uses_the_injected_llm() -> None:
    """Workflow construction does not create a provider behind the caller's back."""
    llm = RecordingBusinessLLM()

    create_genai_service(llm, FakeBusinessContextProvider()).handle(business_context())

    assert len(llm.prompts) == 1


def test_invalid_llm_output_fails_validation() -> None:
    """Malformed model output cannot silently become a BusinessInsight."""
    with pytest.raises(ValidationError):
        create_genai_service(
            InvalidBusinessLLM(), FakeBusinessContextProvider()
        ).handle(business_context())


def test_customer_and_developer_workflows_do_not_use_business_llm() -> None:
    """Only the Business graph invokes the LLM in this milestone."""
    llm = RecordingBusinessLLM()
    service = create_genai_service(llm, FakeBusinessContextProvider())

    customer = service.handle(
        RequestContext(
            user=UserContext(user_id="customer-user", persona=Persona.CUSTOMER),
            request=GenAIRequest(message="Show products"),
        )
    )
    developer = service.handle(
        RequestContext(
            user=UserContext(user_id="developer-user", persona=Persona.DEVELOPER),
            request=GenAIRequest(message="Inspect code"),
        )
    )

    assert isinstance(customer.state, CustomerState)
    assert isinstance(developer.state, DeveloperState)
    assert llm.prompts == []


def test_openai_settings_fails_clearly_without_an_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Provider construction reports unavailable credentials explicitly."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY must be set"):
        OpenAISettings.from_environment()
