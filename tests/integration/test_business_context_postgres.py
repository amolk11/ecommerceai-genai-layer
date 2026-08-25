"""Read-only PostgreSQL smoke tests for the Business context integration."""

import os

import pytest

if not os.getenv("DATABASE_URL"):
    pytest.skip(
        "DATABASE_URL is not configured; PostgreSQL integration validation skipped.",
        allow_module_level=True,
    )

from app.bootstrap import create_genai_service
from app.context.factory import create_business_context_provider
from app.data.postgres import PostgresSettings, create_connection_factory
from app.data.schema import BusinessContextSchemaContract, ecommerce_business_context_schema
from app.models.business_context import BusinessContext
from app.models.business_insight import BusinessInsight
from app.models.request import GenAIRequest
from app.models.request_context import RequestContext
from app.models.user_context import UserContext
from app.routing.persona import Persona


class RecordingBusinessLLM:
    """Offline LLM double proving only structured context reaches the prompt."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate_business_insight(self, prompt: str) -> BusinessInsight:
        self.prompts.append(prompt)
        return BusinessInsight(
            summary="Integration smoke insight.",
            key_points=["Context was supplied."],
            recommended_actions=["Review available intelligence."],
        )


@pytest.fixture(scope="module")
def schema_contract() -> BusinessContextSchemaContract:
    """Return the code-owned mapping verified from the EcommerceAI schema export."""
    return ecommerce_business_context_schema()


@pytest.fixture(scope="module")
def real_user_id(schema_contract: BusinessContextSchemaContract) -> str:
    """Find one deterministic bounded customer identifier, or clearly skip empty data."""
    settings = PostgresSettings.from_environment()
    connection_factory = create_connection_factory(settings)
    column = schema_contract.customer_profile.user_filter_column()
    with connection_factory() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"SELECT {column} AS user_id FROM serving.customer_profile "
                f"WHERE {column} IS NOT NULL LIMIT %s",
                (1,),
            )
            row = cursor.fetchone()
    if row is None:
        pytest.skip("serving.customer_profile contains no customer records.")
    return str(row["user_id"])


def test_real_postgres_context_is_bounded_and_structured(
    schema_contract: BusinessContextSchemaContract, real_user_id: str
) -> None:
    """Real PostgreSQL rows produce a bounded Pydantic BusinessContext."""
    provider = create_business_context_provider(schema_contract)
    context = provider.build(real_user_id)

    assert isinstance(context, BusinessContext)
    assert context.customer_overview.user_id == real_user_id
    assert len(context.products) <= context.data_availability.product_limit
    assert len(context.recommendations) <= context.data_availability.recommendation_limit
    assert len(context.customer_overview.favorite_aisles) <= context.data_availability.product_limit
    assert len(context.customer_overview.favorite_departments) <= context.data_availability.product_limit
    assert "connection" not in context.model_dump()
    assert "DATABASE_URL" not in context.model_dump_json()


def test_real_postgres_context_executes_business_workflow_with_fake_llm(
    schema_contract: BusinessContextSchemaContract, real_user_id: str
) -> None:
    """Real context plus a fake LLM exercises the complete Business workflow safely."""
    llm = RecordingBusinessLLM()
    provider = create_business_context_provider(schema_contract)
    service = create_genai_service(llm, provider)

    result = service.handle(
        RequestContext(
            user=UserContext(user_id=real_user_id, persona=Persona.BUSINESS),
            request=GenAIRequest(message="Summarize the available business context."),
        )
    )

    assert result.state.business_context is not None
    assert result.state.insight is not None
    assert len(llm.prompts) == 1
    assert "Business context (bounded PostgreSQL records)" in llm.prompts[0]
    assert "DATABASE_URL" not in llm.prompts[0]
