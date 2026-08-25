"""Unit tests for the OpenRouter BusinessLLM adapter and selection boundary."""

from types import SimpleNamespace

import pytest

from app.errors import LLMConfigurationError
from app.llm.config import OpenRouterSettings
from app.llm.factory import create_business_llm
from app.llm.openrouter import OpenRouterBusinessLLM
from app.llm.protocols import BusinessLLM
from app.models.business_insight import BusinessInsight


class FakeCompletions:
    """OpenAI-compatible completion endpoint that records the request."""

    def __init__(self, content: object) -> None:
        self.content = content
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))]
        )


def fake_client(content: object) -> tuple[object, FakeCompletions]:
    """Return the narrow client shape consumed by the adapter."""
    completions = FakeCompletions(content)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return client, completions


def test_openrouter_settings_loads_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "router-secret")
    monkeypatch.setenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")

    settings = OpenRouterSettings.from_environment()

    assert settings.api_key == "router-secret"
    assert settings.model == "openai/gpt-4o-mini"


@pytest.mark.parametrize(
    ("variable", "message"),
    [
        ("OPENROUTER_API_KEY", "OPENROUTER_API_KEY must be set"),
        ("OPENROUTER_MODEL", "OPENROUTER_MODEL must be set"),
    ],
)
def test_openrouter_settings_require_key_and_model(
    monkeypatch: pytest.MonkeyPatch, variable: str, message: str
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "router-secret")
    monkeypatch.setenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
    monkeypatch.delenv(variable, raising=False)

    with pytest.raises(LLMConfigurationError, match=message) as exc_info:
        OpenRouterSettings.from_environment()

    assert "router-secret" not in str(exc_info.value)
    assert exc_info.value.public_message == "The AI service is not configured."


def test_factory_selects_openrouter_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Typed settings retain explicit, provider-agnostic factory selection."""
    expected = object()
    monkeypatch.setattr(
        "app.llm.openrouter.OpenRouterBusinessLLM", lambda settings: expected
    )

    result = create_business_llm(
        OpenRouterSettings(api_key="router-secret", model="openai/gpt-4o-mini")
    )

    assert result is expected


def test_factory_rejects_an_unsupported_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "unsupported-provider")

    with pytest.raises(LLMConfigurationError, match="Unsupported LLM_PROVIDER") as exc_info:
        create_business_llm()

    assert "unsupported-provider" not in exc_info.value.public_message


def test_openrouter_adapter_uses_business_llm_protocol_and_validates_output() -> None:
    content = (
        '{"summary":"A concise summary","key_points":["Observed behavior"],'
        '"recommended_actions":["Take action"]}'
    )
    client, completions = fake_client(content)
    adapter = OpenRouterBusinessLLM(
        OpenRouterSettings(api_key="router-secret", model="openai/gpt-4o-mini"),
        client=client,
    )

    result = adapter.generate_business_insight("bounded business prompt")

    assert isinstance(adapter, BusinessLLM)
    assert result == BusinessInsight(
        summary="A concise summary",
        key_points=["Observed behavior"],
        recommended_actions=["Take action"],
    )
    request = completions.calls[0]
    assert request["model"] == "openai/gpt-4o-mini"
    assert request["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "business_insight",
            "strict": True,
            "schema": BusinessInsight.model_json_schema(),
        },
    }


def test_openrouter_adapter_rejects_invalid_provider_output() -> None:
    client, _ = fake_client('{"summary":"Only summary"}')
    adapter = OpenRouterBusinessLLM(
        OpenRouterSettings(api_key="router-secret", model="openai/gpt-4o-mini"),
        client=client,
    )

    with pytest.raises(ValueError, match="invalid business insight"):
        adapter.generate_business_insight("prompt")


def test_openrouter_errors_do_not_include_api_keys() -> None:
    class FailingCompletions:
        def create(self, **kwargs: object) -> object:
            raise RuntimeError("provider rejected router-secret")

    client = SimpleNamespace(chat=SimpleNamespace(completions=FailingCompletions()))
    adapter = OpenRouterBusinessLLM(
        OpenRouterSettings(api_key="router-secret", model="openai/gpt-4o-mini"),
        client=client,
    )

    with pytest.raises(RuntimeError) as exc_info:
        adapter.generate_business_insight("prompt")

    assert "router-secret" not in str(exc_info.value)


def test_openrouter_adapter_constructs_default_client_without_injection() -> None:
    settings = OpenRouterSettings(api_key="router-secret", model="openai/gpt-4o-mini")
    adapter = OpenRouterBusinessLLM(settings)

    assert isinstance(adapter, BusinessLLM)
    assert adapter._client is not None

