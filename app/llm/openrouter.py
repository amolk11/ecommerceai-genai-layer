"""OpenRouter implementation of the provider-agnostic business LLM boundary."""

from typing import Any

from openai import OpenAI

from app.llm.config import OpenRouterSettings
from app.models.business_insight import BusinessInsight


class OpenRouterBusinessLLM:
    """Use OpenRouter's OpenAI-compatible chat completions API with strict JSON Schema output."""

    _provider_name = "OpenRouter"
    _BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(
        self,
        settings: OpenRouterSettings,
        client: Any | None = None,
    ) -> None:
        self._settings = settings
        if client is not None:
            self._client = client
        else:
            self._client = OpenAI(
                base_url=self._BASE_URL,
                api_key=settings.api_key,
            )

    def generate_business_insight(self, prompt: str) -> BusinessInsight:
        """Request and validate a JSON-schema constrained business insight."""
        try:
            response = self._client.chat.completions.create(
                model=self._settings.model,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "business_insight",
                        "strict": True,
                        "schema": BusinessInsight.model_json_schema(),
                    },
                },
            )
        except Exception:
            raise RuntimeError(f"{self._provider_name} LLM invocation failed.") from None

        try:
            content = self._extract_content(response)
            return BusinessInsight.model_validate_json(content)
        except (AttributeError, IndexError, KeyError, TypeError, ValueError):
            raise ValueError(f"{self._provider_name} LLM returned an invalid business insight.") from None

    @staticmethod
    def _extract_content(response: Any) -> str:
        """Extract the structured message content from an OpenAI-compatible completion response."""
        if response is None:
            raise ValueError("No response returned.")
        choices = getattr(response, "choices", None)
        if choices is None and isinstance(response, dict):
            choices = response.get("choices")
        if not choices:
            raise ValueError("No choices returned.")
        first_choice = choices[0]
        message = getattr(first_choice, "message", None)
        if message is None and isinstance(first_choice, dict):
            message = first_choice.get("message")
        if message is None:
            raise ValueError("No message returned.")
        content = getattr(message, "content", None)
        if content is None and isinstance(message, dict):
            content = message.get("content")
        if content is None or not isinstance(content, str):
            raise ValueError("No text content returned.")
        return content
