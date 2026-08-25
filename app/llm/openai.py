"""OpenAI Responses API implementation of the business LLM boundary."""

import json
from collections.abc import Callable
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import ValidationError

from app.models.business_insight import BusinessInsight


class ResponsesSettings(Protocol):
    """The credentials and model required by OpenAI-compatible Responses APIs."""

    api_key: str
    model: str


class OpenAICompatibleResponsesBusinessLLM:
    """Shared structured-output adapter for OpenAI-compatible Responses APIs."""

    _URL: str
    _provider_name: str

    def __init__(
        self,
        settings: ResponsesSettings,
        request_opener: Callable[..., Any] = urlopen,
    ) -> None:
        self._settings = settings
        self._request_opener = request_opener

    def generate_business_insight(self, prompt: str) -> BusinessInsight:
        """Request and validate a JSON-schema constrained business insight."""
        payload = {
            "model": self._settings.model,
            "input": prompt,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "business_insight",
                    "strict": True,
                    "schema": BusinessInsight.model_json_schema(),
                }
            },
        }
        request = Request(
            self._URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._settings.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with self._request_opener(request, timeout=30) as response:
                body: dict[str, Any] = json.loads(response.read())
        except HTTPError as exc:
            raise RuntimeError(f"{self._provider_name} LLM request failed with status {exc.code}.") from exc
        except URLError as exc:
            raise RuntimeError(f"{self._provider_name} LLM is unavailable.") from exc
        except OSError as exc:
            raise RuntimeError(f"{self._provider_name} LLM invocation failed.") from exc

        try:
            return BusinessInsight.model_validate_json(self._output_text(body))
        except (KeyError, TypeError, json.JSONDecodeError, ValidationError) as exc:
            raise ValueError(f"{self._provider_name} LLM returned an invalid business insight.") from exc

    @staticmethod
    def _output_text(body: dict[str, Any]) -> str:
        """Extract the first text item from a completed Responses API result."""
        for item in body["output"]:
            for content in item.get("content", []):
                if content.get("type") == "output_text":
                    return content["text"]
        raise KeyError("No output_text content was returned.")


class OpenAIResponsesBusinessLLM(OpenAICompatibleResponsesBusinessLLM):
    """Generate structured business insights through the OpenAI Responses API."""

    _URL = "https://api.openai.com/v1/responses"
    _provider_name = "OpenAI"
