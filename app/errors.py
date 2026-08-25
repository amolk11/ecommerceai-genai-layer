"""Safe, typed application errors for the GenAI execution boundary."""


class ApplicationError(Exception):
    """Base error with a stable code and safe public message."""

    code = "APPLICATION_ERROR"
    public_message = "The request could not be completed."
    status_code = 500

    def __init__(self, cause: Exception | None = None) -> None:
        self.cause = cause
        super().__init__(self.public_message)


class AuthorizationError(ApplicationError, PermissionError):
    code = "AUTHORIZATION_ERROR"
    public_message = "Request is not authorized."
    status_code = 403


class UnsupportedCapabilityError(ApplicationError, ValueError):
    code = "UNSUPPORTED_CAPABILITY"
    public_message = "This persona has no application-facing insight capability."
    status_code = 422


class WorkflowNotFoundError(ApplicationError, ValueError):
    code = "WORKFLOW_NOT_FOUND"
    public_message = "The requested workflow is unavailable."
    status_code = 503

    def __init__(self, cause: Exception | None = None) -> None:
        """Preserve the established service-layer workflow lookup message."""
        super().__init__(cause)
        if cause is not None:
            self.args = (str(cause),)


class ContextProviderError(ApplicationError):
    code = "CONTEXT_PROVIDER_ERROR"
    public_message = "A required service is currently unavailable."
    status_code = 503


class LLMProviderError(ApplicationError):
    code = "LLM_PROVIDER_ERROR"
    public_message = "The AI service could not complete the request."
    status_code = 502


class LLMConfigurationError(ApplicationError, RuntimeError):
    """A configured LLM provider cannot be safely initialized."""

    code = "LLM_CONFIGURATION_ERROR"
    public_message = "The AI service is not configured."
    status_code = 503

    def __init__(self, internal_message: str) -> None:
        """Retain a useful service-level message without exposing it at HTTP boundaries."""
        super().__init__()
        self.args = (internal_message,)


class StructuredOutputError(ApplicationError):
    code = "STRUCTURED_OUTPUT_ERROR"
    public_message = "The AI service returned an invalid structured response."
    status_code = 502
