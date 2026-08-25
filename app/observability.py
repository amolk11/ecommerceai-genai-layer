"""Small safe logging and timing helpers for request execution."""

import json
import logging
from contextlib import contextmanager
from contextvars import ContextVar
from time import perf_counter
from typing import Iterator


_logger = logging.getLogger("ecommerceai.genai")
_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)


@contextmanager
def request_correlation(request_id: str) -> Iterator[None]:
    """Make a request ID available to application and workflow telemetry only."""
    token = _request_id.set(request_id)
    try:
        yield
    finally:
        _request_id.reset(token)


def log_event(event: str, **fields: object) -> None:
    """Emit a compact structured event using only explicitly supplied safe fields."""
    payload = {"event": event, "request_id": _request_id.get(), **fields}
    _logger.info("%s", json.dumps(payload, default=str, sort_keys=True))


def elapsed_ms(start: float) -> int:
    """Return elapsed monotonic time in milliseconds."""
    return round((perf_counter() - start) * 1000)
