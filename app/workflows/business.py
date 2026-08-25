"""Business workspace workflow."""

from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from app.context.protocols import BusinessContextProvider
from app.errors import ContextProviderError, LLMProviderError, StructuredOutputError
from app.llm.protocols import BusinessLLM
from app.models.business_insight import BusinessInsight
from app.prompts.business import build_business_intelligence_prompt
from app.state.business import BusinessState
from app.observability import elapsed_ms, log_event
from time import perf_counter


def initialize(state: BusinessState) -> BusinessState:
    """Initialize the business workflow."""
    return state


def build_business_context(
    state: BusinessState, context_provider: BusinessContextProvider
) -> BusinessState:
    """Load the compact PostgreSQL-backed context for the business user."""
    started = perf_counter()
    try:
        state.business_context = context_provider.build(state.context.user.user_id)
    except Exception as exc:
        log_event("request_failed", error_code=ContextProviderError.code)
        raise ContextProviderError(exc) from exc
    log_event("context_loaded", duration_ms=elapsed_ms(started), success=True)
    return state


def business_intelligence(state: BusinessState, llm: BusinessLLM) -> BusinessState:
    """Generate and store a validated insight for the business request."""
    if state.business_context is None:
        raise RuntimeError("Business context must be built before LLM execution.")
    prompt = build_business_intelligence_prompt(
        state.context.request.message, state.business_context
    )
    started = perf_counter()
    log_event("llm_started")
    try:
        output = llm.generate_business_insight(prompt)
    except Exception as exc:
        log_event("request_failed", error_code=LLMProviderError.code)
        raise LLMProviderError(exc) from exc
    try:
        state.insight = BusinessInsight.model_validate(output)
    except ValidationError as exc:
        log_event("request_failed", error_code=StructuredOutputError.code)
        raise
    log_event("llm_completed", duration_ms=elapsed_ms(started), success=True)
    return state


def create_business_graph(
    llm: BusinessLLM, context_provider: BusinessContextProvider
) -> StateGraph:
    """Create the business workspace graph."""
    graph = StateGraph(BusinessState)

    graph.add_node("initialize", initialize)
    graph.add_node(
        "build_business_context",
        lambda state: build_business_context(state, context_provider),
    )
    graph.add_node(
        "business_intelligence", lambda state: business_intelligence(state, llm)
    )

    graph.add_edge(START, "initialize")
    graph.add_edge("initialize", "build_business_context")
    graph.add_edge("build_business_context", "business_intelligence")
    graph.add_edge("business_intelligence", END)

    return graph


def build_business_graph(llm: BusinessLLM, context_provider: BusinessContextProvider):
    """Build and compile the business workspace graph."""
    return create_business_graph(llm, context_provider).compile()
