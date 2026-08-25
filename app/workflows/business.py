"""Business workspace workflow."""

from langgraph.graph import END, START, StateGraph

from app.context.protocols import BusinessContextProvider
from app.llm.protocols import BusinessLLM
from app.models.business_insight import BusinessInsight
from app.prompts.business import build_business_intelligence_prompt
from app.state.business import BusinessState


def initialize(state: BusinessState) -> BusinessState:
    """Initialize the business workflow."""
    return state


def build_business_context(
    state: BusinessState, context_provider: BusinessContextProvider
) -> BusinessState:
    """Load the compact PostgreSQL-backed context for the business user."""
    state.business_context = context_provider.build(state.context.user.user_id)
    return state


def business_intelligence(state: BusinessState, llm: BusinessLLM) -> BusinessState:
    """Generate and store a validated insight for the business request."""
    if state.business_context is None:
        raise RuntimeError("Business context must be built before LLM execution.")
    prompt = build_business_intelligence_prompt(
        state.context.request.message, state.business_context
    )
    state.insight = BusinessInsight.model_validate(llm.generate_business_insight(prompt))
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
