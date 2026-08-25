"""Business workspace workflow."""

from langgraph.graph import END, START, StateGraph

from app.llm.protocols import BusinessLLM
from app.models.business_insight import BusinessInsight
from app.prompts.business import build_business_intelligence_prompt
from app.state.business import BusinessState


def initialize(state: BusinessState) -> BusinessState:
    """Initialize the business workflow."""
    return state


def business_intelligence(state: BusinessState, llm: BusinessLLM) -> BusinessState:
    """Generate and store a validated insight for the business request."""
    prompt = build_business_intelligence_prompt(state.context.request.message)
    state.insight = BusinessInsight.model_validate(llm.generate_business_insight(prompt))
    return state


def create_business_graph(llm: BusinessLLM) -> StateGraph:
    """Create the business workspace graph."""
    graph = StateGraph(BusinessState)

    graph.add_node("initialize", initialize)
    graph.add_node("business_intelligence", lambda state: business_intelligence(state, llm))

    graph.add_edge(START, "initialize")
    graph.add_edge("initialize", "business_intelligence")
    graph.add_edge("business_intelligence", END)

    return graph


def build_business_graph(llm: BusinessLLM):
    """Build and compile the business workspace graph."""
    return create_business_graph(llm).compile()
