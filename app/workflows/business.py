"""Business workspace workflow."""

from langgraph.graph import END, START, StateGraph

from app.state.business import BusinessState


def initialize(state: BusinessState) -> BusinessState:
    """Initialize the business workflow."""
    return state


def create_business_graph() -> StateGraph:
    """Create the business workspace graph."""
    graph = StateGraph(BusinessState)

    graph.add_node("initialize", initialize)

    graph.add_edge(START, "initialize")
    graph.add_edge("initialize", END)

    return graph


def build_business_graph():
    """Build and compile the business workspace graph."""
    return create_business_graph().compile()