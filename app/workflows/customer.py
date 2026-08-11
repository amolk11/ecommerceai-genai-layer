"""Customer workspace workflow."""

from langgraph.graph import END, START, StateGraph

from app.state.customer import CustomerState


def initialize(state: CustomerState) -> CustomerState:
    """Initialize the customer workflow."""
    return state


def create_customer_graph() -> StateGraph:
    """Create the customer workspace graph."""
    graph = StateGraph(CustomerState)

    graph.add_node("initialize", initialize)

    graph.add_edge(START, "initialize")
    graph.add_edge("initialize", END)

    return graph


def build_customer_graph():
    """Build and compile the customer workspace graph."""
    return create_customer_graph().compile()