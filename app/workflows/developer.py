"""Developer workspace workflow."""

from langgraph.graph import END, START, StateGraph

from app.state.developer import DeveloperState


def initialize(state: DeveloperState) -> DeveloperState:
    """Initialize the developer workflow."""
    return state


def create_developer_graph() -> StateGraph:
    """Create the developer workspace graph."""
    graph = StateGraph(DeveloperState)

    graph.add_node("initialize", initialize)

    graph.add_edge(START, "initialize")
    graph.add_edge("initialize", END)

    return graph


def build_developer_graph():
    """Build and compile the developer workspace graph."""
    return create_developer_graph().compile()
