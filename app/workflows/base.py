"""Base LangGraph workflow."""

from langgraph.graph import END, START, StateGraph

from app.state.base import BaseGenAIState


def initialize(state: BaseGenAIState) -> BaseGenAIState:
    """Initialize the workflow state."""
    return state


def create_base_graph() -> StateGraph:
    """Create the base GenAI workflow graph."""
    graph = StateGraph(BaseGenAIState)

    graph.add_node("initialize", initialize)

    graph.add_edge(START, "initialize")
    graph.add_edge("initialize", END)

    return graph


def build_base_graph():
    """Build and compile the base GenAI workflow."""
    return create_base_graph().compile()