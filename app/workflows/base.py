"""Base LangGraph workflow."""

from langgraph.graph import END, START, StateGraph

from app.state.base import BaseGenAIState


def build_base_graph():
    """Build the minimal GenAI workflow graph."""

    graph = StateGraph(BaseGenAIState)

    graph.add_node("initialize", lambda state: state)

    graph.add_edge(START, "initialize")
    graph.add_edge("initialize", END)

    return graph.compile()