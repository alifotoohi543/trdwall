"""Decision graph:

    bull -> bear --(disagreement)--> arbiter (strong model) --> risk -> executor
                 \\--(agreement)---------------------------->/
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .nodes import arbiter_node, bear_node, bull_node, executor_node, risk_node, route_after_debate
from .state import TradingState


def build_graph():
    g = StateGraph(TradingState)
    g.add_node("bull", bull_node)
    g.add_node("bear", bear_node)
    g.add_node("arbiter", arbiter_node)
    g.add_node("risk", risk_node)
    g.add_node("executor", executor_node)
    g.add_edge(START, "bull")
    g.add_edge("bull", "bear")
    g.add_conditional_edges("bear", route_after_debate, {"arbiter": "arbiter", "risk": "risk"})
    g.add_edge("arbiter", "risk")
    g.add_edge("risk", "executor")
    g.add_edge("executor", END)
    return g.compile()


_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph
