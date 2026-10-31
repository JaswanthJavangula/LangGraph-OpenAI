"""LangGraph workflow definition."""

from __future__ import annotations

import os
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages


class GraphState(TypedDict):
    """State passed between workflow nodes."""

    messages: Annotated[list[AnyMessage], add_messages]


def build_graph():
    """Build and compile the response workflow."""
    model = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)

    def respond(state: GraphState) -> dict[str, list[AnyMessage]]:
        response = model.invoke(state["messages"])
        return {"messages": [response]}

    workflow = StateGraph(GraphState)
    workflow.add_node("respond", respond)
    workflow.add_edge(START, "respond")
    workflow.add_edge("respond", END)
    return workflow.compile()
