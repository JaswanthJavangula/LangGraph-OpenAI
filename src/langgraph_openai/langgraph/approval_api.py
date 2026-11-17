"""Local HITL approval API, companion to HIL_Approval_NextNode.py.

Run with: python -m langgraph_openai.langgraph.approval_api
Approve:  POST /conversations/{thread_id}/approve  {"decision": "approved"}
Pending:  GET  /conversations/{thread_id}/pending
Swagger UI: http://127.0.0.1:8001/docs

Responses contain conversation content. This utility has no authentication and
must not be exposed outside a trusted local environment.
"""

import os
from pathlib import Path
from typing import Annotated, Any, Literal

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi import Path as ApiPath
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.types import Command
from pydantic import BaseModel, Field

from langgraph_openai.langgraph.HIL_Approval_NextNode import build_graph

app = FastAPI(
    title="LangGraph HITL approval",
    description="Approve or reject a paused code-change request. Local use only; no authentication.",
)


class ApprovalRequest(BaseModel):
    decision: Literal["approved", "rejected"] = Field(description="Reviewer's decision for the paused code change.")


class ApprovalResponse(BaseModel):
    thread_id: str = Field(description="The conversation thread that was resumed.")
    message: str = Field(description="The agent's response after resuming.")


class PendingResponse(BaseModel):
    thread_id: str = Field(description="The conversation thread that was checked.")
    pending: bool = Field(description="Whether the thread is currently paused on an approval request.")
    interrupts: list[Any] = Field(description="Payloads of any pending interrupts.")


def _require_database_url() -> str:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL must be set to inspect or resume checkpoints.")
    return database_url


@app.post(
    "/conversations/{thread_id}/approve",
    response_model=ApprovalResponse,
    summary="Resume a paused code-change approval",
    tags=["Approvals"],
)
def approve(
    thread_id: Annotated[str, ApiPath(description="Conversation thread ID that is paused on an approval request.")],
    body: ApprovalRequest,
) -> dict[str, Any]:
    database_url = _require_database_url()
    config = {"configurable": {"thread_id": thread_id}}

    # Same node/edge wiring as the CLI agent; only the resume trigger differs.
    llm = init_chat_model(model=os.getenv("OPENAI_MODEL"))
    with PostgresSaver.from_conn_string(database_url) as checkpointer:
        graph_instance = build_graph(llm).compile(checkpointer=checkpointer)
        result = graph_instance.invoke(Command(resume=body.decision), config=config)

    return {"thread_id": thread_id, "message": result["messages"][-1].content}


@app.get(
    "/conversations/{thread_id}/pending",
    response_model=PendingResponse,
    summary="Check whether a thread is paused on an approval request",
    tags=["Approvals"],
)
def pending(
    thread_id: Annotated[str, ApiPath(description="Conversation thread ID to check.")],
) -> dict[str, Any]:
    database_url = _require_database_url()
    config = {"configurable": {"thread_id": thread_id}}

    llm = init_chat_model(model=os.getenv("OPENAI_MODEL"))
    with PostgresSaver.from_conn_string(database_url) as checkpointer:
        graph_instance = build_graph(llm).compile(checkpointer=checkpointer)
        snapshot = graph_instance.get_state(config)

    interrupt_values = [item.value for item in snapshot.interrupts]
    return {"thread_id": thread_id, "pending": bool(interrupt_values), "interrupts": interrupt_values}


if __name__ == "__main__":
    # Bind locally because approval payloads may contain sensitive conversation content.
    # Runs on 8001 so it can coexist with checkpoint_api.py's 8000.
    uvicorn.run(app, host="127.0.0.1", port=8001)
