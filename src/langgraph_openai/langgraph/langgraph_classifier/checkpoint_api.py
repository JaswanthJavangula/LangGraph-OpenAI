"""Local checkpoint inspection API.

Run with: python -m langgraph_openai.langgraph.langgraph_classifier.checkpoint_api
Inspect: GET /checkpoints/{thread_id}?limit=20&checkpoint_ns=
Swagger UI: http://127.0.0.1:8000/docs

Responses contain conversation content. This utility has no authentication and
must not be exposed outside a trusted local environment.
"""

import os
from pathlib import Path
from typing import Annotated, Any

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Path as ApiPath, Query
from fastapi.encoders import jsonable_encoder
from langgraph.checkpoint.postgres import PostgresSaver
from pydantic import BaseModel, Field

app = FastAPI(
    title="LangGraph checkpoint inspection",
    description="Inspect persisted checkpoints, including conversation content. Local use only; no authentication.",
)


class CheckpointRecord(BaseModel):
    config: dict[str, Any] = Field(description="Configuration identifying this checkpoint.")
    checkpoint: dict[str, Any] = Field(description="Decoded checkpoint, including graph state.")
    metadata: dict[str, Any] = Field(description="Checkpoint metadata.")
    parent_config: dict[str, Any] | None = Field(description="Configuration of the parent checkpoint, if any.")
    pending_writes: list[list[Any]] | None = Field(description="Pending writes, if any.")


class CheckpointResponse(BaseModel):
    thread_id: str = Field(description="The requested conversation thread ID.")
    checkpoint_ns: str = Field(description="The requested checkpoint namespace.")
    checkpoints: list[CheckpointRecord] = Field(description="Checkpoints ordered newest first.")


@app.get(
    "/checkpoints/{thread_id}",
    response_model=CheckpointResponse,
    summary="List persisted checkpoints for a thread",
    tags=["Checkpoints"],
)
def list_checkpoints(
    thread_id: Annotated[str, ApiPath(description="Conversation thread ID used when running the graph.")],
    limit: Annotated[int, Query(ge=1, le=100, description="Maximum number of checkpoints to return.")] = 20,
    checkpoint_ns: Annotated[str, Query(description="Checkpoint namespace; empty for the root graph.")] = "",
) -> dict[str, Any]:
    """Return up to 100 checkpoints, newest first. An unknown thread returns an empty list."""
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL must be set to inspect checkpoints.")

    # Use the saver to decode state and pending writes instead of reading internal tables.
    with PostgresSaver.from_conn_string(database_url) as checkpointer:
        checkpoints = list(
            checkpointer.list(
                {"configurable": {"thread_id": thread_id, "checkpoint_ns": checkpoint_ns}},
                limit=limit,
            )
        )

    return {
        "thread_id": thread_id,
        "checkpoint_ns": checkpoint_ns,
        "checkpoints": [
            {
                "config": jsonable_encoder(item.config),
                "checkpoint": jsonable_encoder(item.checkpoint),
                "metadata": jsonable_encoder(item.metadata),
                "parent_config": jsonable_encoder(item.parent_config),
                "pending_writes": jsonable_encoder(item.pending_writes),
            }
            for item in checkpoints
        ],
    }


if __name__ == "__main__":
    # Bind locally because checkpoint contents may contain sensitive conversations.
    uvicorn.run(app, host="127.0.0.1", port=8000)
