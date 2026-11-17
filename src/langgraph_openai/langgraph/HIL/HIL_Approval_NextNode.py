# Usage notes (run commands from the repository root):
# 1. Start the agent:
#    PYTHONPATH=src .venv/bin/python -m langgraph_openai.langgraph.HIL_Approval_NextNode --thread-id hitl-test-001
# 2. Send a message that classifies as a code request (e.g. "refactor this function").
#    The graph pauses before code_agent runs and prints the interrupt payload.
# 3. Approve or reject from another terminal, reusing the same thread ID:
#    PYTHONPATH=src .venv/bin/python -m langgraph_openai.langgraph.HIL_Approval_NextNode --thread-id hitl-test-001 --approve approved
#    PYTHONPATH=src .venv/bin/python -m langgraph_openai.langgraph.HIL_Approval_NextNode --thread-id hitl-test-001 --approve rejected
# 4. Or approve via approval_api.py's POST /conversations/{thread_id}/approve (see that file).
# 5. Set DATABASE_URL, OPENAI_MODEL, and OPENAI_API_KEY in the environment or src/langgraph_openai/.env.
#    PostgreSQL must be running; its database user needs permission to create checkpoint tables.
#    Only the 'code' intent is gated; chat/knowledge intents respond immediately.

import argparse
import os
import sys
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from langgraph_openai.langgraph.langgraph_classifier.RequestIntentClassiffier import (
    State,
    classify_intent,
    prompt_llm_chat,
    prompt_llm_code,
    prompt_llm_knowledge,
)


class HitlState(State):
    approval_decision: str | None


def route_by_intent(state: HitlState) -> str:
    return state['human_prompt']


def request_code_change_approval(state: HitlState) -> dict:
    decision = interrupt({
        "action": "approve_code_change",
        "message": "A code-change request needs approval before the code agent responds.",
        "user_request": state['messages'][-1].content,
    })
    # Fail-closed: only an exact "approved" proceeds; any other value is treated as rejected.
    return {"approval_decision": "approved" if decision == "approved" else "rejected"}


def route_by_approval(state: HitlState) -> str:
    return state['approval_decision']


def code_change_rejected(state: HitlState) -> dict:
    return {"messages": [{"role": "assistant", "content": "Code change was not approved; skipping."}]}


def build_graph(llm):
    graph_builder = StateGraph(HitlState)
    graph_builder.add_node('classifier_agent', lambda state: classify_intent(llm, state))
    graph_builder.add_node('chat_agent', lambda state: prompt_llm_chat(llm, state))
    graph_builder.add_node('knowledge_agent', lambda state: prompt_llm_knowledge(llm, state))
    graph_builder.add_node('request_code_change_approval', request_code_change_approval)
    graph_builder.add_node('code_agent', lambda state: prompt_llm_code(llm, state))
    graph_builder.add_node('code_change_rejected', code_change_rejected)

    graph_builder.add_edge(START, 'classifier_agent')
    graph_builder.add_conditional_edges('classifier_agent', route_by_intent, {
        'chat': 'chat_agent',
        'knowledge': 'knowledge_agent',
        'code': 'request_code_change_approval',
    })
    graph_builder.add_conditional_edges('request_code_change_approval', route_by_approval, {
        'approved': 'code_agent',
        'rejected': 'code_change_rejected',
    })
    graph_builder.add_edge('chat_agent', END)
    graph_builder.add_edge('knowledge_agent', END)
    graph_builder.add_edge('code_agent', END)
    graph_builder.add_edge('code_change_rejected', END)
    return graph_builder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the HITL-gated intent-classifying LangGraph agent.")
    parser.add_argument(
        "--thread-id",
        default=os.getenv("LANGGRAPH_THREAD_ID"),
        help="Stable conversation ID; reuse it to continue, resume, or approve the conversation.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume unfinished work for the selected thread instead of starting with new input.",
    )
    parser.add_argument(
        "--approve",
        choices=["approved", "rejected"],
        help="Resume a paused code-change approval for the selected thread instead of prompting for input.",
    )
    return parser.parse_args()


def print_result(result: dict[str, Any]) -> None:
    if "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        print(f"[PAUSED] Awaiting approval: {payload}", file=sys.stderr)
        print("Resume with: --thread-id <id> --approve approved|rejected", file=sys.stderr)
        return
    print(result['messages'][-1].content)


def main() -> None:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    args = parse_args()
    if args.resume and not args.thread_id:
        raise ValueError("--resume requires --thread-id or LANGGRAPH_THREAD_ID.")
    if args.approve and not args.thread_id:
        raise ValueError("--approve requires --thread-id or LANGGRAPH_THREAD_ID.")

    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL must be set to enable persistent checkpoints.")

    thread_id = args.thread_id or str(uuid.uuid4())
    config = {'configurable': {'thread_id': thread_id}}
    # Keep this ID to resume, approve, or recover the same conversation later.
    print(f"Conversation thread ID: {thread_id}", file=sys.stderr)

    llm = init_chat_model(model=os.getenv("OPENAI_MODEL"))
    graph_builder = build_graph(llm)

    # PostgreSQL checkpoints survive process restarts and let a paused approval resume later.
    with PostgresSaver.from_conn_string(database_url) as checkpointer:
        checkpointer.setup()
        graph_instance = graph_builder.compile(checkpointer=checkpointer)

        if args.approve:
            result = graph_instance.invoke(Command(resume=args.approve), config=config)
            print_result(result)
            return

        if args.resume:
            # Reuse the thread ID and omit new input to continue its unfinished graph run.
            result = graph_instance.invoke(None, config=config)
            print_result(result)

        while True:
            user_input = input("Enter your question: ")
            result = graph_instance.invoke(
                {'messages': [{'role': 'user', 'content': user_input}]},
                config=config,
            )
            print_result(result)


if __name__ == "__main__":
    main()


# run sh launch_HIL.sh
#Enter your question: :: Can you refactor this function to use a list comprehension?
#HIL kicks in
#  --thread-id <id> --approve approved|
 