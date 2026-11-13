# Usage notes (run commands from the repository root):
# 1. Start Swagger: PYTHONPATH=src .venv/bin/python -m langgraph_openai.langgraph.langgraph_classifier.checkpoint_api
# 2. In another terminal, start the agent:
#    PYTHONPATH=src .venv/bin/python -m langgraph_openai.langgraph.langgraph_classifier.classifier_multinode_persistance_retry --thread-id persistence-test-001
# 3. Open http://127.0.0.1:8000/docs and use GET /checkpoints/{thread_id}.
# 4. Set DATABASE_URL, OPENAI_MODEL, and OPENAI_API_KEY in the environment or src/langgraph_openai/.env.
#    PostgreSQL must be running; its database user needs permission to create checkpoint tables.
#    Reuse the thread ID across restarts; add --resume to continue an unfinished run.
#    Each model node retries transient OpenAI failures up to three total attempts with a 30-second request timeout.
#    The Swagger endpoint exposes conversation content without authentication; keep it local.
#    Press Ctrl+C in each terminal to stop the agent and server.


import argparse
import logging
import os
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.graph import END, START, StateGraph
from langgraph_openai.langgraph.langgraph_classifier.RequestIntentClassiffier import (
    State,
    classify_intent,
    prompt_llm_chat,
    prompt_llm_knowledge,
    prompt_llm_code,
)
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.types import RetryPolicy
from openai import APIConnectionError, APIStatusError


logger = logging.getLogger(__name__)


def is_transient_openai_error(exc: Exception) -> bool:
    if isinstance(exc, APIConnectionError):
        return True
    if isinstance(exc, APIStatusError):
        if exc.status_code == 429:
            return exc.code != "insufficient_quota"
        return 500 <= exc.status_code < 600
    return False


retry_policy = RetryPolicy(
    max_attempts=3,
    initial_interval=1.0,
    backoff_factor=2.0,
    max_interval=8.0,
    jitter=True,
    retry_on=is_transient_openai_error,
)


def route_by_intent(state: State) -> str:
    return state['human_prompt']


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the intent-classifying LangGraph agent.")
    parser.add_argument(
        "--thread-id",
        default=os.getenv("LANGGRAPH_THREAD_ID"),
        help="Stable conversation ID; reuse it to continue or recover the conversation.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume unfinished work for the selected thread instead of starting with new input.",
    )
    return parser.parse_args()


def build_graph(llm):
    graph_builder = StateGraph(State)
    graph_builder.add_node('classifier_agent', lambda state: classify_intent(llm, state), retry_policy=retry_policy)
    graph_builder.add_node('chat_agent', lambda state: prompt_llm_chat(llm, state), retry_policy=retry_policy)
    graph_builder.add_node('knowledge_agent', lambda state: prompt_llm_knowledge(llm, state), retry_policy=retry_policy)
    graph_builder.add_node('code_agent', lambda state: prompt_llm_code(llm, state), retry_policy=retry_policy)

    graph_builder.add_edge(START, 'classifier_agent')
    graph_builder.add_conditional_edges('classifier_agent', route_by_intent, {
        'chat': 'chat_agent',
        'knowledge': 'knowledge_agent',
        'code': 'code_agent',
    })
    graph_builder.add_edge('chat_agent', END)
    graph_builder.add_edge('knowledge_agent', END)
    graph_builder.add_edge('code_agent', END)
    return graph_builder


def invoke_with_failure_logging(graph, graph_input, config):
    try:
        return graph.invoke(graph_input, config=config)
    except Exception:
        logger.exception(
            "Graph invocation failed for thread_id=%s. Resume with --thread-id and --resume if a checkpoint exists.",
            config['configurable']['thread_id'],
        )
        raise


def main() -> None:
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    args = parse_args()
    if args.resume and not args.thread_id:
        raise ValueError("--resume requires --thread-id or LANGGRAPH_THREAD_ID.")

    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL must be set to enable persistent checkpoints.")

    thread_id = args.thread_id or str(uuid.uuid4())
    config = {'configurable': {'thread_id': thread_id}}
    # Keep this ID to resume the same conversation after restarting the program.
    print(f"Conversation thread ID: {thread_id}", file=sys.stderr)

    # LangGraph owns retries so client retries do not multiply the attempt budget.
    llm = init_chat_model(model=os.getenv("OPENAI_MODEL"), max_retries=0, timeout=30)
    graph_builder = build_graph(llm)

    # PostgreSQL checkpoints survive process restarts and allow failed runs to resume.
    with PostgresSaver.from_conn_string(database_url) as checkpointer:
        checkpointer.setup()
        graph_instance = graph_builder.compile(checkpointer=checkpointer)

        if args.resume:
            # Reuse the thread ID and omit new input to continue its unfinished graph run.
            result = invoke_with_failure_logging(graph_instance, None, config)
            print(result['messages'][-1].content)

        while True:
            user_input = input("Enter your question: ")
            result = invoke_with_failure_logging(
                graph_instance,
                {'messages': [{'role': 'user', 'content': user_input}]},
                config,
            )
            print(result['messages'][-1].content)


if __name__ == "__main__":
    main()
