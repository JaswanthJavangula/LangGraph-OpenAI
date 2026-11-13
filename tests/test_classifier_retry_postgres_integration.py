"""PostgreSQL integration test; set TEST_DATABASE_URL to a dedicated UTF-8 test database.

Run: PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -p test_classifier_retry_postgres_integration.py -v
The test creates checkpoint tables if needed and removes only its own test thread.
"""

import os
import unittest
import uuid
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage
from langgraph.checkpoint.postgres import PostgresSaver
from openai import APIConnectionError

from langgraph_openai.langgraph.langgraph_classifier import classifier_multinode_persistance_retry as workflow
from langgraph_openai.langgraph.langgraph_classifier.RequestIntentClassiffier import RequestIntentClassifier


@unittest.skipUnless(os.getenv("TEST_DATABASE_URL"), "Set TEST_DATABASE_URL to a dedicated PostgreSQL test database")
class PostgresRetryIntegrationTests(unittest.TestCase):
    def test_failed_response_node_resumes_from_postgres_after_reconnect(self):
        database_url = os.environ["TEST_DATABASE_URL"]
        thread_id = f"integration-test-{uuid.uuid4()}"
        config = {"configurable": {"thread_id": thread_id}}
        prompt = "Remember this integration test request"

        def cleanup_thread():
            with PostgresSaver.from_conn_string(database_url) as saver:
                saver.delete_thread(thread_id)

        model = MagicMock()
        classifier = model.with_structured_output.return_value
        classifier.invoke.return_value = RequestIntentClassifier(human_prompt="chat")
        model.invoke.side_effect = [
            APIConnectionError(request=MagicMock()),
            APIConnectionError(request=MagicMock()),
            APIConnectionError(request=MagicMock()),
        ]

        with patch("langgraph.pregel._retry.time.sleep") as sleep:
            with PostgresSaver.from_conn_string(database_url) as saver:
                saver.setup()
                # A unique thread limits cleanup to data created by this test.
                self.addCleanup(cleanup_thread)
                graph = workflow.build_graph(model).compile(checkpointer=saver)
                with self.assertLogs(workflow.logger, level="ERROR") as logged:
                    with self.assertRaises(APIConnectionError):
                        workflow.invoke_with_failure_logging(
                            graph, {"messages": [{"role": "user", "content": prompt}]}, config
                        )
                self.assertEqual(model.invoke.call_count, 3)
                self.assertEqual(sleep.call_count, 2)
                self.assertEqual(classifier.invoke.call_count, 1)
                self.assertIn(f"thread_id={thread_id}", logged.output[0])

        # Reopen the saver and rebuild the graph to simulate a process restart.
        model.invoke.side_effect = None
        model.invoke.return_value = AIMessage(content="Recovered")
        with PostgresSaver.from_conn_string(database_url) as saver:
            graph = workflow.build_graph(model).compile(checkpointer=saver)
            snapshot = graph.get_state(config)
            self.assertEqual(snapshot.next, ("chat_agent",))
            self.assertEqual(snapshot.values["messages"][-1].content, prompt)
            result = workflow.invoke_with_failure_logging(graph, None, config)
            self.assertEqual([message.content for message in result["messages"]], [prompt, "Recovered"])
            self.assertEqual(classifier.invoke.call_count, 1)
            self.assertEqual(model.invoke.call_count, 4)
            self.assertFalse(graph.get_state(config).next)

        with PostgresSaver.from_conn_string(database_url) as saver:
            graph = workflow.build_graph(model).compile(checkpointer=saver)
            snapshot = graph.get_state(config)
            self.assertFalse(snapshot.next)
            self.assertEqual([message.content for message in snapshot.values["messages"]], [prompt, "Recovered"])


if __name__ == "__main__":
    unittest.main()
