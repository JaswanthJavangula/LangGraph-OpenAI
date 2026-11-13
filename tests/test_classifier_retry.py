import argparse
import os
import unittest
from unittest.mock import MagicMock, patch

from langgraph.checkpoint.memory import InMemorySaver
from openai import APIConnectionError, APIStatusError, APITimeoutError

from langgraph_openai.langgraph.langgraph_classifier import classifier_multinode_persistance_retry as workflow


def status_error(status: int, code: str | None = None) -> APIStatusError:
    response = MagicMock()
    response.status_code = status
    return APIStatusError("test error", response=response, body={"code": code})


def connection_error() -> APIConnectionError:
    return APIConnectionError(request=MagicMock())


class RetryPolicyTests(unittest.TestCase):
    def test_only_transient_errors_are_retryable(self):
        cases = [
            (connection_error(), True),
            (APITimeoutError(request=MagicMock()), True),
            (status_error(429), True),
            (status_error(429, "insufficient_quota"), False),
            (status_error(500), True),
            (status_error(503), True),
            (status_error(401), False),
            (status_error(400), False),
            (ValueError("invalid output"), False),
        ]
        for error, expected in cases:
            with self.subTest(error=type(error).__name__, status=getattr(error, "status_code", None)):
                self.assertEqual(workflow.is_transient_openai_error(error), expected)

    @patch("langgraph.pregel._retry.time.sleep")
    def test_each_node_retries_a_transient_failure(self, sleep):
        for route in ("chat", "knowledge", "code"):
            for failed_node in ("classifier", route):
                with self.subTest(route=route, failed_node=failed_node):
                    classifier = MagicMock(return_value={"human_prompt": route})
                    agent = MagicMock(return_value={"messages": [{"role": "assistant", "content": "ok"}]})
                    failed = classifier if failed_node == "classifier" else agent
                    result = {"human_prompt": route} if failed_node == "classifier" else {
                        "messages": [{"role": "assistant", "content": "ok"}]
                    }
                    failed.side_effect = [connection_error(), result]
                    with patch.object(workflow, "classify_intent", classifier), patch.object(
                        workflow, f"prompt_llm_{route}", agent
                    ):
                        graph = workflow.build_graph(None).compile()
                        output = graph.invoke({"messages": [{"role": "user", "content": "hello"}]})
                    self.assertEqual(failed.call_count, 2)
                    self.assertEqual(output["messages"][-1].content, "ok")
        self.assertEqual(sleep.call_count, 6)

    @patch("langgraph.pregel._retry.time.sleep")
    def test_exhausted_run_is_logged_and_can_resume(self, sleep):
        classifier = MagicMock(side_effect=[connection_error(), connection_error(), connection_error()])
        agent = MagicMock(return_value={"messages": [{"role": "assistant", "content": "recovered"}]})
        config = {"configurable": {"thread_id": "retry-thread"}}
        with patch.object(workflow, "classify_intent", classifier), patch.object(
            workflow, "prompt_llm_chat", agent
        ):
            graph = workflow.build_graph(None).compile(checkpointer=InMemorySaver())
            with self.assertLogs(workflow.logger, level="ERROR") as logged:
                with self.assertRaises(APIConnectionError):
                    workflow.invoke_with_failure_logging(
                        graph, {"messages": [{"role": "user", "content": "hello"}]}, config
                    )
            self.assertEqual(classifier.call_count, 3)
            self.assertEqual(sleep.call_count, 2)
            self.assertIn("thread_id=retry-thread", logged.output[0])

            classifier.side_effect = None
            classifier.return_value = {"human_prompt": "chat"}
            result = workflow.invoke_with_failure_logging(graph, None, config)
            self.assertEqual(result["messages"][-1].content, "recovered")

    @patch("langgraph.pregel._retry.time.sleep")
    def test_authentication_error_is_not_retried(self, sleep):
        classifier = MagicMock(side_effect=status_error(401))
        with patch.object(workflow, "classify_intent", classifier):
            graph = workflow.build_graph(None).compile()
            with self.assertRaises(APIStatusError):
                graph.invoke({"messages": [{"role": "user", "content": "hello"}]})
        self.assertEqual(classifier.call_count, 1)
        sleep.assert_not_called()

    def test_main_disables_sdk_retries_and_sets_timeout(self):
        graph = MagicMock()
        graph.invoke.side_effect = ValueError("failed")
        saver_context = MagicMock()
        with patch.object(workflow, "load_dotenv"), patch.object(
            workflow, "parse_args", return_value=argparse.Namespace(thread_id="thread", resume=False)
        ), patch.dict(os.environ, {"DATABASE_URL": "postgresql://test", "OPENAI_MODEL": "gpt-4o-mini"}), patch.object(
            workflow, "init_chat_model"
        ) as model_factory, patch.object(
            workflow.PostgresSaver, "from_conn_string", return_value=saver_context
        ), patch.object(
            workflow, "build_graph"
        ) as builder, patch("builtins.input", return_value="hello"), self.assertLogs(
            workflow.logger, level="ERROR"
        ):
            builder.return_value.compile.return_value = graph
            with self.assertRaisesRegex(ValueError, "failed"):
                workflow.main()
        model_factory.assert_called_once_with(model="gpt-4o-mini", max_retries=0, timeout=30)


if __name__ == "__main__":
    unittest.main()
