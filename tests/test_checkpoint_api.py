import os
import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.base import CheckpointTuple

from langgraph_openai.langgraph.langgraph_classifier.checkpoint_api import app


class CheckpointApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app, raise_server_exceptions=False)
        self.addCleanup(self.client.close)

    @patch("langgraph_openai.langgraph.langgraph_classifier.checkpoint_api.PostgresSaver")
    def test_returns_decoded_checkpoints_for_requested_thread(self, saver_class):
        saver = saver_class.from_conn_string.return_value.__enter__.return_value
        saver.list.return_value = iter([
            CheckpointTuple(
                config={"configurable": {"thread_id": "thread-1", "checkpoint_id": "id-1"}},
                checkpoint={"channel_values": {"messages": [HumanMessage(content="Hello")]}},
                metadata={"step": 1},
                parent_config=None,
                pending_writes=[("task-1", "messages", HumanMessage(content="Pending"))],
            )
        ])
        with patch.dict(os.environ, {"DATABASE_URL": "postgresql://test"}, clear=True):
            response = self.client.get("/checkpoints/thread-1?limit=5&checkpoint_ns=child")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["thread_id"], "thread-1")
        self.assertEqual(data["checkpoint_ns"], "child")
        self.assertEqual(data["checkpoints"][0]["checkpoint"]["channel_values"]["messages"][0]["content"], "Hello")
        self.assertEqual(data["checkpoints"][0]["pending_writes"][0][2]["content"], "Pending")
        saver.list.assert_called_once_with(
            {"configurable": {"thread_id": "thread-1", "checkpoint_ns": "child"}},
            limit=5,
        )

    @patch("langgraph_openai.langgraph.langgraph_classifier.checkpoint_api.PostgresSaver")
    def test_no_checkpoints_returns_empty_list(self, saver_class):
        saver = saver_class.from_conn_string.return_value.__enter__.return_value
        saver.list.return_value = iter([])
        with patch.dict(os.environ, {"DATABASE_URL": "postgresql://test"}, clear=True):
            response = self.client.get("/checkpoints/missing")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["checkpoints"], [])

    def test_invalid_limit_is_rejected(self):
        for limit in (0, 101):
            with self.subTest(limit=limit):
                self.assertEqual(self.client.get(f"/checkpoints/t?limit={limit}").status_code, 422)

    def test_swagger_and_openapi_expose_checkpoint_endpoint(self):
        docs = self.client.get("/docs")
        self.assertEqual(docs.status_code, 200)
        self.assertIn("swagger-ui", docs.text)

        schema = self.client.get("/openapi.json").json()
        operation = schema["paths"]["/checkpoints/{thread_id}"]["get"]
        self.assertEqual(operation["tags"], ["Checkpoints"])
        self.assertEqual(
            operation["responses"]["200"]["content"]["application/json"]["schema"]["$ref"],
            "#/components/schemas/CheckpointResponse",
        )
        self.assertEqual(
            set(schema["components"]["schemas"]["CheckpointResponse"]["properties"]),
            {"thread_id", "checkpoint_ns", "checkpoints"},
        )
        self.assertEqual(
            {parameter["name"] for parameter in operation["parameters"]},
            {"thread_id", "limit", "checkpoint_ns"},
        )

    @patch("langgraph_openai.langgraph.langgraph_classifier.checkpoint_api.load_dotenv")
    def test_missing_database_url_returns_error(self, _load_dotenv: MagicMock):
        with patch.dict(os.environ, {}, clear=True):
            response = self.client.get("/checkpoints/t")

        self.assertEqual(response.status_code, 500)

    @patch("langgraph_openai.langgraph.langgraph_classifier.checkpoint_api.PostgresSaver")
    def test_database_failure_returns_error(self, saver_class):
        saver_class.from_conn_string.side_effect = RuntimeError("database unavailable")
        with patch.dict(os.environ, {"DATABASE_URL": "postgresql://test"}, clear=True):
            response = self.client.get("/checkpoints/t")

        self.assertEqual(response.status_code, 500)


if __name__ == "__main__":
    unittest.main()
