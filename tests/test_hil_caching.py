"""Offline tests: no OpenAI requests, embedding downloads or shared database writes."""

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import uuid

from fastapi.testclient import TestClient
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres import PostgresSaver

from langgraph_openai.langgraph.HIL import HIL_MultiNode_Caching_Persistance as workflow


class FakeRetriever:
    identity = "test-collection:revision-1"

    def __init__(self):
        self.calls = 0
        self.documents = ["LangGraph checkpoints save conversation state."]

    def retrieve(self, question):
        self.calls += 1
        return list(self.documents)


def fake_model(intent="chat"):
    model = MagicMock(spec=BaseChatModel)
    model.with_structured_output.return_value.invoke.return_value = workflow.RequestIntentClassifier(
        human_prompt=intent,
    )
    model.invoke.return_value = AIMessage(content="Test answer")
    return model


def make_service(intent="chat", saver=None, cache=None, model=None):
    return workflow.AgentService(
        model if model is not None else fake_model(intent),
        FakeRetriever(), cache if cache is not None else workflow.L1Cache(),
        "test-model:temperature=0", saver if saver is not None else InMemorySaver(),
    )


class L1CacheTests(unittest.TestCase):
    def test_exact_key_thread_and_layer_isolation_and_copying(self):
        cache = workflow.L1Cache()
        compute = MagicMock(return_value={"items": [1]})
        first = cache.remember("a", "node_output", {"x": 1, "y": 2}, compute)
        first["items"].append(2)
        second = cache.remember("a", "node_output", {"y": 2, "x": 1}, compute)
        self.assertEqual(second, {"items": [1]})
        self.assertEqual(compute.call_count, 1)
        cache.remember("b", "node_output", {"x": 1, "y": 2}, compute)
        cache.remember("a", "llm_response", {"x": 1, "y": 2}, compute)
        self.assertEqual(compute.call_count, 3)
        cache.clear("a")
        self.assertEqual(cache.stats("a")["process_entries"], 1)
        self.assertEqual(sum(cache.stats("a")["conversation_entries"].values()), 0)

    def test_ttl_boundary_and_lru_limit(self):
        clock = MagicMock(return_value=0)
        cache = workflow.L1Cache(ttl=300, max_entries=2, clock=clock)
        compute = MagicMock(return_value="answer")
        for key in ("a", "b", "a", "c"):
            cache.remember("thread", "human_question", key, compute)
        self.assertEqual(compute.call_count, 3)
        self.assertEqual(cache.stats("thread")["process_entries"], 2)
        cache.remember("thread", "human_question", "b", compute)
        self.assertEqual(compute.call_count, 4)
        clock.return_value = 299.999
        cache.remember("thread", "human_question", "b", compute)
        self.assertEqual(compute.call_count, 4)
        clock.return_value = 300
        cache.remember("thread", "human_question", "b", compute)
        self.assertEqual(compute.call_count, 5)

    def test_failures_are_not_cached(self):
        cache = workflow.L1Cache()
        compute = MagicMock(side_effect=[RuntimeError("unavailable"), "recovered"])
        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            cache.remember("a", "llm_response", "prompt", compute)
        self.assertEqual(cache.remember("a", "llm_response", "prompt", compute), "recovered")
        self.assertEqual(compute.call_count, 2)

    def test_invalid_configuration_is_explicit(self):
        for kwargs in ({"ttl": 0}, {"max_entries": 0}):
            with self.assertRaises(ValueError):
                workflow.L1Cache(**kwargs)
        with self.assertRaises(ValueError):
            workflow.L1Cache().remember("", "human_question", "x", lambda: "x")


class CachedNodeTests(unittest.TestCase):
    def setUp(self):
        self.model = fake_model()
        self.retriever = FakeRetriever()
        self.cache = workflow.L1Cache()
        self.nodes = workflow.CachedNodes(self.model, self.retriever, self.cache, "model-a")
        self.config = {"configurable": {"thread_id": "a"}}
        self.state = {
            "messages": [HumanMessage(content="hello")],
            "human_prompt": "chat", "documents": [], "approval_decision": None,
        }

    def test_question_and_node_caches_reuse_exact_inputs(self):
        self.nodes.classify(self.state, self.config)
        self.nodes.classify(self.state, self.config)
        first = self.nodes.respond(self.state, self.config, "chat")
        second = self.nodes.respond(self.state, self.config, "chat")
        self.assertEqual(first, second)
        self.model.with_structured_output.return_value.invoke.assert_called_once()
        self.model.invoke.assert_called_once()
        stats = self.cache.stats("a")
        self.assertEqual(stats["process_hits"]["human_question"], 1)
        self.assertEqual(stats["process_hits"]["node_input"], 1)
        self.assertEqual(stats["process_hits"]["node_output"], 1)
        self.assertGreater(stats["conversation_entries"]["llm_response"], 0)

    def test_llm_response_cache_works_when_node_output_is_evicted(self):
        self.nodes.respond(self.state, self.config, "chat")
        # Simulate eviction of the outer node-result entry, leaving LLM entries.
        with self.cache._lock:
            for key in list(self.cache._entries):
                if key[1] == "node_output":
                    del self.cache._entries[key]
        self.nodes.respond(self.state, self.config, "chat")
        self.model.invoke.assert_called_once()
        self.assertEqual(self.cache.stats("a")["process_hits"]["llm_response"], 1)

    def test_history_model_and_thread_changes_miss(self):
        self.nodes.respond(self.state, self.config, "chat")
        self.state["messages"].insert(0, HumanMessage(content="My name is Alice"))
        self.nodes.respond(self.state, self.config, "chat")
        self.nodes.respond(self.state, {"configurable": {"thread_id": "b"}}, "chat")
        other_model = workflow.CachedNodes(self.model, self.retriever, self.cache, "model-b")
        other_model.respond(self.state, self.config, "chat")
        self.assertEqual(self.model.invoke.call_count, 4)

    def test_retrieval_hit_expiry_revision_and_context_changes(self):
        clock = MagicMock(return_value=0)
        self.cache.clock = clock
        self.nodes.retrieve(self.state, self.config)
        self.nodes.retrieve(self.state, self.config)
        self.assertEqual(self.retriever.calls, 1)
        self.retriever.identity = "test-collection:revision-2"
        self.nodes.retrieve(self.state, self.config)
        self.assertEqual(self.retriever.calls, 2)
        clock.return_value = 300
        self.nodes.retrieve(self.state, self.config)
        self.assertEqual(self.retriever.calls, 3)
        self.state["documents"] = ["old document"]
        self.nodes.respond(self.state, self.config, "knowledge")
        self.state["documents"] = ["new document"]
        self.nodes.respond(self.state, self.config, "knowledge")
        self.assertEqual(self.model.invoke.call_count, 2)
        self.assertIn("new document", self.model.invoke.call_args.args[0][1].content)


class GraphTests(unittest.TestCase):
    def test_all_branches_and_fresh_approval_for_repeated_question(self):
        for intent in ("chat", "knowledge", "code"):
            with self.subTest(intent=intent):
                model = fake_model(intent)
                service = make_service(model=model)
                result = service.run("a", question="same question")
                if intent == "code":
                    self.assertTrue(result["pending"])
                    self.assertIsNone(result["message"])
                    model.invoke.assert_not_called()
                    with self.assertRaises(workflow.ConversationError):
                        service.run("a", question="bypass approval")
                    result = service.run("a", decision="approved")
                self.assertEqual(result["message"], "Test answer")
                self.assertFalse(result["pending"])
                again = service.run("a", question="same question")
                if intent == "code":
                    self.assertTrue(again["pending"])
                    result = service.run("a", decision="rejected")
                    self.assertIn("not approved", result["message"])
                    self.assertEqual(model.invoke.call_count, 1)
                else:
                    self.assertFalse(again["pending"])
                    # The history changed, so the final answer must be recomputed.
                    self.assertEqual(model.invoke.call_count, 2)
                model.with_structured_output.return_value.invoke.assert_called_once()
                messages = service.graph.get_state(service.config("a")).values["messages"]
                self.assertEqual(len(messages), 4)
                self.assertEqual(len({message.id for message in messages}), 4)

    def test_rebuilt_service_resumes_approval_without_classifier_replay(self):
        saver = InMemorySaver()
        model = fake_model("code")
        make_service(model=model, saver=saver).run("a", question="refactor function")
        rebuilt = make_service(model=model, saver=saver)
        self.assertEqual(rebuilt.cache.stats("a")["process_entries"], 0)
        self.assertTrue(rebuilt.pending("a")["pending"])
        self.assertEqual(rebuilt.run("a", decision="approved")["message"], "Test answer")
        model.with_structured_output.return_value.invoke.assert_called_once()

    def test_failed_response_can_resume_and_error_is_not_cached(self):
        model = fake_model()
        model.invoke.side_effect = [RuntimeError("model unavailable"), AIMessage(content="recovered")]
        saver = InMemorySaver()
        service = make_service(model=model, saver=saver)
        with self.assertRaisesRegex(RuntimeError, "model unavailable"):
            service.run("a", question="hello")
        rebuilt = make_service(model=model, saver=saver)
        self.assertEqual(rebuilt.run("a", resume=True)["message"], "recovered")
        model.with_structured_output.return_value.invoke.assert_called_once()

    def test_approval_is_fail_closed(self):
        with patch.object(workflow, "interrupt", return_value="APPROVED"):
            self.assertEqual(workflow.request_approval({"messages": [HumanMessage(content="code")]}),
                             {"approval_decision": "rejected"})


class ChromaTests(unittest.TestCase):
    def test_empty_collection_is_explicit_and_never_downloads_embeddings(self):
        collection = MagicMock()
        collection.count.return_value = 0
        with patch("chromadb.PersistentClient") as factory:
            factory.return_value.get_or_create_collection.return_value = collection
            retriever = workflow.ChromaRetriever("/tmp/hil-cache-test", "HIL_Caching_node_persistance")
            with self.assertLogs(workflow.logger, level="WARNING"):
                self.assertEqual(retriever.retrieve("question"), [])
            collection.query.assert_not_called()
            collection.count.return_value = 1
            collection.query.return_value = {"documents": [["document"]]}
            self.assertEqual(retriever.retrieve("question"), ["document"])
            collection.query.assert_called_once_with(
                query_texts=["question"], n_results=1, include=["documents"],
            )
            collection.query.side_effect = RuntimeError("database unavailable")
            with self.assertRaisesRegex(RuntimeError, "database unavailable"):
                retriever.retrieve("question")

    def test_real_persistent_collection_creation_and_reopen(self):
        with tempfile.TemporaryDirectory() as directory:
            first = workflow.ChromaRetriever(directory, "HIL_Caching_node_persistance")
            self.assertEqual(first.collection.count(), 0)
            # Explicit vectors keep this test offline; no default embedding download.
            first.collection.add(ids=["test"], documents=["test document"], embeddings=[[1.0, 0.0]])
            second = workflow.ChromaRetriever(directory, "HIL_Caching_node_persistance")
            self.assertEqual(second.collection.count(), 1)


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.model = fake_model("code")
        self.service = make_service(model=self.model)
        self.client = TestClient(workflow.create_app(self.service), raise_server_exceptions=False)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)

    def test_question_pending_approval_and_cache_routes(self):
        base = "/conversations/a"
        response = self.client.post(f"{base}/messages", json={"question": "refactor function"})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["pending"])
        self.assertIsNone(response.json()["message"])
        self.assertTrue(self.client.get(f"{base}/pending").json()["pending"])
        self.assertEqual(self.client.post(f"{base}/messages", json={"question": "new"}).status_code, 409)
        self.assertEqual(self.client.post(f"{base}/resume").status_code, 409)
        result = self.client.post(f"{base}/approve", json={"decision": "approved"})
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json()["message"], "Test answer")
        self.assertFalse(self.client.get(f"{base}/pending").json()["pending"])
        stats = self.client.get(f"{base}/cache").json()
        self.assertGreater(stats["process_entries"], 0)
        self.assertEqual(set(stats["conversation_entries"]), set(workflow.LAYERS))
        self.assertNotIn("refactor function", str(stats))
        self.assertGreater(self.client.delete(f"{base}/cache").json()["cleared_entries"], 0)
        self.assertEqual(self.client.get(f"{base}/cache").json()["process_entries"], 0)
        self.assertFalse(self.service.pending("a")["pending"])

    def test_validation_missing_thread_and_no_pending_approval(self):
        self.assertEqual(self.client.get("/conversations/missing/pending").status_code, 404)
        for question in ("", "   ", "x" * 16001):
            self.assertEqual(self.client.post("/conversations/a/messages", json={
                "question": question,
            }).status_code, 422)
        self.assertEqual(self.client.post("/conversations/a/approve", json={
            "decision": "APPROVED",
        }).status_code, 422)
        self.assertEqual(self.client.post("/conversations/a/approve", json={
            "decision": "approved",
        }).status_code, 409)

    def test_retrieval_and_resume_errors_are_not_success_responses(self):
        self.model.with_structured_output.return_value.invoke.return_value = workflow.RequestIntentClassifier(
            human_prompt="knowledge",
        )
        self.model.invoke.side_effect = RuntimeError("model unavailable")
        self.assertEqual(self.client.post("/conversations/a/messages", json={"question": "what?"}).status_code, 500)
        self.model.invoke.side_effect = None
        self.assertEqual(self.client.post("/conversations/a/resume").status_code, 200)

    def test_swagger_lists_new_routes(self):
        schema = self.client.get("/openapi.json").json()
        self.assertEqual(self.client.get("/docs").status_code, 200)
        for route in ("messages", "approve", "resume", "pending", "cache"):
            self.assertIn(f"/conversations/{{thread_id}}/{route}", schema["paths"])

    def test_api_lifespan_opens_service_only_once(self):
        context = MagicMock()
        context.__enter__.return_value = self.service
        with patch.object(workflow, "open_service", return_value=context) as factory:
            with TestClient(workflow.create_app()) as client:
                for _ in range(2):
                    self.assertEqual(client.get("/conversations/a/cache").status_code, 200)
            factory.assert_called_once()
            context.__exit__.assert_called_once()


class StartupTests(unittest.TestCase):
    def test_configuration_creates_persistent_service(self):
        environment = {
            "DATABASE_URL": "postgresql://test",
            "OPENAI_MODEL": "test-model",
            "CHROMA_PATH": "/tmp/hil-example",
            "CHROMA_COLLECTION": "HIL_Caching_node_persistance",
            "CHROMA_REVISION": "2",
        }
        with patch.object(workflow, "load_dotenv"), patch.dict(
            os.environ, environment, clear=True,
        ), patch.object(workflow, "ChromaRetriever") as retriever, patch.object(
            workflow, "init_chat_model", return_value=fake_model(),
        ) as model_factory, patch.object(workflow.PostgresSaver, "from_conn_string") as saver_factory:
            saver = InMemorySaver()
            saver.setup = MagicMock()
            saver_factory.return_value.__enter__.return_value = saver
            with workflow.open_service() as service:
                self.assertEqual(service.run("a", question="hello")["message"], "Test answer")
            model_factory.assert_called_once_with(model="test-model", temperature=0)
            retriever.assert_called_once_with("/tmp/hil-example", "HIL_Caching_node_persistance", "2")
            saver.setup.assert_called_once()
            saver_factory.assert_called_once_with("postgresql://test")

    def test_missing_configuration_fails_explicitly(self):
        with patch.object(workflow, "load_dotenv"), patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "DATABASE_URL and OPENAI_MODEL"):
                with workflow.open_service():
                    self.fail("Must not start without configuration")

    def test_cli_approval_and_serve_modes(self):
        service = make_service("code")
        service.run("a", question="refactor")
        context = MagicMock()
        context.__enter__.return_value = service
        module = "langgraph_openai.langgraph.HIL.HIL_MultiNode_Caching_Persistance"
        with patch.object(workflow, "load_dotenv"), patch.object(
            workflow, "open_service", return_value=context,
        ), patch("sys.argv", [module, "--thread-id", "a", "--approve", "approved"]), patch(
            "builtins.print",
        ) as output:
            workflow.main()
            self.assertIn("Test answer", output.call_args.args[0])
        with patch.object(workflow, "load_dotenv"), patch(
            "sys.argv", [module, "--serve"],
        ), patch("uvicorn.run") as run:
            workflow.main()
            run.assert_called_once_with(workflow.app, host="127.0.0.1", port=8002)


@unittest.skipUnless(os.getenv("TEST_DATABASE_URL"), "Set TEST_DATABASE_URL to a dedicated PostgreSQL database")
class PostgresIntegrationTests(unittest.TestCase):
    def test_approval_survives_reconnect_with_cold_cache(self):
        thread = f"hil-cache-test-{uuid.uuid4()}"
        model = fake_model("code")
        url = os.environ["TEST_DATABASE_URL"]

        def cleanup():
            with PostgresSaver.from_conn_string(url) as saver:
                saver.delete_thread(thread)

        with PostgresSaver.from_conn_string(url) as saver:
            saver.setup()
            self.addCleanup(cleanup)
            first = make_service(model=model, saver=saver)
            self.assertTrue(first.run(thread, question="refactor")["pending"])
        with PostgresSaver.from_conn_string(url) as saver:
            second = make_service(model=model, saver=saver)
            self.assertTrue(second.pending(thread)["pending"])
            self.assertEqual(second.cache.stats(thread)["process_entries"], 0)
            self.assertEqual(second.run(thread, decision="approved")["message"], "Test answer")
        model.with_structured_output.return_value.invoke.assert_called_once()


if __name__ == "__main__":
    unittest.main()
