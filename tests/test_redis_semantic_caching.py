"""Offline Redis/Lua tests; no OpenAI calls or shared Redis writes."""

import time
import unittest
from unittest.mock import MagicMock, patch

import fakeredis
from redis.exceptions import ConnectionError

from langgraph_openai.langgraph.semantic_caching import (
    SemanticRedisCaching_TextEmbeddingCaching as workflow,
)


class RedisSemanticCacheTests(unittest.TestCase):
    def setUp(self):
        self.server = fakeredis.FakeServer()
        self.redis = fakeredis.FakeRedis(server=self.server, decode_responses=True)
        self.addCleanup(self.redis.close)
        self.cache = workflow.RedisSemanticCache(self.redis, "chat", "embedding")

    def test_shared_cache_returns_best_paired_response_and_strict_threshold(self):
        self.cache.store([1.0, 0.0], "France")
        self.cache.store([0.0, 1.0], "India")
        other_client = fakeredis.FakeRedis(server=self.server, decode_responses=True)
        self.addCleanup(other_client.close)
        other_cache = workflow.RedisSemanticCache(other_client, "chat", "embedding")
        self.assertEqual(other_cache.lookup([0.0, 1.0]), ("India", 1.0))
        self.assertEqual(other_cache.lookup([0.9, 0.0]), (None, 0.9))
        self.assertEqual(other_cache.lookup([0.91, 0.0]), ("France", 0.91))
        for model, embedding, namespace in [
            ("other", "embedding", "semantic-demo"),
            ("chat", "other", "semantic-demo"),
            ("chat", "embedding", "other-tenant"),
        ]:
            isolated = workflow.RedisSemanticCache(self.redis, model, embedding, namespace)
            self.assertEqual(isolated.lookup([1.0, 0.0]), (None, 0.0))

    def test_capacity_is_ten_pairs_and_oldest_is_evicted(self):
        origin = time.time()
        for index in range(11):
            with patch("time.time", return_value=origin + index):
                self.cache.store([float(index), 0.0], str(index))
        entries = self.cache._read(keys=[self.cache.key])
        self.assertEqual(len(entries), 10)
        self.assertEqual(self.cache.lookup([1.0, 0.0]), ("10", 10.0))
        import json

        self.assertEqual(
            [json.loads(entry)["response"] for entry in entries],
            [str(index) for index in range(1, 11)],
        )

    def test_each_pair_expires_at_five_minutes_and_hits_do_not_extend_ttl(self):
        origin = int(time.time())
        with patch("time.time", return_value=origin):
            self.cache.store([1.0, 0.0], "old")
            self.assertEqual(self.redis.ttl(self.cache.key), 300)
        with patch("time.time", return_value=origin + 100):
            self.cache.store([0.0, 1.0], "new")
        with patch("time.time", return_value=origin + 299):
            self.assertEqual(self.cache.lookup([1.0, 0.0]), ("old", 1.0))
        with patch("time.time", return_value=origin + 300):
            self.assertIsNone(self.cache.lookup([1.0, 0.0])[0])
            self.assertEqual(self.cache.lookup([0.0, 1.0]), ("new", 1.0))
        with patch("time.time", return_value=origin + 400):
            self.assertEqual(self.cache.lookup([0.0, 1.0]), (None, 0.0))
            self.assertFalse(self.redis.exists(self.cache.key))

    def test_hit_skips_response_call_but_still_embeds_and_failure_is_not_cached(self):
        client = MagicMock()
        client.embeddings.create.return_value.data[0].embedding = [1.0, 0.0]
        client.responses.create.return_value.output_text = "Paris"
        args = ("France capital?", client, self.cache, "chat", "embedding")
        self.assertEqual(workflow.answer_question(*args), ("Paris", 0.0, False))
        self.assertEqual(workflow.answer_question(*args), ("Paris", 1.0, True))
        self.assertEqual(client.embeddings.create.call_count, 2)
        client.responses.create.assert_called_once()
        client.embeddings.create.return_value.data[0].embedding = [0.0, 1.0]
        client.responses.create.side_effect = RuntimeError("model unavailable")
        with self.assertRaisesRegex(RuntimeError, "model unavailable"):
            workflow.answer_question(*args)
        self.assertEqual(self.redis.zcard(self.cache.key), 1)

    def test_redis_failure_is_visible_without_model_fallback(self):
        client = MagicMock()
        client.embeddings.create.return_value.data[0].embedding = [1.0, 0.0]
        self.server.connected = False
        with self.assertRaises(ConnectionError):
            workflow.answer_question("question", client, self.cache, "chat", "embedding")
        client.responses.create.assert_not_called()


if __name__ == "__main__":
    unittest.main()
