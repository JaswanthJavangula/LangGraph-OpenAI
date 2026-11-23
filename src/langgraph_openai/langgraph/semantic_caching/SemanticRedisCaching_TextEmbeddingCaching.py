"""Redis-backed semantic response cache. See src/README.md for local setup."""

import hashlib
import json
import os
import time

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from redis import Redis

INSTRUCTIONS = "Answer in one sentence."
TTL_SECONDS = 300
MAX_ENTRIES = 10
SIMILARITY_THRESHOLD = 0.9

# Redis time and atomic scripts keep expiration and the size limit consistent
# across processes. Each sorted-set member contains both embedding and response.
READ_ENTRIES = """
local clock = redis.call('TIME')
local now = tonumber(clock[1]) + tonumber(clock[2]) / 1000000
redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', now)
return redis.call('ZRANGE', KEYS[1], 0, -1)
"""

WRITE_ENTRY = """
local clock = redis.call('TIME')
local now = tonumber(clock[1]) + tonumber(clock[2]) / 1000000
redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', now)
redis.call('ZADD', KEYS[1], now + tonumber(ARGV[2]), ARGV[1])
local excess = redis.call('ZCARD', KEYS[1]) - tonumber(ARGV[3])
if excess > 0 then
    redis.call('ZREMRANGEBYRANK', KEYS[1], 0, excess - 1)
end
redis.call('EXPIRE', KEYS[1], ARGV[2])
return redis.call('ZCARD', KEYS[1])
"""


class RedisSemanticCache:
    def __init__(
        self, redis_client: Redis, model: str, embedding_model: str,
        namespace: str = "semantic-demo",
    ) -> None:
        self.redis = redis_client
        identity = json.dumps(
            ["v1", model, embedding_model, INSTRUCTIONS], separators=(",", ":"),
        )
        self.key = f"{namespace}:{hashlib.sha256(identity.encode()).hexdigest()}"
        self._read = self.redis.register_script(READ_ENTRIES)
        self._write = self.redis.register_script(WRITE_ENTRY)

    def lookup(self, embedding: list[float]) -> tuple[str | None, float]:
        entries = [json.loads(value) for value in self._read(keys=[self.key])]
        if not entries:
            return None, 0.0
        similarities = np.array([entry["embedding"] for entry in entries]) @ embedding
        best_match = int(np.argmax(similarities))
        similarity = float(similarities[best_match])
        if similarity > SIMILARITY_THRESHOLD:
            return entries[best_match]["response"], similarity
        return None, similarity

    def store(self, embedding: list[float], response: str) -> None:
        entry = json.dumps(
            {"embedding": embedding, "response": response},
            separators=(",", ":"), allow_nan=False,
        )
        self._write(keys=[self.key], args=[entry, TTL_SECONDS, MAX_ENTRIES])


def answer_question(
    question: str, client: OpenAI, cache: RedisSemanticCache,
    model: str, embedding_model: str,
) -> tuple[str, float, bool]:
    embedding = client.embeddings.create(
        model=embedding_model, input=question,
    ).data[0].embedding
    cached_response, similarity = cache.lookup(embedding)
    if cached_response is not None:
        return cached_response, similarity, True
    response = client.responses.create(
        model=model, instructions=INSTRUCTIONS, input=question,
    ).output_text
    cache.store(embedding, response)
    return response, similarity, False


def main() -> None:
    load_dotenv()
    model = os.environ["OPENAI_MODEL"]
    embedding_model = os.environ["OPENAI_EMBEDDING_MODEL_SMALL"]
    with Redis.from_url(
        os.getenv("REDIS_URL", "redis://localhost:6379/0"),
        decode_responses=True, socket_connect_timeout=5, socket_timeout=5,
    ) as redis_client, OpenAI() as client:
        redis_client.ping()
        cache = RedisSemanticCache(
            redis_client, model, embedding_model,
            namespace=os.getenv("REDIS_CACHE_NAMESPACE", "semantic-demo"),
        )
        questions = [
            "What is the capital of France?",
            "What is the capital of India?",
            "What is the capital of France?",
            "Tell me france country capital?",
        ]
        for question in questions:
            start = time.perf_counter()
            response, similarity, hit = answer_question(
                question, client, cache, model, embedding_model,
            )
            source = "cache" if hit else "Model"
            print(f"Similarity : {similarity} & Value from {source} : {response}")
            print(time.perf_counter() - start)


if __name__ == "__main__":
    main()
