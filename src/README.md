# LangGraph-OpenAI

A small Python project demonstrating a LangGraph workflow powered by OpenAI.

## Requirements

- Python 3.10+
- An OpenAI API key

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .
cp .env.example .env
```

Set `OPENAI_API_KEY` in `.env`, then run:

```bash
langgraph-openai "Explain LangGraph in one sentence"
```

You can also run the module directly:

```bash
python -m langgraph_openai "What is a state graph?"
```

## Project structure

- `src/langgraph_openai/graph.py` defines the LangGraph workflow.
- `src/langgraph_openai/main.py` provides the command-line entry point.
- `.env.example` documents the required environment variables.

## License

This sample is provided for demonstration purposes.

## Redis semantic response caching example

Implementation: [SemanticRedisCaching_TextEmbeddingCaching.py](langgraph_openai/langgraph/semantic_caching/SemanticRedisCaching_TextEmbeddingCaching.py).
The original in-memory embedding example is unchanged.

### One-command launcher

Set `OPENAI_API_KEY`, `OPENAI_MODEL`, and `OPENAI_EMBEDDING_MODEL_SMALL` in your
environment or a `.env` beside the example or in a parent directory. For example,
use `gpt-4o-mini` and `text-embedding-3-small` for the two models.
Install `uv` and Docker Desktop (or Docker with Colima) first, then run:

```bash
bash src/langgraph_openai/langgraph/semantic_caching/launch_SemanticRedisCaching_TextEmbeddingCaching.sh
```

The [launcher](langgraph_openai/langgraph/semantic_caching/launch_SemanticRedisCaching_TextEmbeddingCaching.sh)
installs dependencies, validates configuration, starts the selected default
Colima context or Docker Desktop on macOS if needed, creates/reuses the named
Redis container, waits for readiness, checks the connection, and runs the demo.
It resolves paths relative to itself, so it can also be invoked from another directory.
Other Docker contexts/platforms require their Docker runtime to be running already.

Redis stays running afterward so subsequent launches can reuse cached responses.
Stop it with `docker stop langgraph-semantic-redis`.
The namespace stays unchanged across launches unless you set `REDIS_CACHE_NAMESPACE`.
For any `REDIS_URL` other than `redis://localhost:6379/0` or
`redis://127.0.0.1:6379/0`, the launcher skips Docker and connects to the configured
server, which must already be running. No credentials are printed by the launcher.

### Manual setup

From the repository root, with Python 3.14+ and Docker running:

```bash
uv sync --extra redis-caching
docker run -d --name langgraph-semantic-redis \
  -p 127.0.0.1:6379:6379 \
  -v langgraph-semantic-redis-data:/data redis:7-alpine redis-server --appendonly yes
docker exec langgraph-semantic-redis redis-cli ping

export REDIS_URL='redis://localhost:6379/0'
export OPENAI_MODEL='gpt-4o-mini'
export OPENAI_EMBEDDING_MODEL_SMALL='text-embedding-3-small'
# Set OPENAI_API_KEY in your environment or local .env; never commit it.
uv run --extra redis-caching python \
  src/langgraph_openai/langgraph/semantic_caching/SemanticRedisCaching_TextEmbeddingCaching.py
```

The ping should return `PONG`. On later runs, use
`docker start langgraph-semantic-redis` instead of creating another container.
Stop it with `docker stop langgraph-semantic-redis` when finished.
The loopback-only Redis setup is for local development; use authentication and
TLS (`rediss://`) for remote deployments.

- Redis replaces both Python cache lists with **up to 10 paired records**:
  10 embeddings and their 10 responses, per model/prompt/namespace combination.
- Each pair expires **300 seconds after insertion**. Hits do not extend its
  lifetime. When full, the oldest pair is evicted; reads/writes remove expired
  pairs atomically. An idle cache key also expires automatically.
- Each request still calls the embedding API. The script fetches the small
  Redis collection and computes the same dot-product similarity locally.
  Only a best score **strictly greater than 0.9** skips response generation.
- Cache hits survive Python process restarts. The Docker volume and AOF preserve
  Redis data across container restarts, subject to expiration.
- Keys include the chat model, embedding model, instructions and cache version.
  Each result prints the Redis `Cache-Key`; it identifies the shared sorted set
  holding the embedding/response pairs, not an individual question.
  Set `REDIS_CACHE_NAMESPACE` (default `semantic-demo`) separately for different
  users/tenants or datasets; do not share private responses across those scopes.
- Redis/OpenAI failures propagate explicitly; there is no silent L1 fallback.
  Concurrent misses can still generate duplicate model calls.

Offline tests exercise Redis commands and Lua using fakeredis (no API key needed):

```bash
uv run --extra redis-caching --group test python -m unittest discover \
  -s tests -p 'test_redis_semantic_*.py' -v
```

## Combined multi-node, persistence, HIL and L1 caching example

Implementation: [HIL_MultiNode_Caching_Persistance.py](langgraph_openai/langgraph/HIL/HIL_MultiNode_Caching_Persistance.py).
This is a separate example; it does not change the existing classifier or approval API.
Its classes separate cache storage, Chroma retrieval, node logic, conversation
lifecycle, and CLI/API entry points while keeping implementation in one file.

### Setup and run

This example uses the project's Python 3.14+ runtime. From the repository root:

```bash
.venv/bin/python -m pip install -e '.[caching]'
export DATABASE_URL='postgresql://user:password@localhost:5432/your_database'
export OPENAI_MODEL='gpt-4o-mini'
export OPENAI_API_KEY='your-key'
export CHROMA_PATH='src/langgraph_openai/langchain/chroma_db'
export CHROMA_COLLECTION='HIL_Caching_node_persistance'
export CHROMA_REVISION='1'

PYTHONPATH=src .venv/bin/python -m langgraph_openai.langgraph.HIL.HIL_MultiNode_Caching_Persistance --thread-id cached-demo
```

Alternatively, put environment values in `src/langgraph_openai/.env`. PostgreSQL
must already be running; its database user needs permission to create checkpoint
tables. Reuse the thread ID to retain history across restarts.

The Chroma collection is created empty if absent; existing collections are not
overwritten and no sample documents are added. Populate it with your own documents
using Chroma's `collection.add(...)` before expecting grounded knowledge answers.
Queries use Chroma's default embedding function, which may download its embedding
model on first use. Documents must use the same compatible embedding function.
An empty collection logs a warning and the knowledge prompt explicitly asks the
model to report insufficient context rather than invent a retrieved answer.

### Graph and human approval

```text
START -> classifier_agent -> chat_agent --------------------------> END
                         -> retrieval_agent -> knowledge_agent --> END
                         -> request_code_change_approval
                               -> approved -> code_agent --------> END
                               -> rejected -> code_change_rejected -> END
```

Only code requests pause. This example generates a coding response; it does not
edit files or execute code. Each new code request requires fresh approval,
including repeated/cached questions. Approval decisions are never cached.

When the CLI pauses, it prints the interrupt and exits. Approve or reject with:

```bash
PYTHONPATH=src .venv/bin/python -m langgraph_openai.langgraph.HIL.HIL_MultiNode_Caching_Persistance --thread-id cached-demo --approve approved
# Substitute "rejected" to skip the code response.
```

Use the same module and `--thread-id cached-demo --resume` after a failed,
unfinished run. A pending approval requires `--approve`, not `--resume`.
New questions are rejected while the conversation has unfinished work.

### What each L1 layer does

All layers share a process-local TTL/LRU store: **300 seconds per entry and at
most 512 entries total across all conversations/layers**, with exact-match keys
partitioned by conversation ID. Hits do not renew an entry's expiration.

| Layer | Cached value | Key boundary |
|---|---|---|
| Human question | Intent classification, not a final answer | Exact latest question plus model/prompt/schema identity |
| Node input | Prepared classifier/agent messages or retrieval inputs | Node identity and all meaningful inputs |
| Node output | Classification result or response text | Prepared classifier input or agent's full semantic history and retrieved context |
| LLM response | Successful structured classification or answer text | Complete model request identity |
| Retrieval | Chroma document texts, including an empty result | Question, collection path/name, revision and top-k |

The outer layer may satisfy a request before an inner layer is consulted; every
layer's hit count need not increase on a repeated question. Full-history response
keys deliberately miss when the same question is asked later in a conversation.
The context-free classifier and question-based retrieval can still hit.
Transport message IDs and token-usage metadata do not affect response keys.
Cached answers become fresh assistant messages so message IDs are not reused.

Failures are propagated, not cached. Retrieval uses only its own result cache
(not an additional node-output cache) to avoid masking retrieval expiration.
Change `CHROMA_REVISION` and restart after changing collection contents, or clear
each affected conversation's cache immediately. Bump `CACHE_VERSION` when changing
node behavior. Model configuration, prompt text and classifier schema are included
in keys. The local demo uses a fixed model at temperature zero.

**Persistence is not caching:** PostgreSQL saves graph state and pending approvals;
Chroma persists documents; L1 saves disposable computations. A restart loses all
L1 entries, not conversation history. Each cache layer has its own entry lifetime;
this is not a hard end-to-end freshness guarantee for external data.

### New local API

```bash
PYTHONPATH=src .venv/bin/python -m langgraph_openai.langgraph.HIL.HIL_MultiNode_Caching_Persistance --serve
```

Swagger: `http://127.0.0.1:8002/docs`. The API creates one graph/cache for its
lifespan, not a new cache per HTTP request.

| Method | Route | Body / purpose |
|---|---|---|
| POST | `/conversations/{thread_id}/messages` | `{"question":"Refactor this function"}` |
| GET | `/conversations/{thread_id}/pending` | Inspect pending approval |
| POST | `/conversations/{thread_id}/approve` | `{"decision":"approved"}` or `{"decision":"rejected"}` |
| POST | `/conversations/{thread_id}/resume` | Resume unfinished non-approval work |
| GET | `/conversations/{thread_id}/cache` | Per-conversation entry counts and process-wide layer hit/miss totals |
| DELETE | `/conversations/{thread_id}/cache` | Clear this conversation's five cache layers, not its checkpoints |

```bash
curl -X POST http://127.0.0.1:8002/conversations/api-demo/messages \
  -H 'Content-Type: application/json' \
  -d '{"question":"Refactor this function: def add(a,b): return a+b"}'

curl -X POST http://127.0.0.1:8002/conversations/api-demo/approve \
  -H 'Content-Type: application/json' -d '{"decision":"approved"}'

curl http://127.0.0.1:8002/conversations/api-demo/cache
```

Agent replies contain `thread_id`, `pending`, `interrupts` and `message`. A paused
reply has `message: null`. Missing conversations return 404 on pending inspection;
invalid lifecycle operations return 409; invalid request bodies return 422.
Unexpected model/database failures remain server errors, not successful replies.

**Local demo only:** no authentication. Conversation IDs partition cache entries,
but are not authorization. Do not expose this service publicly. Cached content
is held in process memory until expiration/eviction/clearing; cache stats expose
counts only. Use one worker and do not operate on the same thread concurrently
from separate CLI/API processes. Local service operations are serialized; a
production service needs authenticated access and cross-process concurrency control.
Do not use the older approval API to resume this graph: its node wiring differs.

### Validation

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -p test_hil_caching.py -v
```

Offline tests cover cache hits/misses, isolation, TTL boundaries, LRU capacity,
all routes, fresh approvals, failure recovery, cache invalidation and Chroma
collection persistence. Models are faked; tests do not send prompts to OpenAI or
download embedding models. Set `TEST_DATABASE_URL` to a **dedicated test database**
to also test PostgreSQL reconnect/resume. That test deletes only its own UUID thread.
