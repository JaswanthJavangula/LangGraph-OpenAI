#!/usr/bin/env bash
set -euo pipefail

# Run from any directory. Configure OpenAI in the environment or a .env beside
# the example or in a parent directory. Redis stays running after the demo exits.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../../../.." && pwd)"
container="langgraph-semantic-redis"
cd "$repo_root"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required. Install it from https://docs.astral.sh/uv/getting-started/installation/." >&2
    exit 1
fi

echo "[1/5] Installing Redis demo dependencies..."
uv sync --extra redis-caching

echo "[2/5] Checking configuration..."
# Match the Python example's nearest-.env lookup without executing it as shell.
redis_url="$(uv run --no-sync python - "$script_dir" <<'PY'
import os
import sys
from dotenv import find_dotenv, load_dotenv

os.chdir(sys.argv[1])
load_dotenv(find_dotenv(usecwd=True))
required = ("OPENAI_API_KEY", "OPENAI_MODEL", "OPENAI_EMBEDDING_MODEL_SMALL")
missing = [name for name in required if not os.getenv(name, "").strip()]
if missing:
    sys.exit(
        "Set these variables in your environment or .env before launching: "
        + ", ".join(missing)
        + "\nExample models: OPENAI_MODEL=gpt-4o-mini, "
        "OPENAI_EMBEDDING_MODEL_SMALL=text-embedding-3-small"
    )
print(os.getenv("REDIS_URL", "redis://localhost:6379/0"))
PY
)"
export REDIS_URL="$redis_url"

if [[ "$REDIS_URL" == "redis://localhost:6379/0" || "$REDIS_URL" == "redis://127.0.0.1:6379/0" ]]; then
    echo "[3/5] Starting local Redis..."
    if ! command -v docker >/dev/null 2>&1; then
        echo "Install Docker Desktop or Docker with Colima, then rerun this launcher." >&2
        exit 1
    fi

    if ! docker info >/dev/null 2>&1; then
        context="$(docker context show)"
        if [[ "$context" == "colima" ]] && command -v colima >/dev/null 2>&1; then
            echo "Starting Colima..."
            colima start
        elif [[ "$(uname -s)" == "Darwin" && ( "$context" == "desktop-linux" || "$context" == "default" ) ]]; then
            echo "Starting Docker Desktop..."
            open -a Docker
        else
            echo "Docker is unavailable for context '$context'. Start its runtime and rerun." >&2
            exit 1
        fi
        ready=false
        for ((attempt = 0; attempt < 60; attempt++)); do
            if docker info >/dev/null 2>&1; then
                ready=true
                break
            fi
            sleep 2
        done
        if [[ "$ready" != true ]]; then
            echo "Docker did not become ready within 120 seconds." >&2
            exit 1
        fi
    fi

    # List first so daemon/permission errors are not mistaken for a missing container.
    existing="$(docker container ls -a --filter "name=^/${container}$" --format '{{.Names}}')"
    if [[ "$existing" == "$container" ]]; then
        image="$(docker inspect --format '{{.Config.Image}}' "$container")"
        if [[ "$image" != "redis:7-alpine" ]]; then
            echo "Container '$container' uses unexpected image '$image'; refusing to modify it." >&2
            exit 1
        fi
        docker start "$container"
    else
        docker run -d --name "$container" \
            -p 127.0.0.1:6379:6379 \
            -v langgraph-semantic-redis-data:/data \
            redis:7-alpine redis-server --appendonly yes
    fi

    ready=false
    for ((attempt = 0; attempt < 30; attempt++)); do
        if [[ "$(docker exec "$container" redis-cli ping 2>/dev/null)" == "PONG" ]]; then
            ready=true
            break
        fi
        sleep 1
    done
    if [[ "$ready" != true ]]; then
        echo "Redis did not become ready. Inspect: docker logs $container" >&2
        exit 1
    fi
    echo "Redis will remain running. To stop it: docker stop $container"
else
    echo "[3/5] Using configured REDIS_URL; skipping local Docker startup."
fi

echo "[4/5] Verifying the Redis connection..."
uv run --no-sync python - <<'PY'
import os
from redis import Redis

with Redis.from_url(
    os.environ["REDIS_URL"], socket_connect_timeout=5, socket_timeout=5,
) as client:
    client.ping()
print("Redis connection ready.")
PY

echo "[5/5] Running the semantic caching demo (10 pairs, 300-second expiration)..."
exec uv run --no-sync python "$script_dir/SemanticRedisCaching_TextEmbeddingCaching.py"