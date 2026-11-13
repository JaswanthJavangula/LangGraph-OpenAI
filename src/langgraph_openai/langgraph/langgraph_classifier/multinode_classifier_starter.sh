#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../../../.." && pwd)"
python="$repo_root/.venv/bin/python"
port="${CHECKPOINT_API_PORT:-8000}"

if [[ ! -x "$python" ]]; then
    echo "Python environment not found at $python; create the project .venv and install its dependencies." >&2
    exit 1
fi

if [[ ! "$port" =~ ^[0-9]+$ ]] || (( 10#$port < 1 || 10#$port > 65535 )); then
    echo "CHECKPOINT_API_PORT must be a port number between 1 and 65535." >&2
    exit 1
fi

if lsof -nP -iTCP:"$port" -sTCP:LISTEN -t >/dev/null; then
    echo "Port $port is already in use; stop its server or set CHECKPOINT_API_PORT to another port." >&2
    exit 1
fi

export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
"$python" -m uvicorn \
    langgraph_openai.langgraph.langgraph_classifier.checkpoint_api:app \
    --host 127.0.0.1 --port "$port" &
server_pid=$!

cleanup() {
    if kill -0 "$server_pid" 2>/dev/null; then
        kill "$server_pid"
    fi
    wait "$server_pid" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

ready=false
for ((attempt = 0; attempt < 30; attempt++)); do
    if ! kill -0 "$server_pid" 2>/dev/null; then
        echo "Checkpoint API server failed to start." >&2
        exit 1
    fi
    if curl --noproxy '*' -fsS --max-time 1 "http://127.0.0.1:$port/openapi.json" >/dev/null 2>&1; then
        ready=true
        break
    fi
    sleep 1
done

if [[ "$ready" != true ]]; then
    echo "Checkpoint API server did not become ready on port $port." >&2
    exit 1
fi

echo "Swagger UI: http://127.0.0.1:$port/docs"
echo "The checkpoint API will stop when the agent exits."
"$python" -m langgraph_openai.langgraph.langgraph_classifier.langgraph_classifier_multinode_agent "$@"
