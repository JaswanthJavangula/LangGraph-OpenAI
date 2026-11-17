#!/usr/bin/env bash
# Launches the HITL approval API in the background, then the HIL CLI agent in
# the foreground. Pass CLI args through, e.g.: ./launch.sh --thread-id demo-1
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"
python="$repo_root/.venv/bin/python"
port="${APPROVAL_API_PORT:-8001}"

if [[ ! -x "$python" ]]; then
    echo "Python environment not found at $python; create the project .venv and install its dependencies." >&2
    exit 1
fi

source "$repo_root/.venv/bin/activate"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"

if lsof -nP -iTCP:"$port" -sTCP:LISTEN -t >/dev/null; then
    echo "Port $port is already in use; stop its server or set APPROVAL_API_PORT to another port." >&2
    exit 1
fi

"$python" -m uvicorn \
    langgraph_openai.langgraph.approval_api:app \
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
        echo "Approval API server failed to start." >&2
        exit 1
    fi
    if curl --noproxy '*' -fsS --max-time 1 "http://127.0.0.1:$port/openapi.json" >/dev/null 2>&1; then
        ready=true
        break
    fi
    sleep 1
done

if [[ "$ready" != true ]]; then
    echo "Approval API server did not become ready on port $port." >&2
    exit 1
fi

echo "Approval API Swagger UI: http://127.0.0.1:$port/docs"
echo "The approval API will stop when the agent exits."
"$python" -m langgraph_openai.langgraph.HIL_Approval_NextNode "$@"
