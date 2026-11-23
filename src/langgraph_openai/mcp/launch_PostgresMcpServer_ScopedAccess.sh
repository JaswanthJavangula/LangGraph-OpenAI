#!/usr/bin/env bash
set -euo pipefail

# EDIT THIS VALUE, then restart the server to apply the new permissions.
# "employee:read" = get_employee; "employee:list" = list_employees.
# An empty value disables both tools. These permissions apply to the whole process.
export MCP_TOOL_SCOPES="employee:list employee:read"

# Startup:
# 1. Install uv: https://docs.astral.sh/uv/getting-started/installation/
# 2. Start your existing PostgreSQL server.
# 3. Set DATABASE_URL in your environment or src/langgraph_openai/.env.
#    Do not put database credentials in this script or commit your .env.
# 4. Run: bash /absolute/path/to/launch_PostgresMcpServer_ScopedAccess.sh
#
# This launches a foreground STDIO MCP server, not an HTTP endpoint or chat UI.
# Configure your MCP client to run "bash" with this script's absolute path as
# its argument. The client sends protocol requests; Ctrl+C stops a manual run.
# Dependencies are installed in an isolated uv environment, not the project venv.
# The database user needs CREATE on schema public and SELECT/INSERT on
# public.mcp_employee. Startup creates this demo table and inserts missing
# synthetic rows (IDs 1-5). Existing rows are preserved; hr.employee is untouched.
# For tests, call get_employee with employee_id=1 or list_employees from a client.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../../.." && pwd)"
cd "$repo_root"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required. Install it using the link in this script, then rerun." >&2
    exit 1
fi

export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
export PGCONNECT_TIMEOUT="${PGCONNECT_TIMEOUT:-5}"
python=(
    uv run --no-project --python 3.14
    --with "mcp>=2,<3"
    --with "psycopg2-binary>=2.9,<3"
    --with "python-dotenv>=1,<2"
    python
)

# STDIO reserves stdout for MCP messages; all startup diagnostics go to stderr.
echo "[1/2] Preparing dependencies and initializing public.mcp_employee..." >&2
"${python[@]}" - >&2 <<'PY'
from contextlib import closing
import os
from pathlib import Path
import sys

from langgraph_openai.mcp.PostgresMcpServer_ScopedAccess import (
    _COLUMNS, _TABLE, _TOOL_SCOPES, get_connection,
)

if not os.getenv("DATABASE_URL", "").strip():
    sys.exit("Set DATABASE_URL in the environment or src/langgraph_openai/.env.")

unknown = _TOOL_SCOPES - {"employee:list", "employee:read"}
if unknown:
    sys.exit("MCP_TOOL_SCOPES contains an unsupported value. Use employee:list and/or employee:read.")

if not _TOOL_SCOPES:
    print("WARNING: MCP_TOOL_SCOPES is empty; both tools will deny access.")

with closing(get_connection()) as conn:
    with conn:
        with conn.cursor() as cur:
            cur.execute("SET LOCAL statement_timeout = '5s'")
            cur.execute("SELECT pg_advisory_xact_lock(hashtext('mcp_employee_demo_setup'))")
            cur.execute(Path("src/langgraph_openai/db/mcp_employee.sql").read_text(encoding="utf-8"))
            cur.execute(f"SELECT {', '.join(_COLUMNS)} FROM {_TABLE} LIMIT 0")

print("public.mcp_employee ready: synthetic IDs 1-5 inserted if missing; existing rows preserved.")
print("Enabled tool scopes:", " ".join(sorted(_TOOL_SCOPES)) or "(none)")
PY

echo "[2/2] Starting STDIO MCP server; waiting for an MCP client (Ctrl+C to stop)..." >&2
exec "${python[@]}" -m langgraph_openai.mcp.PostgresMcpServer_ScopedAccess