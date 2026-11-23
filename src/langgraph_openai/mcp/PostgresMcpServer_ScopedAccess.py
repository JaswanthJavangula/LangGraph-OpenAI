"""Local, process-wide tool permissions configured by the trusted launcher.

Set MCP_TOOL_SCOPES="employee:list employee:read" to enable both tools, or grant
only the required space-separated scope. Missing/empty scopes deny both tools.
Scopes are read at startup; restart the server after changing them.
This is not per-user authorization and does not restrict employee rows or fields.
The launcher seeds public.mcp_employee with synthetic data; hr.employee is unused.
"""

from contextlib import closing
import logging
import os

from mcp.server.mcpserver import MCPServer, UserMessage

from langgraph_openai.db.connection import get_connection

server = MCPServer("postgres-employee-server")

logger = logging.getLogger(__name__)
_TOOL_SCOPES = frozenset(os.environ.get("MCP_TOOL_SCOPES", "").split())
_TABLE = "public.mcp_employee"
_COLUMNS = ["employee_id", "name", "dob", "created_at", "updated_at"]


def _require_scope(scope: str) -> None:
    if scope not in _TOOL_SCOPES:
        logger.warning("MCP tool access denied: missing required scope '%s'.", scope)
        raise PermissionError(f"Missing required tool scope: {scope}")


#isoformat converts to strings; it is used because the date objects are not JSON native by default
def _row_to_dict(row: tuple) -> dict:
    employee_id, name, dob, created_at, updated_at = row
    return {
        "employee_id": employee_id,
        "name": name,
        "dob": dob.isoformat() if dob else None,
        "created_at": created_at.isoformat() if created_at else None,
        "updated_at": updated_at.isoformat() if updated_at else None,
    }



@server.tool()
def list_employees() -> list[dict]:
    """List demo employees in public.mcp_employee. Requires employee:list."""
    _require_scope("employee:list")
    with closing(get_connection()) as conn:
        cur = conn.cursor()
        cur.execute(f"SELECT {', '.join(_COLUMNS)} FROM {_TABLE} ORDER BY employee_id")
        return [_row_to_dict(row) for row in cur.fetchall()]


@server.tool()
def get_employee(employee_id: int) -> dict | None:
    """Fetch a demo employee by employee_id. Requires employee:read."""
    _require_scope("employee:read")
    with closing(get_connection()) as conn:
        cur = conn.cursor()
        cur.execute(f"SELECT {', '.join(_COLUMNS)} FROM {_TABLE} WHERE employee_id = %s", (employee_id,))
        row = cur.fetchone()
        return _row_to_dict(row) if row else None


#tool decides to call; prompt is when user manually requests
#claudeui -> + -> connectors -> lookup_employee -> value
@server.prompt()
def lookup_employee(employee_id: int) -> list[UserMessage]:
    """Ask about a specific demo employee by employee_id."""
    return [UserMessage(f"Look up demo employee {employee_id} and summarize their details.")]


if __name__ == "__main__":
    server.run()
