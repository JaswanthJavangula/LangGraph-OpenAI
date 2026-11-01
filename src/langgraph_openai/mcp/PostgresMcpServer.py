from contextlib import closing

from mcp.server.mcpserver import MCPServer, UserMessage

from langgraph_openai.db.connection import get_connection

server = MCPServer("postgres-employee-server")

_COLUMNS = ["employee_id", "name", "dob", "created_at", "updated_at"]


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
    """List all employees in hr.employee."""
    with closing(get_connection()) as conn:
        cur = conn.cursor()
        cur.execute(f"SELECT {', '.join(_COLUMNS)} FROM hr.employee ORDER BY employee_id")
        return [_row_to_dict(row) for row in cur.fetchall()]


@server.tool()
def get_employee(employee_id: int) -> dict | None:
    """Fetch a single employee by employee_id."""
    with closing(get_connection()) as conn:
        cur = conn.cursor()
        cur.execute(f"SELECT {', '.join(_COLUMNS)} FROM hr.employee WHERE employee_id = %s", (employee_id,))
        row = cur.fetchone()
        return _row_to_dict(row) if row else None


#tool decides to call; prompt is when user manually requests
#claudeui -> + -> connectors -> lookup_employee -> value
@server.prompt()
def lookup_employee(employee_id: int) -> list[UserMessage]:
    """Ask about a specific employee by employee_id."""
    return [UserMessage(f"Look up employee {employee_id} and summarize their details.")]


if __name__ == "__main__":
    server.run()
