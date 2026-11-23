"""Real MCP SDK tests with mocked tools' database connections.

Set TEST_DATABASE_URL to an empty, dedicated PostgreSQL test database to also
test the seed SQL. That test rolls back all its changes.
Run with the launcher's isolated mcp/psycopg2-binary/python-dotenv dependencies:
PYTHONPATH=src uv run --no-project --python 3.14 --with 'mcp>=2,<3' \
--with 'psycopg2-binary>=2.9,<3' --with 'python-dotenv>=1,<2' \
python -m unittest discover -s tests -p test_mcp_scoped_access.py -v
"""

from contextlib import closing
from datetime import date
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from mcp import Client
import psycopg2

from langgraph_openai.mcp import PostgresMcpServer_ScopedAccess as workflow


class ScopedToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_authorized_calls_query_demo_table_and_preserve_output(self):
        with patch.object(workflow, "_TOOL_SCOPES", frozenset({"employee:list", "employee:read"})):
            with patch.object(workflow, "get_connection") as connect:
                cursor = connect.return_value.cursor.return_value
                row = (1, "Demo Employee One", date(1990, 1, 15), None, None)
                cursor.fetchall.return_value = [row]
                cursor.fetchone.return_value = row
                async with Client(workflow.server) as client:
                    listed = await client.call_tool("list_employees")
                    self.assertFalse(listed.is_error)
                    self.assertIn("Demo Employee One", str(listed.structured_content))
                    fetched = await client.call_tool("get_employee", {"employee_id": 1})
                    self.assertFalse(fetched.is_error)
                    self.assertIn("1990-01-15", str(fetched.structured_content))
                queries = cursor.execute.call_args_list
                self.assertEqual(len(queries), 2)
                for call in queries:
                    self.assertIn("FROM public.mcp_employee", call.args[0])
                    self.assertNotIn("hr.employee", call.args[0])
                self.assertEqual(queries[1].args[1], (1,))
                self.assertEqual(connect.return_value.close.call_count, 2)

    async def test_missing_scope_denies_before_connecting(self):
        for scopes, tool, arguments in [
            (frozenset(), "list_employees", {}),
            (frozenset(), "get_employee", {"employee_id": 1}),
            (frozenset({"employee:read"}), "list_employees", {}),
            (frozenset({"employee:list"}), "get_employee", {"employee_id": 1}),
        ]:
            with self.subTest(scopes=scopes, tool=tool):
                with patch.object(workflow, "_TOOL_SCOPES", scopes):
                    with patch.object(workflow, "get_connection") as connect:
                        with self.assertLogs(level="WARNING"):
                            async with Client(workflow.server) as client:
                                result = await client.call_tool(tool, arguments)
                        self.assertTrue(result.is_error)
                        connect.assert_not_called()

    async def test_unknown_demo_employee_returns_null(self):
        with patch.object(workflow, "_TOOL_SCOPES", frozenset({"employee:read"})):
            with patch.object(workflow, "get_connection") as connect:
                connect.return_value.cursor.return_value.fetchone.return_value = None
                async with Client(workflow.server) as client:
                    result = await client.call_tool("get_employee", {"employee_id": 999})
                    self.assertFalse(result.is_error)
                    self.assertIsNone(result.structured_content["result"])


@unittest.skipUnless(os.getenv("TEST_DATABASE_URL"), "Set TEST_DATABASE_URL for PostgreSQL seed tests")
class DemoSeedTests(unittest.TestCase):
    def test_seed_is_repeatable_preserves_existing_rows_and_restores_missing_rows(self):
        sql_path = Path(__file__).resolve().parents[1] / "src/langgraph_openai/db/mcp_employee.sql"
        setup_sql = sql_path.read_text(encoding="utf-8")
        with closing(psycopg2.connect(os.environ["TEST_DATABASE_URL"], connect_timeout=5)) as conn:
            try:
                with conn.cursor() as cur:
                    cur.execute("SET LOCAL statement_timeout = '5s'")
                    cur.execute("SELECT to_regclass('public.mcp_employee')")
                    if cur.fetchone()[0] is not None:
                        self.skipTest("Use an empty dedicated test database; demo table already exists")
                    cur.execute(setup_sql)
                    cur.execute("SELECT employee_id, name, dob FROM public.mcp_employee ORDER BY employee_id")
                    rows = cur.fetchall()
                    self.assertEqual(len(rows), 5)
                    self.assertEqual(rows[0], (1, "Demo Employee One", date(1990, 1, 15)))
                    self.assertEqual(rows[-1], (5, "Demo Employee Five", None))
                    cur.execute("UPDATE public.mcp_employee SET name = 'Preserved demo edit' WHERE employee_id = 1")
                    cur.execute("DELETE FROM public.mcp_employee WHERE employee_id = 5")
                    cur.execute(setup_sql)
                    cur.execute("SELECT count(*) FROM public.mcp_employee")
                    self.assertEqual(cur.fetchone()[0], 5)
                    cur.execute("SELECT name FROM public.mcp_employee WHERE employee_id = 1")
                    self.assertEqual(cur.fetchone()[0], "Preserved demo edit")
                    cur.execute(setup_sql)
                    cur.execute("SELECT count(*) FROM public.mcp_employee")
                    self.assertEqual(cur.fetchone()[0], 5)
            finally:
                conn.rollback()


if __name__ == "__main__":
    unittest.main()
