"""Exercise the launcher with fake executables; never start services or call APIs."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


LAUNCHER = (
    Path(__file__).resolve().parents[1]
    / "src/langgraph_openai/langgraph/semantic_caching"
    / "launch_SemanticRedisCaching_TextEmbeddingCaching.sh"
)

FAKE_COMMAND = """#!/usr/bin/env python3
import os
from pathlib import Path
import sys

name = Path(sys.argv[0]).name
args = sys.argv[1:]
with open(os.environ["COMMAND_LOG"], "a") as log:
    log.write(name + " " + " ".join(args) + "\\n")
if name == "uv":
    if args[0] == "sync" and os.getenv("FAIL_SYNC"):
        sys.exit(1)
    if args[:4] == ["run", "--no-sync", "python", "-"] and len(args) == 5:
        print(os.getenv("TEST_REDIS_URL", "redis://localhost:6379/0"))
elif name == "docker":
    if args == ["info"] and os.getenv("START_COLIMA"):
        if not Path(os.environ["RUNTIME_MARKER"]).exists():
            sys.exit(1)
    elif args == ["context", "show"]:
        print("colima")
    elif args[:2] == ["container", "ls"]:
        if os.getenv("EXISTING_CONTAINER"):
            print("langgraph-semantic-redis")
    elif args[0] == "inspect":
        print(os.getenv("CONTAINER_IMAGE", "redis:7-alpine"))
    elif args[0] == "exec":
        print("PONG")
elif name == "colima":
    Path(os.environ["RUNTIME_MARKER"]).touch()
"""


class RedisLauncherTests(unittest.TestCase):
    def run_launcher(self, **overrides):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for command in ("uv", "docker", "colima"):
                executable = root / command
                executable.write_text(FAKE_COMMAND)
                executable.chmod(0o755)
            log = root / "commands.log"
            env = {
                **os.environ,
                "PATH": f"{root}{os.pathsep}{os.environ['PATH']}",
                "COMMAND_LOG": str(log),
                "RUNTIME_MARKER": str(root / "runtime-ready"),
                **overrides,
            }
            result = subprocess.run(
                ["bash", str(LAUNCHER)], cwd=directory, env=env,
                text=True, capture_output=True, timeout=15,
            )
            return result, log.read_text()

    def test_starts_runtime_creates_redis_and_runs_demo_from_another_directory(self):
        result, commands = self.run_launcher(START_COLIMA="1")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("colima start", commands)
        self.assertIn("docker run -d --name langgraph-semantic-redis", commands)
        self.assertIn("-p 127.0.0.1:6379:6379", commands)
        self.assertIn("redis-cli ping", commands)
        self.assertTrue(commands.rstrip().endswith("SemanticRedisCaching_TextEmbeddingCaching.py"))

    def test_reuses_existing_container(self):
        result, commands = self.run_launcher(EXISTING_CONTAINER="1")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("docker start langgraph-semantic-redis", commands)
        self.assertNotIn("docker run", commands)

    def test_external_url_skips_docker(self):
        result, commands = self.run_launcher(TEST_REDIS_URL="redis://external.example:6379/0")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("docker ", commands)
        self.assertIn("Verifying the Redis connection", result.stdout)

    def test_dependency_failure_stops_before_services_start(self):
        result, commands = self.run_launcher(FAIL_SYNC="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("docker ", commands)
        self.assertNotIn("uv run", commands)

    def test_unexpected_container_is_not_started_or_replaced(self):
        result, commands = self.run_launcher(EXISTING_CONTAINER="1", CONTAINER_IMAGE="other")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("refusing to modify", result.stderr)
        self.assertNotIn("docker start", commands)
        self.assertNotIn("docker run", commands)


if __name__ == "__main__":
    unittest.main()
