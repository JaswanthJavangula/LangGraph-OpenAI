#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# load DATABASE_URL (and friends) from .env, exporting them for psql
set -a
source "$DIR/../.env"
set +a

DB_NAME="${DB_NAME:-langgraph_openai}"

createdb "$DB_NAME" 2>/dev/null || true

psql "$DATABASE_URL" -f "$DIR/schema.sql"
psql "$DATABASE_URL" -f "$DIR/Postgres_TestData.sql"
