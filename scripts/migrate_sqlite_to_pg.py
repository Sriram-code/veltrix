#!/usr/bin/env python3
"""
One-shot migration: copies all data from the SQLite veltrix.db into PostgreSQL.

Usage:
    python scripts/migrate_sqlite_to_pg.py

Environment variables required:
    DATABASE_URL   - PostgreSQL connection string
    DB_PATH        - path to the SQLite file (default: ./data/veltrix.db)

The script is idempotent: it TRUNCATEs target tables before inserting.
Run it ONCE after provisioning the Render Postgres database.
"""
from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path

# Resolve project root so we can import app.config
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Load .env before importing settings
from app.config import _load_dotenv, settings  # noqa: E402

_load_dotenv()

import psycopg2  # noqa: E402
import psycopg2.extras  # noqa: E402

# ---------------------------------------------------------------------------
# Tables to migrate (order respects FK constraints)
# ---------------------------------------------------------------------------
TABLES = [
    "employees",
    "leave_balances",
    "leave_requests",
    "ingested_holidays",
]


def sqlite_rows(sqlite_conn: sqlite3.Connection, table: str) -> tuple[list[str], list[tuple]]:
    sqlite_conn.row_factory = sqlite3.Row
    cur = sqlite_conn.execute(f"SELECT * FROM {table}")
    rows = cur.fetchall()
    if not rows:
        return [], []
    cols = list(rows[0].keys())
    data = [tuple(r) for r in rows]
    return cols, data


def pg_upsert(pg_conn, table: str, cols: list[str], rows: list[tuple]) -> int:
    if not rows:
        return 0
    with pg_conn.cursor() as cur:
        cur.execute(f"TRUNCATE TABLE {table} CASCADE")
        placeholders = "(" + ",".join(["%s"] * len(cols)) + ")"
        col_str = "(" + ",".join(cols) + ")"
        psycopg2.extras.execute_values(
            cur,
            f"INSERT INTO {table} {col_str} VALUES %s",
            rows,
            template=placeholders,
        )
        return len(rows)


def main() -> None:
    db_path = settings.db_path
    database_url = settings.database_url or os.getenv("DATABASE_URL", "")

    if not database_url:
        sys.exit("ERROR: DATABASE_URL environment variable is not set.")
    if not Path(db_path).exists():
        sys.exit(f"ERROR: SQLite file not found: {db_path}")

    print(f"Source SQLite : {db_path}")
    print(f"Target Postgres: {database_url[:40]}…")
    print()

    sqlite_conn = sqlite3.connect(db_path)
    pg_conn = psycopg2.connect(database_url)
    pg_conn.autocommit = False

    try:
        for table in TABLES:
            try:
                cols, rows = sqlite_rows(sqlite_conn, table)
            except sqlite3.OperationalError:
                print(f"  SKIP  {table:30s} (table not found in SQLite)")
                continue

            count = pg_upsert(pg_conn, table, cols, rows)
            print(f"  OK    {table:30s}  {count:5d} rows")

        pg_conn.commit()
        print("\nMigration complete ✓")
    except Exception as exc:
        pg_conn.rollback()
        sys.exit(f"\nMigration FAILED (rolled back): {exc}")
    finally:
        sqlite_conn.close()
        pg_conn.close()


if __name__ == "__main__":
    main()
