"""Data access layer – backed by PostgreSQL (psycopg2).

Connection string is read from the DATABASE_URL environment variable.
All SQL uses %s placeholders (psycopg2 style).
The schema is created on startup with CREATE ... IF NOT EXISTS, and the
tables are auto-seeded from the baked-in SQLite snapshot when empty.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Optional

import psycopg2
import psycopg2.extras
import psycopg2.pool

from app.config import settings

# ---------------------------------------------------------------------------
# Connection pool (created lazily on first use)
# ---------------------------------------------------------------------------
_pool: Optional[psycopg2.pool.ThreadedConnectionPool] = None


def _get_pool() -> psycopg2.pool.ThreadedConnectionPool:
    global _pool
    if _pool is None:
        _pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=1,
            maxconn=5,
            dsn=settings.database_url,
        )
    return _pool


# ---------------------------------------------------------------------------
# Public primitives
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Employee:
    employee_id: str
    full_name: str
    email: str
    region: str
    level: str
    role: str
    manager_id: Optional[str]
    join_date: str
    # phone is deliberately never loaded


def connect(db_path: Optional[str] = None) -> psycopg2.extensions.connection:
    """Borrow a connection from the pool. Caller must call _return(conn).

    db_path is kept for API compatibility but ignored.
    """
    conn = _get_pool().getconn()
    conn.autocommit = False
    return conn


def _return(conn) -> None:
    _get_pool().putconn(conn)


@contextmanager
def transaction() -> Iterator[psycopg2.extras.RealDictCursor]:
    """Yield a DictCursor inside a transaction; commits on exit, rolls back on error."""
    conn = connect()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            yield cur
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        _return(conn)


@contextmanager
def _read_cursor() -> Iterator[psycopg2.extras.RealDictCursor]:
    """Yield a cursor for read-only queries; always rolls back to end the txn."""
    conn = connect()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            yield cur
    finally:
        try:
            conn.rollback()
        finally:
            _return(conn)


# ---------------------------------------------------------------------------
# Schema bootstrap — runs on every startup, safe to re-run (IF NOT EXISTS)
# ---------------------------------------------------------------------------

_SCHEMA_LOCK_ID = 727274  # arbitrary constant for pg_advisory_xact_lock


def ensure_extra_schema(db_path: Optional[str] = None) -> None:
    """Create all tables and indexes, then seed from SQLite if empty.

    db_path is kept for API compatibility but ignored (Postgres uses
    DATABASE_URL).

    Everything runs in one transaction under an advisory lock, so concurrent
    instances starting at the same time cannot race each other.
    """
    with transaction() as cur:
        cur.execute("SELECT pg_advisory_xact_lock(%s)", (_SCHEMA_LOCK_ID,))

        # Core tables (order matters: employees before the tables that FK into it)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS employees (
                employee_id TEXT PRIMARY KEY,
                full_name   TEXT NOT NULL,
                email       TEXT NOT NULL,
                phone       TEXT NOT NULL,
                region      TEXT NOT NULL,
                join_date   TEXT NOT NULL,
                level       TEXT NOT NULL,
                role        TEXT NOT NULL,
                manager_id  TEXT REFERENCES employees(employee_id)
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS leave_balances (
                employee_id  TEXT NOT NULL REFERENCES employees(employee_id),
                leave_type   TEXT NOT NULL,
                balance_days REAL NOT NULL,
                PRIMARY KEY (employee_id, leave_type)
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS leave_requests (
                request_id   TEXT PRIMARY KEY,
                employee_id  TEXT NOT NULL REFERENCES employees(employee_id),
                leave_type   TEXT NOT NULL,
                start_date   TEXT NOT NULL,
                end_date     TEXT NOT NULL,
                working_days INTEGER NOT NULL,
                status       TEXT NOT NULL,
                created_at   TEXT NOT NULL
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS ingested_holidays (
                doc_id       TEXT NOT NULL,
                holiday_date TEXT NOT NULL,
                name         TEXT,
                locator      TEXT,
                optional     INTEGER NOT NULL DEFAULT 0
            )
        """)
        # Indexes (after all tables exist)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_ing_hol_doc
                ON ingested_holidays (doc_id)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_leave_req_emp
                ON leave_requests (employee_id, status)
        """)

        # Pre-embedded chunk vectors (used when VECTOR_BACKEND=pgvector).
        # Created in a separate helper because it needs a mid-transaction commit
        # for the extension DDL.
        _ensure_chunk_embeddings_ddl(cur)

        _seed_if_empty(cur)



def _ensure_chunk_embeddings_ddl(cur) -> None:
    """Create the chunk_embeddings table if it doesn't exist.

    Tries to enable pgvector first.  If the extension is unavailable (e.g. on
    an older Postgres instance), falls back to a BYTEA column so that
    embed_to_pg.py can still write serialised numpy arrays.

    Uses SAVEPOINTs so that a failed CREATE EXTENSION doesn't abort the
    surrounding advisory-lock transaction.
    """
    # Check if already exists to avoid redundant extension attempts
    cur.execute("""
        SELECT 1 FROM information_schema.tables
        WHERE table_schema = current_schema()
          AND table_name = 'chunk_embeddings'
    """)
    if cur.fetchone() is not None:
        return  # table already exists – nothing to do

    # Try pgvector using a savepoint so failure is recoverable
    has_pgvector = False
    try:
        cur.execute("SAVEPOINT sp_pgvector")
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute("RELEASE SAVEPOINT sp_pgvector")
        has_pgvector = True
    except Exception:
        cur.execute("ROLLBACK TO SAVEPOINT sp_pgvector")
        cur.execute("RELEASE SAVEPOINT sp_pgvector")

    if has_pgvector:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS chunk_embeddings (
                chunk_id   TEXT PRIMARY KEY,
                doc_id     TEXT NOT NULL,
                locator    TEXT NOT NULL,
                text       TEXT NOT NULL,
                tokens     INTEGER NOT NULL,
                embedding  vector(384)
            )
        """)
        try:
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_chunk_emb_cosine
                    ON chunk_embeddings USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 10)
            """)
        except Exception:
            pass  # not enough rows yet; embed_to_pg.py recreates it
    else:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS chunk_embeddings (
                chunk_id   TEXT PRIMARY KEY,
                doc_id     TEXT NOT NULL,
                locator    TEXT NOT NULL,
                text       TEXT NOT NULL,
                tokens     INTEGER NOT NULL,
                embedding  BYTEA NOT NULL DEFAULT '',
                embed_dim  INTEGER NOT NULL DEFAULT 0
            )
        """)



def _pg_columns(cur, table: str) -> list[str]:
    cur.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_schema = current_schema() AND table_name = %s "
        "ORDER BY ordinal_position",
        (table,),
    )
    return [r["column_name"] for r in cur.fetchall()]


def _seed_if_empty(cur) -> None:
    """If the employees table is empty, import all data from the baked-in SQLite DB.

    A no-op after the first successful boot.
    """
    import os as _os
    import sqlite3 as _sqlite3

    cur.execute("SELECT 1 FROM employees LIMIT 1")
    if cur.fetchone() is not None:
        return  # already seeded

    sqlite_path = _os.getenv("DB_PATH", "./data/veltrix.db")
    if not _os.path.exists(sqlite_path):
        return  # no SQLite snapshot available

    # Parents before children so FKs are satisfied.
    tables = ["employees", "leave_balances", "leave_requests", "ingested_holidays"]

    sq = _sqlite3.connect(sqlite_path)
    sq.row_factory = _sqlite3.Row
    try:
        for table in tables:
            try:
                rows = sq.execute(f"SELECT * FROM {table}").fetchall()
            except _sqlite3.OperationalError:
                continue  # table missing in SQLite — skip
            if not rows:
                continue

            # Only copy columns that exist on both sides.
            pg_cols = set(_pg_columns(cur, table))
            cols = [c for c in rows[0].keys() if c in pg_cols]
            if not cols:
                continue

            data = [tuple(r[c] for c in cols) for r in rows]

            # employees.manager_id is a self-referencing FK: insert with
            # manager_id NULL first, then fill it in once all rows exist.
            manager_pairs: list[tuple] = []
            if table == "employees" and "manager_id" in cols:
                mi = cols.index("manager_id")
                ei = cols.index("employee_id")
                manager_pairs = [
                    (row[mi], row[ei]) for row in data if row[mi] is not None
                ]
                data = [row[:mi] + (None,) + row[mi + 1:] for row in data]

            col_str = "(" + ", ".join(cols) + ")"
            psycopg2.extras.execute_values(
                cur,
                f"INSERT INTO {table} {col_str} VALUES %s ON CONFLICT DO NOTHING",
                data,
            )

            if manager_pairs:
                psycopg2.extras.execute_batch(
                    cur,
                    "UPDATE employees SET manager_id = %s WHERE employee_id = %s",
                    manager_pairs,
                )
    finally:
        sq.close()


# ---------------------------------------------------------------------------
# Employee helpers
# ---------------------------------------------------------------------------

_EMP_COLS = "employee_id, full_name, email, region, level, role, manager_id, join_date"


def _row_to_emp(r: dict) -> Employee:
    return Employee(**{k: r[k] for k in (
        "employee_id", "full_name", "email", "region",
        "level", "role", "manager_id", "join_date",
    )})


def get_employee(employee_id: str, db_path: Optional[str] = None) -> Optional[Employee]:
    # db_path kept for API compatibility but ignored (Postgres uses DATABASE_URL)
    with _read_cursor() as cur:
        cur.execute(
            f"SELECT {_EMP_COLS} FROM employees WHERE employee_id = %s",
            (employee_id,),
        )
        r = cur.fetchone()
        return _row_to_emp(r) if r else None


def get_direct_reports(manager_id: str, db_path: Optional[str] = None) -> list[Employee]:
    with _read_cursor() as cur:
        cur.execute(
            f"SELECT {_EMP_COLS} FROM employees WHERE manager_id = %s ORDER BY employee_id",
            (manager_id,),
        )
        return [_row_to_emp(r) for r in cur.fetchall()]


def all_employees(db_path: Optional[str] = None) -> list[Employee]:
    with _read_cursor() as cur:
        cur.execute(f"SELECT {_EMP_COLS} FROM employees ORDER BY employee_id")
        return [_row_to_emp(r) for r in cur.fetchall()]


# ---------------------------------------------------------------------------
# Holiday helpers
# ---------------------------------------------------------------------------

def replace_holidays(rows: list[tuple], db_path: Optional[str] = None) -> None:
    with transaction() as cur:
        cur.execute("DELETE FROM ingested_holidays")
        if rows:
            psycopg2.extras.execute_values(
                cur,
                "INSERT INTO ingested_holidays(doc_id, holiday_date, name, locator, optional) VALUES %s",
                rows,
            )


def holidays_for_doc(doc_id: str, db_path: Optional[str] = None) -> list[dict]:
    with _read_cursor() as cur:
        cur.execute(
            "SELECT holiday_date, name, locator, optional FROM ingested_holidays "
            "WHERE doc_id = %s ORDER BY holiday_date",
            (doc_id,),
        )
        return [dict(r) for r in cur.fetchall()]


# ---------------------------------------------------------------------------
# Generic query helpers used by leave_tools (accept %s SQL directly)
# ---------------------------------------------------------------------------

def query(sql: str, params: tuple = ()) -> list[dict]:
    """Run a SELECT and return rows as dicts."""
    with _read_cursor() as cur:
        cur.execute(sql, params)
        return [dict(r) for r in cur.fetchall()]


def execute(sql: str, params: tuple = ()) -> None:
    """Run a DML statement inside a transaction."""
    with transaction() as cur:
        cur.execute(sql, params)