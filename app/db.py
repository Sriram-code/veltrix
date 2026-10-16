"""Data access layer – backed by PostgreSQL (psycopg2).

Connection string is read from the DATABASE_URL environment variable.
All SQL uses %s placeholders (psycopg2 style).
Existing tables are never altered; we only CREATE IF NOT EXISTS the
`ingested_holidays` table and the relevant indexes.
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


def connect() -> psycopg2.extensions.connection:
    """Borrow a connection from the pool. Caller must call pool.putconn(conn)."""
    pool = _get_pool()
    conn = pool.getconn()
    conn.autocommit = False
    return conn


def _return(conn) -> None:
    _get_pool().putconn(conn)


@contextmanager
def transaction() -> Iterator[psycopg2.extensions.cursor]:
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
    """Yield a read-only cursor (no explicit transaction needed)."""
    conn = connect()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            yield cur
    finally:
        _return(conn)


# ---------------------------------------------------------------------------
# Schema bootstrap
# ---------------------------------------------------------------------------

def ensure_extra_schema() -> None:
    """Create tables / indexes that don't come from the seed SQL."""
    with transaction() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS ingested_holidays (
                doc_id        TEXT NOT NULL,
                holiday_date  TEXT NOT NULL,
                name          TEXT,
                locator       TEXT,
                optional      INTEGER NOT NULL DEFAULT 0
            )
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_ing_hol_doc
                ON ingested_holidays (doc_id)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_leave_req_emp
                ON leave_requests (employee_id, status)
        """)


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
# Generic query helper used by leave_tools (accepts %s SQL directly)
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
