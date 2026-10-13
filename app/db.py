"""Data access layer for veltrix.db. Existing tables are never altered;
we only add the `ingested_holidays` table and indexes."""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Optional

from app.config import settings


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


def connect(db_path: Optional[str] = None) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path or settings.db_path, timeout=10, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def transaction(db_path: Optional[str] = None) -> Iterator[sqlite3.Connection]:
    """BEGIN IMMEDIATE ... COMMIT, rollback on any error."""
    conn = connect(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        yield conn
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()


def ensure_extra_schema(db_path: Optional[str] = None) -> None:
    conn = connect(db_path)
    try:
        conn.execute("""CREATE TABLE IF NOT EXISTS ingested_holidays (
                            doc_id TEXT NOT NULL, holiday_date TEXT NOT NULL, name TEXT,
                            locator TEXT, optional INTEGER NOT NULL DEFAULT 0)""")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ing_hol_doc ON ingested_holidays(doc_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_leave_req_emp ON leave_requests(employee_id, status)")
    finally:
        conn.close()


_EMP_COLS = "employee_id, full_name, email, region, level, role, manager_id, join_date"


def _row_to_emp(r: sqlite3.Row) -> Employee:
    return Employee(**{k: r[k] for k in r.keys()})


def get_employee(employee_id: str, db_path: Optional[str] = None) -> Optional[Employee]:
    conn = connect(db_path)
    try:
        r = conn.execute(f"SELECT {_EMP_COLS} FROM employees WHERE employee_id = ?", (employee_id,)).fetchone()
        return _row_to_emp(r) if r else None
    finally:
        conn.close()


def get_direct_reports(manager_id: str, db_path: Optional[str] = None) -> list[Employee]:
    conn = connect(db_path)
    try:
        rows = conn.execute(f"SELECT {_EMP_COLS} FROM employees WHERE manager_id = ? ORDER BY employee_id",
                            (manager_id,)).fetchall()
        return [_row_to_emp(r) for r in rows]
    finally:
        conn.close()


def all_employees(db_path: Optional[str] = None) -> list[Employee]:
    conn = connect(db_path)
    try:
        return [_row_to_emp(r) for r in conn.execute(f"SELECT {_EMP_COLS} FROM employees ORDER BY employee_id")]
    finally:
        conn.close()


def replace_holidays(rows: list[tuple], db_path: Optional[str] = None) -> None:
    with transaction(db_path) as conn:
        conn.execute("DELETE FROM ingested_holidays")
        conn.executemany("INSERT INTO ingested_holidays(doc_id, holiday_date, name, locator, optional) "
                         "VALUES (?,?,?,?,?)", rows)


def holidays_for_doc(doc_id: str, db_path: Optional[str] = None) -> list[dict]:
    conn = connect(db_path)
    try:
        rows = conn.execute("SELECT holiday_date, name, locator, optional FROM ingested_holidays "
                            "WHERE doc_id = ? ORDER BY holiday_date", (doc_id,)).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()
