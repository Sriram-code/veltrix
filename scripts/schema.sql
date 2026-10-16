-- PostgreSQL schema for Veltrix PeopleOps
-- Run this ONCE after provisioning the Render Postgres database,
-- BEFORE running scripts/migrate_sqlite_to_pg.py

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
);

CREATE TABLE IF NOT EXISTS leave_balances (
    employee_id  TEXT NOT NULL REFERENCES employees(employee_id),
    leave_type   TEXT NOT NULL,
    balance_days REAL NOT NULL,
    PRIMARY KEY (employee_id, leave_type)
);

CREATE TABLE IF NOT EXISTS leave_requests (
    request_id   TEXT PRIMARY KEY,
    employee_id  TEXT NOT NULL REFERENCES employees(employee_id),
    leave_type   TEXT NOT NULL,
    start_date   TEXT NOT NULL,
    end_date     TEXT NOT NULL,
    working_days INTEGER NOT NULL,
    status       TEXT NOT NULL,
    created_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ingested_holidays (
    doc_id       TEXT NOT NULL,
    holiday_date TEXT NOT NULL,
    name         TEXT,
    locator      TEXT,
    optional     INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_ing_hol_doc    ON ingested_holidays (doc_id);
CREATE INDEX IF NOT EXISTS idx_leave_req_emp  ON leave_requests (employee_id, status);
