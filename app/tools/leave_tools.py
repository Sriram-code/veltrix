"""The five leave tools. Every permission check runs HERE, using the requester resolved from the
X-Employee-Id header (ToolContext.requester), never from LLM-chosen values.

SQL uses %s placeholders (psycopg2 / PostgreSQL style).
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Optional

from app import db
from app.config import LEAVE_YEAR
from app.retrieval.precedence import DocMeta
from app.tools.workdays import count_working_days, holidays_for_range

LEAVE_TYPES = ("annual", "sick")
STATUSES = ("pending", "approved", "cancelled")
NOT_ALLOWED = "I can't share information about that employee."


@dataclass
class ToolContext:
    requester: db.Employee
    today: date
    docs: list[DocMeta]
    db_path: Optional[str] = None  # kept for API compat; ignored (uses DATABASE_URL)


@dataclass
class ToolResult:
    name: str
    args: dict
    status: str  # success | error | pending_confirmation
    data: Any = None
    error: Optional[str] = None
    error_kind: Optional[str] = None  # permission | validation


def _perm(name, args) -> ToolResult:
    return ToolResult(name, args, "error", error=NOT_ALLOWED, error_kind="permission")


def _bad(name, args, msg) -> ToolResult:
    return ToolResult(name, args, "error", error=msg, error_kind="validation")


def can_view_employee(ctx: ToolContext, target_id: str) -> bool:
    """Self; direct manager; HR (one at a time). Unknown IDs -> False (same as forbidden)."""
    r = ctx.requester
    if target_id == r.employee_id:
        return True
    target = db.get_employee(target_id)
    if target is None:
        return False
    return r.role == "hr" or (r.role == "manager" and target.manager_id == r.employee_id)


# ------------------------------------------------------------------ read tools
def get_leave_balance(ctx: ToolContext, employee_id: str, leave_type: Optional[str] = None) -> ToolResult:
    args = {"employee_id": employee_id} | ({"leave_type": leave_type} if leave_type else {})
    if not employee_id or not can_view_employee(ctx, employee_id):
        return _perm("get_leave_balance", args)
    if leave_type and leave_type not in LEAVE_TYPES:
        return _bad("get_leave_balance", args, "Leave type must be 'annual' or 'sick'.")

    sql = "SELECT leave_type, balance_days FROM leave_balances WHERE employee_id = %s"
    params: tuple = (employee_id,)
    if leave_type:
        sql += " AND leave_type = %s"
        params += (leave_type,)
    sql += " ORDER BY leave_type"
    rows = db.query(sql, params)
    return ToolResult("get_leave_balance", args, "success", {"employee_id": employee_id, "balances": rows})


def list_leave_requests(ctx: ToolContext, employee_id: str, status: Optional[str] = None) -> ToolResult:
    args = {"employee_id": employee_id} | ({"status": status} if status else {})
    if not employee_id or not can_view_employee(ctx, employee_id):
        return _perm("list_leave_requests", args)
    if status and status not in STATUSES:
        return _bad("list_leave_requests", args, "Status must be pending, approved or cancelled.")

    sql = ("SELECT request_id, leave_type, start_date, end_date, working_days, status "
           "FROM leave_requests WHERE employee_id = %s")
    params: tuple = (employee_id,)
    if status:
        sql += " AND status = %s"
        params += (status,)
    sql += " ORDER BY start_date"
    rows = db.query(sql, params)
    return ToolResult("list_leave_requests", args, "success", {"employee_id": employee_id, "requests": rows})


def get_team_leave(ctx: ToolContext, start_date: str, end_date: str) -> ToolResult:
    args = {"start_date": start_date, "end_date": end_date}
    if ctx.requester.role != "manager":
        return ToolResult("get_team_leave", args, "error", error="Only managers can view team leave.",
                          error_kind="permission")
    try:
        s, e = date.fromisoformat(start_date), date.fromisoformat(end_date)
    except (TypeError, ValueError):
        return _bad("get_team_leave", args, "Dates must be valid YYYY-MM-DD dates.")
    if e < s:
        return _bad("get_team_leave", args, "The end date is before the start date.")
    reports = db.get_direct_reports(ctx.requester.employee_id)
    names = {r.employee_id: r.full_name for r in reports}
    rows: list[dict] = []
    if reports:
        placeholders = ",".join(["%s"] * len(names))
        rows = db.query(
            f"SELECT request_id, employee_id, leave_type, start_date, end_date, working_days, status "
            f"FROM leave_requests WHERE employee_id IN ({placeholders}) "
            f"AND status IN ('pending','approved') "
            f"AND start_date <= %s AND end_date >= %s ORDER BY start_date, employee_id",
            (*names.keys(), end_date, start_date),
        )
    for r in rows:
        r["full_name"] = names[r["employee_id"]]  # direct reports: manager may see names
    return ToolResult("get_team_leave", args, "success", {"team_size": len(reports), "leave": rows})


# ------------------------------------------------------------------ write tools
def validate_apply(ctx: ToolContext, leave_type: str, start_date: str, end_date: str) -> ToolResult:
    """All Section 6.2 checks. Success -> status 'pending_confirmation' with working_days."""
    args = {"leave_type": leave_type, "start_date": start_date, "end_date": end_date}
    n = "apply_leave"
    if leave_type not in LEAVE_TYPES:
        return _bad(n, args, "Leave type must be 'annual' or 'sick'.")
    try:
        s, e = date.fromisoformat(start_date), date.fromisoformat(end_date)
    except (TypeError, ValueError):
        return _bad(n, args, "Dates must be valid calendar dates (YYYY-MM-DD).")
    if s.year != LEAVE_YEAR or e.year != LEAVE_YEAR:
        return _bad(n, args, f"Leave can only be booked within the {LEAVE_YEAR} calendar year.")
    if s < ctx.today:
        return _bad(n, args, f"The start date {s} is in the past (today is {ctx.today}).")
    if e < s:
        return _bad(n, args, "The end date is before the start date.")
    r = ctx.requester
    hols = holidays_for_range(ctx.docs, r.role, r.region, s, e, ctx.db_path)
    wd = count_working_days(s, e, hols)
    if wd == 0:
        return _bad(n, args, "That range has no working days (only weekends and/or public holidays).")

    bal_rows = db.query(
        "SELECT balance_days FROM leave_balances WHERE employee_id=%s AND leave_type=%s",
        (r.employee_id, leave_type),
    )
    ov_rows = db.query(
        "SELECT request_id, start_date, end_date, status FROM leave_requests "
        "WHERE employee_id=%s AND status IN ('pending','approved') "
        "AND start_date <= %s AND end_date >= %s ORDER BY start_date",
        (r.employee_id, end_date, start_date),
    )

    balance = bal_rows[0]["balance_days"] if bal_rows else 0.0
    if wd > balance:
        return _bad(n, args, f"This needs {wd} working days but your {leave_type} leave balance is {balance:g} days.")
    if ov_rows:
        ov = ov_rows[0]
        return _bad(n, args, f"It overlaps your existing {ov['status']} request {ov['request_id']} "
                             f"({ov['start_date']} to {ov['end_date']}).")
    return ToolResult(n, args, "pending_confirmation", {"working_days": wd, "balance_before": balance,
                                                        "holidays_excluded": sorted(h.isoformat() for h in hols)})


def apply_leave(ctx: ToolContext, leave_type: str, start_date: str, end_date: str) -> ToolResult:
    """Execute after confirmation; re-checks inside the write transaction."""
    args = {"leave_type": leave_type, "start_date": start_date, "end_date": end_date}
    v = validate_apply(ctx, leave_type, start_date, end_date)
    if v.status == "error":
        return v
    wd, rid, r = v.data["working_days"], str(uuid.uuid4()), ctx.requester
    with db.transaction() as cur:
        cur.execute(
            "SELECT balance_days FROM leave_balances WHERE employee_id=%s AND leave_type=%s",
            (r.employee_id, leave_type),
        )
        bal = cur.fetchone()
        cur.execute(
            "SELECT 1 FROM leave_requests WHERE employee_id=%s AND status IN ('pending','approved') "
            "AND start_date <= %s AND end_date >= %s",
            (r.employee_id, end_date, start_date),
        )
        ov = cur.fetchone()
        if not bal or bal["balance_days"] < wd:
            return _bad("apply_leave", args, "Your balance changed and is no longer sufficient.")
        if ov:
            return _bad("apply_leave", args, "It now overlaps an existing request.")
        cur.execute(
            "INSERT INTO leave_requests(request_id, employee_id, leave_type, start_date, end_date, "
            "working_days, status, created_at) VALUES (%s,%s,%s,%s,%s,%s,'pending',%s)",
            (rid, r.employee_id, leave_type, start_date, end_date, wd,
             datetime.now(timezone.utc).isoformat(timespec="seconds")),
        )
        cur.execute(
            "UPDATE leave_balances SET balance_days = balance_days - %s WHERE employee_id=%s AND leave_type=%s",
            (wd, r.employee_id, leave_type),
        )
        new_bal = bal["balance_days"] - wd
    return ToolResult("apply_leave", args, "success",
                      {"request_id": rid, "working_days": wd, "status": "pending", "new_balance": new_bal})


def validate_cancel(ctx: ToolContext, request_id: str) -> ToolResult:
    args = {"request_id": request_id}
    rows = db.query("SELECT * FROM leave_requests WHERE request_id = %s", (request_id,))
    row = rows[0] if rows else None
    if row is None or row["employee_id"] != ctx.requester.employee_id:
        return _bad("cancel_leave", args, "I couldn't find that request among your leave requests.")
    if row["status"] not in ("pending", "approved"):
        return _bad("cancel_leave", args, f"That request is already {row['status']}.")
    if date.fromisoformat(str(row["start_date"])) <= ctx.today:
        return _bad("cancel_leave", args, f"Only requests that start after today can be cancelled; this one starts "
                                          f"on {row['start_date']}.")
    return ToolResult("cancel_leave", args, "pending_confirmation",
                      {k: row[k] for k in ("request_id", "leave_type", "start_date", "end_date", "working_days",
                                            "status")})


def cancel_leave(ctx: ToolContext, request_id: str) -> ToolResult:
    args = {"request_id": request_id}
    v = validate_cancel(ctx, request_id)
    if v.status == "error":
        return v
    with db.transaction() as cur:
        cur.execute(
            "SELECT * FROM leave_requests WHERE request_id=%s AND employee_id=%s "
            "AND status IN ('pending','approved')",
            (request_id, ctx.requester.employee_id),
        )
        row = cur.fetchone()
        if row is None:
            return _bad("cancel_leave", args, "That request can no longer be cancelled.")
        cur.execute("UPDATE leave_requests SET status='cancelled' WHERE request_id=%s", (request_id,))
        cur.execute(
            "UPDATE leave_balances SET balance_days = balance_days + %s WHERE employee_id=%s AND leave_type=%s",
            (row["working_days"], row["employee_id"], row["leave_type"]),
        )
    return ToolResult("cancel_leave", args, "success", {**v.data, "status": "cancelled"})


def cancellable_requests(ctx: ToolContext) -> list[dict]:
    return db.query(
        "SELECT request_id, leave_type, start_date, end_date, working_days, status FROM leave_requests "
        "WHERE employee_id=%s AND status IN ('pending','approved') AND start_date > %s ORDER BY start_date",
        (ctx.requester.employee_id, ctx.today.isoformat()),
    )
