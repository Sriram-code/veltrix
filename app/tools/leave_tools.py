"""The five leave tools. Every permission check runs HERE, using the requester resolved from the
X-Employee-Id header (ToolContext.requester), never from LLM-chosen values."""
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
    db_path: Optional[str] = None


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
    target = db.get_employee(target_id, ctx.db_path)
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
    conn = db.connect(ctx.db_path)
    try:
        q, p = "SELECT leave_type, balance_days FROM leave_balances WHERE employee_id = ?", [employee_id]
        if leave_type:
            q, p = q + " AND leave_type = ?", p + [leave_type]
        rows = [dict(r) for r in conn.execute(q + " ORDER BY leave_type", p)]
    finally:
        conn.close()
    return ToolResult("get_leave_balance", args, "success", {"employee_id": employee_id, "balances": rows})


def list_leave_requests(ctx: ToolContext, employee_id: str, status: Optional[str] = None) -> ToolResult:
    args = {"employee_id": employee_id} | ({"status": status} if status else {})
    if not employee_id or not can_view_employee(ctx, employee_id):
        return _perm("list_leave_requests", args)
    if status and status not in STATUSES:
        return _bad("list_leave_requests", args, "Status must be pending, approved or cancelled.")
    conn = db.connect(ctx.db_path)
    try:
        q = ("SELECT request_id, leave_type, start_date, end_date, working_days, status FROM leave_requests "
             "WHERE employee_id = ?")
        p = [employee_id]
        if status:
            q, p = q + " AND status = ?", p + [status]
        rows = [dict(r) for r in conn.execute(q + " ORDER BY start_date", p)]
    finally:
        conn.close()
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
    reports = db.get_direct_reports(ctx.requester.employee_id, ctx.db_path)
    names = {r.employee_id: r.full_name for r in reports}
    rows = []
    if reports:
        conn = db.connect(ctx.db_path)
        try:
            qs = ",".join("?" * len(names))
            rows = [dict(r) for r in conn.execute(
                f"SELECT request_id, employee_id, leave_type, start_date, end_date, working_days, status "
                f"FROM leave_requests WHERE employee_id IN ({qs}) AND status IN ('pending','approved') "
                f"AND start_date <= ? AND end_date >= ? ORDER BY start_date, employee_id",
                [*names, end_date, start_date])]
        finally:
            conn.close()
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
    conn = db.connect(ctx.db_path)
    try:
        bal = conn.execute("SELECT balance_days FROM leave_balances WHERE employee_id=? AND leave_type=?",
                           (r.employee_id, leave_type)).fetchone()
        ov = conn.execute("SELECT request_id, start_date, end_date, status FROM leave_requests WHERE employee_id=? "
                          "AND status IN ('pending','approved') AND start_date <= ? AND end_date >= ? "
                          "ORDER BY start_date", (r.employee_id, end_date, start_date)).fetchone()
    finally:
        conn.close()
    balance = bal["balance_days"] if bal else 0.0
    if wd > balance:
        return _bad(n, args, f"This needs {wd} working days but your {leave_type} leave balance is {balance:g} days.")
    if ov:
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
    with db.transaction(ctx.db_path) as conn:
        bal = conn.execute("SELECT balance_days FROM leave_balances WHERE employee_id=? AND leave_type=?",
                           (r.employee_id, leave_type)).fetchone()
        ov = conn.execute("SELECT 1 FROM leave_requests WHERE employee_id=? AND status IN ('pending','approved') "
                          "AND start_date <= ? AND end_date >= ?", (r.employee_id, end_date, start_date)).fetchone()
        if not bal or bal["balance_days"] < wd:
            return _bad("apply_leave", args, "Your balance changed and is no longer sufficient.")
        if ov:
            return _bad("apply_leave", args, "It now overlaps an existing request.")
        conn.execute("INSERT INTO leave_requests(request_id, employee_id, leave_type, start_date, end_date, "
                     "working_days, status, created_at) VALUES (?,?,?,?,?,?,'pending',?)",
                     (rid, r.employee_id, leave_type, start_date, end_date, wd,
                      datetime.now(timezone.utc).isoformat(timespec="seconds")))
        conn.execute("UPDATE leave_balances SET balance_days = balance_days - ? WHERE employee_id=? AND leave_type=?",
                     (wd, r.employee_id, leave_type))
        new_bal = bal["balance_days"] - wd
    return ToolResult("apply_leave", args, "success",
                      {"request_id": rid, "working_days": wd, "status": "pending", "new_balance": new_bal})


def validate_cancel(ctx: ToolContext, request_id: str) -> ToolResult:
    args = {"request_id": request_id}
    conn = db.connect(ctx.db_path)
    try:
        row = conn.execute("SELECT * FROM leave_requests WHERE request_id = ?", (request_id,)).fetchone()
    finally:
        conn.close()
    if row is None or row["employee_id"] != ctx.requester.employee_id:
        return _bad("cancel_leave", args, "I couldn't find that request among your leave requests.")
    if row["status"] not in ("pending", "approved"):
        return _bad("cancel_leave", args, f"That request is already {row['status']}.")
    if date.fromisoformat(row["start_date"]) <= ctx.today:
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
    with db.transaction(ctx.db_path) as conn:
        row = conn.execute("SELECT * FROM leave_requests WHERE request_id=? AND employee_id=? "
                           "AND status IN ('pending','approved')", (request_id, ctx.requester.employee_id)).fetchone()
        if row is None:
            return _bad("cancel_leave", args, "That request can no longer be cancelled.")
        conn.execute("UPDATE leave_requests SET status='cancelled' WHERE request_id=?", (request_id,))
        conn.execute("UPDATE leave_balances SET balance_days = balance_days + ? WHERE employee_id=? AND leave_type=?",
                     (row["working_days"], row["employee_id"], row["leave_type"]))
    return ToolResult("cancel_leave", args, "success", {**v.data, "status": "cancelled"})


def cancellable_requests(ctx: ToolContext) -> list[dict]:
    conn = db.connect(ctx.db_path)
    try:
        return [dict(r) for r in conn.execute(
            "SELECT request_id, leave_type, start_date, end_date, working_days, status FROM leave_requests "
            "WHERE employee_id=? AND status IN ('pending','approved') AND start_date > ? ORDER BY start_date",
            (ctx.requester.employee_id, ctx.today.isoformat()))]
    finally:
        conn.close()
