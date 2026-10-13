from app import db
from app.tools import leave_tools as T
from tests.conftest import TODAY


def ctx(emp_id, kb, db_path):
    return T.ToolContext(db.get_employee(emp_id, db_path), TODAY, list(kb.docs.values()), db_path)


def balance(db_path, emp, lt):
    c = db.connect(db_path)
    v = c.execute("SELECT balance_days FROM leave_balances WHERE employee_id=? AND leave_type=?", (emp, lt)).fetchone()
    c.close()
    return v[0]


def test_validate_excludes_holiday(kb, db_path):
    v = T.validate_apply(ctx("E1002", kb, db_path), "annual", "2026-10-13", "2026-10-15")
    assert v.status == "pending_confirmation" and v.data["working_days"] == 2


def test_invalid_type(kb, db_path):
    assert T.validate_apply(ctx("E1002", kb, db_path), "casual", "2026-11-02", "2026-11-02").status == "error"


def test_past_start(kb, db_path):
    assert "past" in T.validate_apply(ctx("E1002", kb, db_path), "annual", "2026-10-09", "2026-10-13").error


def test_outside_2026(kb, db_path):
    assert "2026" in T.validate_apply(ctx("E1002", kb, db_path), "annual", "2026-12-31", "2027-01-04").error


def test_end_before_start(kb, db_path):
    assert T.validate_apply(ctx("E1002", kb, db_path), "annual", "2026-11-05", "2026-11-02").status == "error"


def test_zero_working_days(kb, db_path):
    assert "no working days" in T.validate_apply(ctx("E1002", kb, db_path), "annual", "2026-10-17",
                                                 "2026-10-18").error


def test_exceeds_balance(kb, db_path):
    assert "balance" in T.validate_apply(ctx("E1002", kb, db_path), "sick", "2026-11-02", "2026-11-13").error


def test_overlap(kb, db_path):
    assert "overlaps" in T.validate_apply(ctx("E1002", kb, db_path), "annual", "2026-10-27", "2026-10-28").error


def test_apply_transaction(kb, db_path):
    r = T.apply_leave(ctx("E1002", kb, db_path), "annual", "2026-11-02", "2026-11-04")
    assert r.status == "success" and r.data["working_days"] == 3
    assert balance(db_path, "E1002", "annual") == 7
    c = db.connect(db_path)
    row = c.execute("SELECT * FROM leave_requests WHERE request_id=?", (r.data["request_id"],)).fetchone()
    c.close()
    assert row["status"] == "pending" and row["working_days"] == 3 and len(row["request_id"]) == 36


def test_cancel_restores_balance(kb, db_path):
    assert T.cancel_leave(ctx("E1002", kb, db_path), "R-OLD").status == "success"
    assert balance(db_path, "E1002", "annual") == 12


def test_cannot_cancel_started_others_or_unknown(kb, db_path):
    assert T.cancel_leave(ctx("E1002", kb, db_path), "R-PAST").status == "error"
    assert T.cancel_leave(ctx("E1003", kb, db_path), "R-OLD").status == "error"
    assert T.cancel_leave(ctx("E1002", kb, db_path), "nope").status == "error"


def test_cannot_cancel_twice(kb, db_path):
    c = ctx("E1002", kb, db_path)
    assert T.cancel_leave(c, "R-OLD").status == "success"
    assert "already cancelled" in T.cancel_leave(c, "R-OLD").error
