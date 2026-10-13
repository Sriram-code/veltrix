"""Access control for documents (retrieval filter) and leave data (tools), for every role."""
import pytest

from app import db
from app.retrieval.retriever import allowed_chunk_indices, build_context
from app.tools import leave_tools as T
from tests.conftest import TODAY


def ctx(emp_id, kb, db_path):
    return T.ToolContext(db.get_employee(emp_id, db_path), TODAY, list(kb.docs.values()), db_path)


def visible(kb, role, region):
    return {kb.chunks[i].doc_id for i in allowed_chunk_indices(kb, role, region)}


def test_employee_never_gets_restricted_docs(kb):
    assert not {"HR-COMP", "HB-MGR"} & visible(kb, "employee", "IN")


def test_manager_gets_manager_docs_not_hr(kb):
    v = visible(kb, "manager", "IN")
    assert "HB-MGR" in v and "HR-COMP" not in v


def test_hr_gets_everything(kb):
    assert {"HB-MGR", "HR-COMP"} <= visible(kb, "hr", "IN")


def test_region_filter(kb):
    assert not {"ADD-IN", "MEMO"} & visible(kb, "employee", "DE")


from app.ingestion.parsers import _find_tesseract  # noqa: E402


@pytest.mark.skipif(_find_tesseract() is None,
                    reason="Tesseract OCR not installed")
def test_scanned_pdf_was_ocred(kb):
    assert any(c.doc_id == "MEMO" and "office" in c.text.lower() for c in kb.chunks)


def test_restricted_text_never_in_context(kb):
    c = build_context(kb, "employee", "IN", TODAY, "secret salary band L3 compensation bands")
    assert "900000" not in c.rendered and "HR-COMP" not in c.rendered


def test_context_budget(kb):
    assert build_context(kb, "hr", "IN", TODAY, "leave sick annual holiday memo travel",
                         budget=100000).corpus_tokens <= 3000


@pytest.mark.parametrize("requester,target,allowed", [
    ("E1002", "E1002", True), ("E1002", "E1003", False), ("E1002", "E1001", False),
    ("E1001", "E1002", True), ("E1001", "E1004", False), ("E1000", "E1004", True), ("E1002", "E9999", False),
])
def test_leave_data_permissions(kb, db_path, requester, target, allowed):
    c = ctx(requester, kb, db_path)
    for res in (T.get_leave_balance(c, target), T.list_leave_requests(c, target)):
        assert (res.status == "success") if allowed else (res.error == T.NOT_ALLOWED)


def test_same_refusal_for_unknown_and_forbidden(kb, db_path):
    c = ctx("E1002", kb, db_path)
    assert T.get_leave_balance(c, "E9999").error == T.get_leave_balance(c, "E1004").error


def test_team_leave_managers_only(kb, db_path):
    assert T.get_team_leave(ctx("E1002", kb, db_path), "2026-10-19", "2026-10-30").error_kind == "permission"
    assert T.get_team_leave(ctx("E1000", kb, db_path), "2026-10-19", "2026-10-30").error_kind == "permission"
    r = T.get_team_leave(ctx("E1001", kb, db_path), "2026-10-19", "2026-10-30")
    assert r.status == "success" and [x["employee_id"] for x in r.data["leave"]] == ["E1002"]
    assert "+91" not in str(r.data) and "@" not in str(r.data)
