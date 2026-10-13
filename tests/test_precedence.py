from datetime import date

from app.retrieval import precedence as P


def doc(i, t, regions, eff, acc="all"):
    return P.DocMeta(i, i, i, t, tuple(regions), date.fromisoformat(eff), acc)


V1 = doc("v1", "policy", ["GLOBAL"], "2025-01-01")
V2 = doc("v2", "policy", ["GLOBAL"], "2026-04-01")
FAQ = doc("faq", "faq", ["GLOBAL"], "2026-05-01")
DE = doc("de", "addendum", ["DE"], "2025-06-01")
V3 = doc("v3", "policy", ["GLOBAL"], "2027-01-01")
ALL = [V1, V2, FAQ, DE, V3]
TODAY = date(2026, 9, 15)


def eligible(region, ref):
    return [d for d in ALL if P.is_eligible(d, "employee", region, ref)]


def test_worked_example_india():
    el = eligible("IN", TODAY)
    assert {d.doc_id for d in el} == {"v1", "v2", "faq"}
    assert [d.doc_id for d in P.winners(el)] == ["v2"]
    assert [d.doc_id for d in P.upcoming_that_would_win(P.winners(el), [V3])] == ["v3"]


def test_worked_example_germany():
    el = eligible("DE", TODAY)
    assert [d.doc_id for d in P.winners(el)] == ["de"]
    assert P.upcoming_that_would_win(P.winners(el), [V3]) == []


def test_worked_example_past_date():
    assert [d.doc_id for d in eligible("IN", date(2025, 8, 1))] == ["v1"]


def test_authority_beats_recency():
    assert P.compare(V2, FAQ) == 1


def test_scope_beats_recency():
    assert P.compare(DE, V2) == 1


def test_unresolved_conflict_returns_both():
    a = doc("a", "policy", ["GLOBAL"], "2026-01-01")
    b = doc("b", "memo", ["GLOBAL"], "2026-01-01")
    assert {d.doc_id for d in P.winners([a, b])} == {"a", "b"}


def test_access_levels():
    hr = doc("hr", "policy", ["GLOBAL"], "2025-01-01", "hr")
    mg = doc("mg", "handbook", ["GLOBAL"], "2025-01-01", "managers")
    assert not P.is_eligible(hr, "employee", "IN", TODAY)
    assert not P.is_eligible(hr, "manager", "IN", TODAY)
    assert P.is_eligible(hr, "hr", "IN", TODAY)
    assert not P.is_eligible(mg, "employee", "IN", TODAY)
    assert P.is_eligible(mg, "manager", "IN", TODAY)


def test_multi_region_doc():
    m = doc("m", "memo", ["IN", "US"], "2026-07-01")
    assert P.is_eligible(m, "employee", "US", TODAY)
    assert not P.is_eligible(m, "employee", "DE", TODAY)
