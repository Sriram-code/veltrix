"""Working-day calculation (pure) + resolution of the applicable holiday calendar."""
from __future__ import annotations

from datetime import date, timedelta
from typing import Iterable, Optional

from app import db
from app.retrieval import precedence as P


def daterange(start: date, end: date) -> Iterable[date]:
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def count_working_days(start: date, end: date, holidays: set[date]) -> int:
    return sum(1 for d in daterange(start, end) if d.weekday() < 5 and d not in holidays)


def holidays_for_range(docs: list[P.DocMeta], role: str, region: str, start: date, end: date,
                       db_path: Optional[str] = None) -> set[date]:
    """For each year in the range, pick the winning calendar (Section 5) among eligible calendars
    that list dates in that year, and return its non-optional holiday dates."""
    result: set[date] = set()
    for year in range(start.year, end.year + 1):
        ref = min(end, date(year, 12, 31))
        cands, per_doc = [], {}
        for cal in (d for d in docs if d.doc_type == "calendar" and P.is_eligible(d, role, region, ref)):
            days = {date.fromisoformat(h["holiday_date"]) for h in db.holidays_for_doc(cal.doc_id, db_path)
                    if not h["optional"] and h["holiday_date"].startswith(str(year))}
            if days:
                cands.append(cal)
                per_doc[cal.doc_id] = days
        for w in P.winners(cands):  # tie -> union (conservative)
            result |= per_doc[w.doc_id]
    return {d for d in result if start <= d <= end}
