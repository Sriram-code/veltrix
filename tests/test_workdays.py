from datetime import date

from app.tools.workdays import count_working_days, holidays_for_range


def test_plain_week():
    assert count_working_days(date(2026, 10, 19), date(2026, 10, 23), set()) == 5


def test_weekend_only_is_zero():
    assert count_working_days(date(2026, 10, 17), date(2026, 10, 18), set()) == 0


def test_range_spanning_weekend():
    assert count_working_days(date(2026, 10, 16), date(2026, 10, 20), set()) == 3  # Fri, Mon, Tue


def test_holiday_excluded_in_range_spanning_weekend():
    assert count_working_days(date(2026, 10, 12), date(2026, 10, 18), {date(2026, 10, 14)}) == 4


def test_holiday_on_weekend_not_double_counted():
    assert count_working_days(date(2026, 10, 12), date(2026, 10, 18), {date(2026, 10, 17)}) == 5


def test_single_day():
    assert count_working_days(date(2026, 10, 13), date(2026, 10, 13), set()) == 1


def test_holidays_come_from_applicable_calendar(kb, db_path):
    docs = list(kb.docs.values())
    h_in = holidays_for_range(docs, "employee", "IN", date(2026, 10, 1), date(2026, 10, 31), db_path)
    assert date(2026, 10, 14) in h_in and date(2026, 10, 2) in h_in
    assert date(2026, 10, 16) not in h_in  # optional holiday is not a day off
    assert holidays_for_range(docs, "employee", "US", date(2026, 10, 1), date(2026, 10, 31), db_path) == set()
