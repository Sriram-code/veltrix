"""Deterministic date helpers (Section 8.4). Weeks start on Monday."""
from __future__ import annotations

import re
from datetime import date, timedelta

WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


def next_week(today: date) -> tuple[date, date]:
    monday = today - timedelta(days=today.weekday()) + timedelta(days=7)
    return monday, monday + timedelta(days=4)


def this_week(today: date) -> tuple[date, date]:
    monday = today - timedelta(days=today.weekday())
    return monday, monday + timedelta(days=4)


def next_weekday(today: date, weekday: int) -> date:
    """'Next Friday' = the first Friday after today."""
    return today + timedelta(days=((weekday - today.weekday()) % 7) or 7)


def date_cheatsheet(today: date) -> str:
    nw, tw = next_week(today), this_week(today)
    lines = [f"today = {today} ({WEEKDAYS[today.weekday()].title()})", f"tomorrow = {today + timedelta(days=1)}",
             f"this week (Mon-Fri) = {tw[0]} to {tw[1]}", f"next week (Mon-Fri) = {nw[0]} to {nw[1]}"]
    lines += [f"next {wd} = {next_weekday(today, i)}" for i, wd in enumerate(WEEKDAYS)]
    return "\n".join(lines)


def deterministic_range(message: str, today: date) -> tuple[date, date] | None:
    """If the ONLY date phrase is 'next week', return Mon-Fri of next week."""
    if re.search(r"\bnext week\b", message, re.I) and not re.search(
            r"\d|monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
            r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)", message, re.I):
        return next_week(today)
    return None
