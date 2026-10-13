"""Extract holiday dates from calendar documents automatically during ingestion."""
from __future__ import annotations

import re
from datetime import date

from app.ingestion.parsers import Block

MONTHS = {
    "jan": 1, "january": 1, "januar": 1, "feb": 2, "february": 2, "februar": 2, "mar": 3, "march": 3,
    "märz": 3, "maerz": 3, "apr": 4, "april": 4, "may": 5, "mai": 5, "jun": 6, "june": 6, "juni": 6,
    "jul": 7, "july": 7, "juli": 7, "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10, "oktober": 10, "okt": 10, "nov": 11, "november": 11, "dec": 12,
    "december": 12, "dezember": 12, "dez": 12,
}
_MON = "|".join(sorted(MONTHS, key=len, reverse=True))
_ORD = r"(?:st|nd|rd|th)?"
PATTERNS = [
    ("ymd", re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")),
    ("dmy_num", re.compile(r"\b(\d{1,2})[./](\d{1,2})[./](\d{4})\b")),
    ("d_mon", re.compile(rf"\b(\d{{1,2}}){_ORD}\.?\s+({_MON})\.?,?(?:\s+(\d{{4}}))?\b", re.I)),
    ("mon_d", re.compile(rf"\b({_MON})\.?\s+(\d{{1,2}}){_ORD}\b(?:,?\s+(\d{{4}}))?", re.I)),
]
SKIP_WORDS = re.compile(r"\b(effective|issued|updated|published|version|revised|last reviewed|approved on)\b", re.I)
OPTIONAL_WORDS = re.compile(r"\b(optional|restricted|floating)\b", re.I)


def find_dates(text: str, default_year: int, us_style: bool = False) -> list[date]:
    found, taken = [], []
    for kind, rx in PATTERNS:
        for m in rx.finditer(text):
            if any(a < m.end() and m.start() < b for a, b in taken):
                continue
            try:
                if kind == "ymd":
                    d = date(int(m[1]), int(m[2]), int(m[3]))
                elif kind == "dmy_num":
                    a, b, y = int(m[1]), int(m[2]), int(m[3])
                    if us_style and "/" in m.group(0):
                        a, b = b, a
                    d = date(y, b, a)
                elif kind == "d_mon":
                    d = date(int(m[3]) if m[3] else default_year, MONTHS[m[2].lower()], int(m[1]))
                else:
                    d = date(int(m[3]) if m[3] else default_year, MONTHS[m[1].lower()], int(m[2]))
            except (ValueError, KeyError):
                continue
            taken.append((m.start(), m.end()))
            found.append((m.start(), d))
    return [d for _, d in sorted(found)]


_WEEKDAY_ONLY = re.compile(
    r"(mon|tues|wednes|thurs|fri|satur|sun)day"
    r"|mon|tue|tues|wed|thu|thur|thurs|fri|sat|sun",
    re.I,
)


def _strip_dates(text: str) -> str:
    """Holiday name from a row: drop date cells and cells that
    are only a weekday, but keep names like 'Good Friday'."""
    parts = []
    for cell in re.split(r"[;|]", text):
        label = cell.split(":", 1)[0]
        if ":" in cell and len(label) <= 20:
            cell = cell.split(":", 1)[1]
        for _, rx in PATTERNS:
            cell = rx.sub(" ", cell)
        cell = cell.strip(" ,-–")
        if not cell or _WEEKDAY_ONLY.fullmatch(cell):
            continue
        parts.append(cell)
    return re.sub(r"\s+", " ", " ".join(parts)).strip()



def extract_holidays(doc_id: str, blocks: list[Block], default_year: int, us_style: bool) -> list[tuple]:
    """Return rows (doc_id, iso_date, name, locator, optional). Table rows are preferred."""
    rows = [b for b in blocks if b.kind == "table_row"]
    candidates = rows if any(find_dates(b.text, default_year, us_style) for b in rows) else \
        [b for b in blocks if b.kind == "text"]
    out, seen = [], set()
    for b in candidates:
        if SKIP_WORDS.search(b.text):
            continue
        parts = [b.text] if b.kind == "table_row" else re.split(r"(?<=[.;])\s+|\n", b.text)
        for part in parts:
            ds = find_dates(part, default_year, us_style)
            if not ds:
                continue
            optional = int(bool(OPTIONAL_WORDS.search(part) or (b.heading and OPTIONAL_WORDS.search(b.heading))))
            name = _strip_dates(part)[:120] or None
            for d in ds:
                if (d, optional) in seen:
                    continue
                seen.add((d, optional))
                out.append((doc_id, d.isoformat(), name, b.heading or f"p. {b.page}", optional))
    return out
