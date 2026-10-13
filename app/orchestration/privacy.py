"""Personal-data guards for answers and tool_call args (Section 7.4)."""
from __future__ import annotations

import re
from typing import Any

_EMAIL = re.compile(r"\b([A-Za-z0-9])[A-Za-z0-9._%+-]*@([A-Za-z0-9.-]+\.[A-Za-z]{2,})\b")
_PHONE = re.compile(r"(?<![\w-])\+?\(?\d[\d\s().-]{7,}\d(?![\w-])")


def mask_email(text: str) -> str:
    return _EMAIL.sub(lambda m: f"{m.group(1)}***@{m.group(2)}", text)


def _phone_sub(m: re.Match) -> str:
    s = m.group(0)
    if sum(c.isdigit() for c in re.sub(r"\d{4}-\d{2}-\d{2}", "", s)) < 9:  # ISO dates are not phones
        return s
    return "[phone number removed]"


def scrub_text(text: str) -> str:
    return _PHONE.sub(_phone_sub, mask_email(text))


def scrub(obj: Any) -> Any:
    if isinstance(obj, str):
        return scrub_text(obj)
    if isinstance(obj, dict):
        return {k: scrub(v) for k, v in obj.items() if k != "phone"}
    if isinstance(obj, list):
        return [scrub(v) for v in obj]
    return obj
