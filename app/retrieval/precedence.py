"""Section 5 rules: eligibility and precedence. Pure functions, unit tested."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

LOW_AUTHORITY = {"faq", "wiki"}
ROLE_CAN_READ = {"employee": {"all"}, "manager": {"all", "managers"}, "hr": {"all", "managers", "hr"}}


@dataclass(frozen=True)
class DocMeta:
    doc_id: str
    file: str
    title: str
    doc_type: str
    regions: tuple[str, ...]
    effective_date: date
    access_level: str

    @staticmethod
    def from_manifest(d: dict) -> "DocMeta":
        return DocMeta(d["doc_id"], d["file"], d.get("title", d["doc_id"]), d.get("doc_type", "policy").lower(),
                       tuple(r.upper() for r in d.get("regions", ["GLOBAL"])),
                       date.fromisoformat(d["effective_date"]), d.get("access_level", "all").lower())


def can_read(role: str, access_level: str) -> bool:
    return access_level in ROLE_CAN_READ.get(role, set())


def region_matches(doc: DocMeta, region: str) -> bool:
    return "GLOBAL" in doc.regions or region.upper() in doc.regions


def readable_for(doc: DocMeta, role: str, region: str) -> bool:
    """Access + region: the hard filter applied BEFORE retrieval."""
    return can_read(role, doc.access_level) and region_matches(doc, region)


def is_eligible(doc: DocMeta, role: str, region: str, ref: date) -> bool:
    return readable_for(doc, role, region) and doc.effective_date <= ref


def authority(doc: DocMeta) -> int:
    return 0 if doc.doc_type in LOW_AUTHORITY else 1


def scope(doc: DocMeta) -> int:
    return 0 if set(doc.regions) == {"GLOBAL"} else 1


def precedence_key(doc: DocMeta) -> tuple[int, int, str]:
    """Higher wins: authority, then scope, then recency."""
    return (authority(doc), scope(doc), doc.effective_date.isoformat())


def compare(a: DocMeta, b: DocMeta) -> int:
    """1 if a wins, -1 if b wins, 0 if unresolved."""
    ka, kb = precedence_key(a), precedence_key(b)
    return (ka > kb) - (ka < kb)


def winners(docs: list[DocMeta]) -> list[DocMeta]:
    """All docs sharing the top key (more than one => unresolved conflict)."""
    if not docs:
        return []
    best = max(precedence_key(d) for d in docs)
    return [d for d in docs if precedence_key(d) == best]


def upcoming_that_would_win(current: list[DocMeta], upcoming: list[DocMeta]) -> list[DocMeta]:
    return [u for u in upcoming if all(compare(u, c) > 0 for c in current)]
