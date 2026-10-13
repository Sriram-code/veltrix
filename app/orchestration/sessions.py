"""In-memory session store: owner, short history, at most one pending proposal."""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Proposal:
    tool: str  # apply_leave | cancel_leave
    args: dict


@dataclass
class Session:
    owner: str
    history: list[tuple[str, str]] = field(default_factory=list)
    pending: Optional[Proposal] = None
    lock: threading.Lock = field(default_factory=threading.Lock)


class SessionStore:
    def __init__(self):
        self._s: dict[str, Session] = {}
        self._lock = threading.Lock()

    def get_or_create(self, session_id: str, employee_id: str) -> Optional[Session]:
        """None if the session belongs to another employee (-> HTTP 403)."""
        with self._lock:
            s = self._s.setdefault(session_id, Session(owner=employee_id))
            return s if s.owner == employee_id else None
