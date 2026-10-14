"""Live trace of one /chat request, for the monitor panel.

In memory only (last 100 requests). Nothing here changes the
/chat or /health contract. Disable with TRACE_ENABLED=0.
"""
from __future__ import annotations

import os
import threading
import time
from collections import deque

ENABLED = os.getenv("TRACE_ENABLED", "1") != "0"
_store: deque = deque(maxlen=100)
_lock = threading.Lock()
_local = threading.local()


def start(request_id, employee_id, role, region, message):
    if not ENABLED:
        return
    t = {
        "request_id": request_id,
        "employee_id": employee_id,
        "role": role,
        "region": region,
        "message": message[:500],
        "started": time.time(),
        "steps": [],
        "llm_calls": [],
        "route": None,
        "total_ms": None,
    }
    _local.trace = t
    with _lock:
        _store.append(t)


def current():
    return getattr(_local, "trace", None)


def step(name, **data):
    """Add one pipeline step (plan, retrieval, tool, ...)."""
    t = current()
    if t is None:
        return
    ms = int((time.time() - t["started"]) * 1000)
    t["steps"].append({"name": name, "at_ms": ms, **data})


def llm_call(**data):
    t = current()
    if t is None:
        return
    t["llm_calls"].append(data)


def finish(route):
    t = current()
    if t is None:
        return
    t["route"] = route
    t["total_ms"] = int((time.time() - t["started"]) * 1000)
    calls = t["llm_calls"]
    t["totals"] = {
        "llm_calls": len(calls),
        "corpus_tokens": sum(c.get("corpus_tokens", 0) for c in calls),
        "input_tokens": sum(c.get("input_tokens", 0) for c in calls),
        "output_tokens": sum(c.get("output_tokens", 0) for c in calls),
    }
    _local.trace = None


def get(request_id, employee_id):
    """Only the employee who sent the request can see its trace."""
    with _lock:
        for t in reversed(_store):
            if t["request_id"] == request_id:
                if t["employee_id"] == employee_id:
                    return t
                return None
    return None


def recent(employee_id, limit=20):
    with _lock:
        mine = [t for t in _store if t["employee_id"] == employee_id]
    return list(reversed(mine))[:limit]