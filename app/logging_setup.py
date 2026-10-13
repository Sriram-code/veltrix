"""JSON-line logging. Callers log employee IDs only, never names/emails/phones.
A redaction pass is applied as a second line of defence."""
from __future__ import annotations

import json
import logging
import os
import re
import sys

_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE = re.compile(r"\+?\d[\d\s().-]{7,}\d")


def redact(text: str) -> str:
    text = _EMAIL.sub("[email]", text)
    return _PHONE.sub(lambda m: "[phone]" if sum(c.isdigit() for c in re.sub(r"\d{4}-\d{2}-\d{2}", "", m.group())) >= 9
                      else m.group(), text)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {"ts": self.formatTime(record), "level": record.levelname, "event": record.getMessage()}
        payload.update(getattr(record, "fields", {}) or {})
        return redact(json.dumps(payload, default=str))


def setup_logging(log_file: str) -> None:
    root = logging.getLogger()
    if getattr(root, "_veltrix", False):
        return
    root.setLevel(logging.INFO)
    for noisy in ("httpx", "huggingface_hub", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    fmt = JsonFormatter()
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(fmt)
    root.addHandler(h)
    try:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        root.addHandler(fh)
    except OSError:
        pass
    root._veltrix = True  # type: ignore[attr-defined]


def log(event: str, **fields) -> None:
    logging.getLogger("veltrix").info(event, extra={"fields": fields})
