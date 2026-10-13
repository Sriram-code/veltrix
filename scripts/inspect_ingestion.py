"""Debug helper: run ingestion on ./data and print what was extracted.
Usage: docker compose run --rm app python scripts/inspect_ingestion.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import db  # noqa: E402
from app.ingestion.pipeline import ingest  # noqa: E402

kb = ingest(use_dense=False)
print(f"docs={len(kb.docs)} chunks={len(kb.chunks)} holidays={kb.holiday_count} failed={kb.failed}\n")
for d in kb.docs.values():
    cs = [c for c in kb.chunks if c.doc_id == d.doc_id]
    print(f"== {d.doc_id} ({d.file}) chunks={len(cs)} locators={sorted({c.locator for c in cs})[:6]}")
    if cs:
        print("   first chunk:", cs[0].text[:150].replace("\n", " / "))
    if d.doc_type == "calendar":
        for h in db.holidays_for_doc(d.doc_id):
            print("   ", h["holiday_date"], "|", h["name"], "| optional" if h["optional"] else "")
