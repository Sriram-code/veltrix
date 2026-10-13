"""Shows which passages the bot sends to the LLM.
Usage:
python scripts/debug_retrieval.py E2004 "How many sick days do I get?"
python scripts/debug_retrieval.py E2004 "sick leave" ADD-LEAVE-IN
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import db  # noqa: E402
from app.config import settings  # noqa: E402
from app.ingestion.pipeline import ingest  # noqa: E402
from app.retrieval.retriever import build_context  # noqa: E402

emp_id = sys.argv[1]
question = sys.argv[2]
emp = db.get_employee(emp_id)
if emp is None:
    sys.exit("unknown employee id")

kb = ingest()
print()
print("role:", emp.role, "| region:", emp.region)

if len(sys.argv) > 3:
    doc_id = sys.argv[3]
    print("All chunks of", doc_id)
    for c in kb.chunks:
        if c.doc_id == doc_id:
            print("--", c.locator)
            print(c.text)
    sys.exit(0)

ref = settings.today
ctx = build_context(kb, emp.role, emp.region, ref, question)
print("corpus tokens:", ctx.corpus_tokens)
print()
print(ctx.precedence_notes)
print()
for s in ctx.sources:
    tag = "UPCOMING" if s.upcoming else "in force"
    print(s.sid, s.doc_id, "|", s.locator, "|", tag)
    print("   ", s.text[:150].replace("\n", " "))