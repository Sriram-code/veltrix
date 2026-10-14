"""Evaluation for the Veltrix assistant.

Retrieval only (no LLM, no cost; run it with the reranker on and off):
    python scripts/run_eval.py --retrieval

End to end against a running server (uses the LLM):
    python scripts/run_eval.py --e2e
    python scripts/run_eval.py --e2e --url http://localhost:8000 --runs 3

Results are printed and saved to eval/report_<mode>.json.
Read-only: the eval set never books or cancels leave.
"""
import argparse
import json
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CASES = ROOT / "eval" / "eval_set.json"


def load_cases(only=None):
    cases = json.loads(CASES.read_text(encoding="utf-8"))
    if only:
        cases = [c for c in cases if c["id"] in only]
    return cases


def contains_fact(answer, facts):
    a = (answer or "").lower()
    return any(f.lower() in a for f in facts)


# ------------------------------------------------------- retrieval
def run_retrieval(cases):
    from app import db
    from app.config import settings
    from app.ingestion.pipeline import ingest
    from app.retrieval.retriever import build_context

    print("Ingesting corpus (embeddings, reranker)...")
    kb = ingest()
    print("Search:", kb.index.describe() if kb.index else "none")
    rows, recalls, rr, tokens, secs = [], [], [], [], []
    for c in cases:
        exp, forbid = c.get("expect_docs"), c.get("forbid_docs", [])
        if not exp and not forbid:
            continue  # tool / refusal cases have no retrieval target
        emp = db.get_employee(c["employee"])
        t0 = time.time()
        ctx = build_context(kb, emp.role, emp.region, settings.today,
                            c["question"])
        secs.append(time.time() - t0)
        docs = [s.doc_id for s in ctx.sources]
        leak = [d for d in forbid if d in docs]
        row = {"id": c["id"], "docs": docs, "leak": leak,
               "tokens": ctx.corpus_tokens}
        if exp:
            pos = next((n for n, d in enumerate(docs, 1) if d in exp),
                       None)
            row["rank"] = pos
            recalls.append(1 if pos else 0)
            rr.append(1 / pos if pos else 0)
        tokens.append(ctx.corpus_tokens)
        rows.append(row)
        mark = "LEAK" if leak else ("ok" if row.get("rank") or not exp
                                    else "MISS")
        print(f"{c['id']:<18} {mark:<5} rank={row.get('rank')} "
              f"tokens={ctx.corpus_tokens} passages={len(docs)}")
    n = len(recalls) or 1
    summary = {
        "mode": "retrieval",
        "search": kb.index.describe() if kb.index else "none",
        "recall_at_selected": round(sum(recalls) / n, 3),
        "mrr": round(sum(rr) / n, 3),
        "leaks": sum(1 for r in rows if r["leak"]),
        "avg_corpus_tokens": round(sum(tokens) / (len(tokens) or 1)),
        "avg_retrieval_ms": round(1000 * sum(secs) / (len(secs) or 1)),
    }
    return summary, rows


# ------------------------------------------------------- end to end
def run_e2e(cases, url, runs, sleep):
    import httpx

    rows = []
    for run in range(1, runs + 1):
        for c in cases:
            sid = "eval-" + uuid.uuid4().hex[:8]
            h = {"X-Employee-Id": c["employee"]}
            t0 = time.time()
            try:
                r = httpx.post(url + "/chat", headers=h, timeout=60,
                               json={"session_id": sid,
                                     "message": c["question"]})
                data = r.json()
            except Exception as e:
                data, r = {"error": str(e)}, None
            ms = int((time.time() - t0) * 1000)
            cited = [x["doc_id"] for x in data.get("citations", [])]
            route = data.get("route")
            checks = {"route": route in c["route"]}
            if c.get("expect_docs"):
                checks["cited"] = any(d in cited for d in c["expect_docs"])
            if c.get("facts_any"):
                checks["fact"] = contains_fact(data.get("answer"),
                                               c["facts_any"])
            checks["no_leak"] = not any(d in cited
                                        for d in c.get("forbid_docs", []))
            tok = None
            if data.get("request_id"):
                try:
                    t = httpx.get(url + "/debug/trace/"
                                  + data["request_id"], headers=h,
                                  timeout=10)
                    if t.status_code == 200:
                        tok = t.json().get("totals")
                except Exception:
                    pass
            ok = all(checks.values())
            rows.append({"run": run, "id": c["id"], "ok": ok,
                         "checks": checks, "route": route,
                         "cited": cited, "ms": ms, "tokens": tok,
                         "answer": (data.get("answer") or
                                    data.get("error", ""))[:300]})
            failed = [k for k, v in checks.items() if not v]
            print(f"run{run} {c['id']:<18} "
                  f"{'PASS' if ok else 'FAIL ' + ','.join(failed):<20} "
                  f"route={route} cited={cited} {ms}ms")
            time.sleep(sleep)

    def rate(key):
        vals = [r["checks"][key] for r in rows if key in r["checks"]]
        return round(sum(vals) / (len(vals) or 1), 3)

    with_tok = [r["tokens"] for r in rows if r["tokens"]]
    summary = {
        "mode": "e2e", "runs": runs, "cases": len(cases),
        "pass_rate": round(sum(r["ok"] for r in rows) / len(rows), 3),
        "route_accuracy": rate("route"),
        "citation_recall": rate("cited"),
        "fact_accuracy": rate("fact"),
        "leaks": sum(1 for r in rows if not r["checks"]["no_leak"]),
        "avg_ms": round(sum(r["ms"] for r in rows) / len(rows)),
        "max_ms": max(r["ms"] for r in rows),
        "avg_input_tokens": round(sum(t["input_tokens"] for t in with_tok)
                                  / (len(with_tok) or 1)),
        "avg_corpus_tokens": round(sum(t["corpus_tokens"]
                                       for t in with_tok)
                                   / (len(with_tok) or 1)),
    }
    return summary, rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--retrieval", action="store_true")
    p.add_argument("--e2e", action="store_true")
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--sleep", type=float, default=1.5,
                   help="seconds between requests (rate limits)")
    p.add_argument("--only", nargs="*", help="case ids to run")
    a = p.parse_args()
    if not (a.retrieval or a.e2e):
        p.error("choose --retrieval or --e2e")
    cases = load_cases(a.only)
    if a.retrieval:
        summary, rows = run_retrieval(cases)
    else:
        summary, rows = run_e2e(cases, a.url.rstrip("/"), a.runs, a.sleep)
    print("\nSUMMARY")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    out = ROOT / "eval" / f"report_{summary['mode']}.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows},
                              indent=2), encoding="utf-8")
    print("Saved", out)
    if summary["leaks"]:
        sys.exit(1)


if __name__ == "__main__":
    main()