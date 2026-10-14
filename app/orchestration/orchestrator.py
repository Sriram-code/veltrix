"""Turn handling: guards -> exact cache -> confirmation -> planning -> tools -> retrieval -> answer -> route."""
from __future__ import annotations

import json
import hashlib
import re
import uuid
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

from app import db, trace
from app import cache as C
from app.ingestion.pipeline import KnowledgeBase
from app.llm.prompts import ANSWER_SYSTEM, PLANNER_SYSTEM
from app.logging_setup import log
from app.orchestration import dates as D
from app.orchestration.privacy import scrub, scrub_text
from app.orchestration.sessions import Proposal, Session
from app.retrieval import precedence as P
from app.retrieval.retriever import build_context
from app.tools import leave_tools as T
from app.tools.workdays import holidays_for_range

REFUSE_OFF_TOPIC = ("Sorry, I can't help with that. I can answer questions about Veltrix HR policies and public "
                    "holidays, and help with your leave: checking balances and requests, applying and cancelling.")
REFUSE_MANIPULATION = ("I can't help with that request. I can answer questions about Veltrix HR policies and "
                       "holidays, and help with leave you are allowed to see.")
REFUSE_BULK = ("I can't share data about many employees at once. I can help with one employee you're allowed to "
               "see at a time.")
NO_ANSWER = ("The HR documents available to me don't cover this question, so I can't answer it reliably. "
             "Please contact the HR team for help.")

_INJECTION = re.compile(
    r"(ignore|disregard|forget|override)\s+(all\s+|any\s+|your\s+|the\s+|previous\s+|prior\s+|above\s+)*"
    r"(instructions|rules|prompt|guidelines|restrictions)|system\s+prompt|developer\s+mode|jailbreak|"
    r"reveal\s+(your|the)\s+(prompt|instructions|rules)|(print|show|repeat)\s+(your|the)\s+(system|hidden|initial)\s+"
    r"(prompt|instructions|message)|you\s+are\s+now\s+(a|an|in)\b|act\s+as\s+(an?\s+)?(admin|hr|root)", re.I)
_CONFIRM = re.compile(
    r"^\s*(yes|y|yeah|yep|yup|sure|ok|okay|confirm|confirmed|i confirm|go ahead|proceed|do it|please do|"
    r"yes please|yes,? (go ahead|confirm|please|do it|proceed)|sounds good|that'?s (right|correct))[\s.!]*$", re.I)
_HOLIDAY_WORDS = re.compile(r"holiday|bank holiday|festival|office closed", re.I)
_EMP_ID = re.compile(r"^[A-Za-z]{1,3}\d{2,}$")
_SOURCE_REF = re.compile(
    r"\s*[\(\[]\s*[SH]\d+(?:\s*,\s*[SH]\d+)*\s*[\)\]]"
)
# Follow-up style messages depend on conversation history, so they must never be cached.
_FOLLOWUP = re.compile(r"^\s*(and|also|what about|how about|why|then|so)\b", re.I)

# Messages that look like they need personal data or an action: before the planner,
# only an exact cache match is allowed for these.
_PERSONAL = re.compile(
    r"\b(balance|remaining|left|requests?|apply|applied|cancel\w*|team|employee|reports?|"
    r"pending|approved|booked)\b"
    r"|\bmy\s+(leave|leaves|days|manager|team|requests?|balance)\b"
    r"|\bhow many\b.*\b(have|taken|used)\b", re.I)

def _cacheable(message: str) -> bool:
    """Short or follow-up messages depend on history -> skip the cache (lookup and store)."""
    return len(message.split()) >= 4 and not _FOLLOWUP.match(message)


@dataclass
class Turn:
    answer: str = ""
    route: str = "rag"
    citations: list[dict] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)
    pending_confirmation: bool = False


def _lt(v) -> Optional[str]:
    v = (v or "").lower().strip()
    return v if v in T.LEAVE_TYPES else None


class Orchestrator:
    def __init__(self, kb: KnowledgeBase, llm, today_fn, db_path: Optional[str] = None):
        self.kb, self.llm, self.today_fn, self.db_path = kb, llm, today_fn, db_path
        self.cache = C.AnswerCache(embed_fn=self._make_embed_fn())
        models = []
        for p in getattr(llm, "chain", None) or []:
            models.append(p + ":" + llm.model_for(p))
        self.version = hashlib.sha256(
            (PLANNER_SYSTEM + ANSWER_SYSTEM + ",".join(models)).encode()
        ).hexdigest()[:12]

    def _make_embed_fn(self):
        """Reuse the local bge-small model (if loaded) for meaning match."""
        idx = getattr(self.kb, "index", None)
        emb = getattr(idx, "embedder", None)
        if emb is None:
            return None

        def embed(text):
            vec = next(iter(emb.query_embed([text])))
            return [float(x) for x in vec]
        return embed

    # ------------------------------------------------------------------ cache
    def _cache_partition(self, ctx):
        """Partition depends only on what is known BEFORE the planner runs."""
        r = ctx.requester
        return C.make_partition(
            role=r.role, region=r.region, today=ctx.today, ref_date=ctx.today,
            holiday=False,
            corpus_hash=getattr(self.kb, "corpus_hash", ""),
            version=self.version,
        )

    def _try_cache(self, part, message, rid, pre_planner=False) -> Optional[Turn]:
        personal = bool(_PERSONAL.search(message))
        hit, info = self.cache.lookup(
            part, message,
            exact_only=pre_planner and personal,
            min_meaning=C.PREPLAN_SEMANTIC_MIN if pre_planner else None,
            count_miss=not pre_planner)
        if hit is None:
            if not pre_planner:
                trace.step("cache", detail="MISS: answer will be generated and stored")
            return None
        note = "HIT (" + info["match"] + " match"
        if info.get("score") is not None:
            note += ", score " + str(info["score"])
        note += ")"
        if info.get("cached_question"):
            note += ' - reused answer to: "' + info["cached_question"] + '"'
        note += (". Planner, retrieval and answer LLM calls skipped (0 LLM calls)." if pre_planner
                 else ". Retrieval and the answer LLM call were skipped (0 corpus tokens).")
        if info["match"] in ("spelling", "meaning"):
            self.cache.store(part, message, {
                "answer": hit["answer"],
                "citations": [dict(c) for c in hit["citations"]]})
            note += " This wording was saved as an alias."
        trace.step("cache", detail=note)
        log("cache_hit", request_id=rid, match=info["match"])
        return Turn(hit["answer"], "rag", citations=[dict(c) for c in hit["citations"]])

    # ------------------------------------------------------------------ entry
    def handle(self, session: Session, requester: db.Employee, message: str) -> dict:
        rid = str(uuid.uuid4())
        trace.start(rid, requester.employee_id, requester.role,
                    requester.region, message)
        ctx = T.ToolContext(requester, self.today_fn(), list(self.kb.docs.values()), self.db_path)
        with session.lock:
            turn = self._finalise(self._turn(session, ctx, message, rid))
            session.history = (session.history + [(message[:500], turn.answer[:500])])[-4:]
        trace.step("response", citations=turn.citations,
                   answer=turn.answer[:300])
        trace.finish(turn.route)
        log("chat_turn", request_id=rid, employee_id=requester.employee_id, route=turn.route,
            tools=[t["name"] + ":" + t["status"] for t in turn.tool_calls], citations=len(turn.citations))
        return {"request_id": rid, "answer": turn.answer, "route": turn.route, "citations": turn.citations,
                "tool_calls": turn.tool_calls, "pending_confirmation": turn.pending_confirmation}

    # ------------------------------------------------------------------ main logic
    def _turn(self, session: Session, ctx: T.ToolContext, message: str, rid: str) -> Turn:
        if _INJECTION.search(message):
            trace.step("guard", detail="Blocked by manipulation "
                       "check before any LLM call")
            session.pending = None
            return Turn(REFUSE_MANIPULATION, "refuse")
        if session.pending and _CONFIRM.match(message):  # step 2 of a write
            prop, session.pending = session.pending, None
            return self._execute(ctx, prop)

        # Stage 1: exact-match cache BEFORE the planner (saves the first LLM call).
        part = self._cache_partition(ctx)
        use_cache = _cacheable(message)
        if use_cache and session.pending is None:
            hit = self._try_cache(part, message, rid, pre_planner=True)
            if hit:
                return hit

        plan = self._plan(session, ctx, message, rid)
        session.pending = None  # any other message discards the proposal
        cat = plan.get("category", "hr")
        if cat == "manipulation":
            return Turn(REFUSE_MANIPULATION, "refuse")
        if cat == "bulk_request":
            return Turn(REFUSE_BULK, "refuse")
        if cat == "off_topic":
            return Turn(REFUSE_OFF_TOPIC, "refuse")

        actions = [a for a in plan.get("actions") or [] if isinstance(a, dict) and a.get("tool")]
        write = next((a for a in actions if a["tool"] in ("apply_leave", "cancel_leave")), None)
        if write:
            return self._handle_write(session, ctx, write)
        if plan.get("clarify_question") and not actions and not plan.get("needs_documents"):
            return Turn(plan["clarify_question"], "clarify")

        results: list[T.ToolResult] = []
        for a in actions:
            r = self._run_read(ctx, a)
            if isinstance(r, Turn):
                return r
            results.append(r)
        tool_calls = [self._tc(r) for r in results]
        if not plan.get("needs_documents"):
            if results:
                return Turn(self._render_reads(ctx, results), "tool", tool_calls=tool_calls)
            if plan.get("clarify_question"):
                return Turn(plan["clarify_question"], "clarify")
            return Turn(REFUSE_OFF_TOPIC, "refuse")

        ref = self._reference_date(plan, ctx.today)

        # Stage 2: full matching (typo + meaning) now that the planner confirmed
        # this is a document-only question. Only document-only answers are cached.
        if use_cache and not results:
            hit = self._try_cache(part, message, rid)
            if hit:
                return hit

        status, answer, cites = self._answer_from_docs(
            ctx, ref, (plan.get("search_query") or "") + " " + message,
            message, results, plan, rid)
        if status == "no_answer" or not cites:
            if results:
                return Turn(self._render_reads(ctx, results)
                            + "\n\nThe HR documents don't cover the rest "
                            "of your question; please contact the HR "
                            "team.", "tool", tool_calls=tool_calls)
            return Turn(NO_ANSWER, "no_answer")
        if use_cache and not results and status in ("answered", "partial"):
            self.cache.store(part, message,
                             {"answer": answer,
                              "citations": [dict(c) for c in cites]})
        return Turn(answer, "hybrid" if results else "rag",
                    citations=cites, tool_calls=tool_calls)

    # ------------------------------------------------------------------ planning
    def _plan(self, session: Session, ctx: T.ToolContext, message: str, rid: str) -> dict:
        r = ctx.requester
        pending = json.dumps({"tool": session.pending.tool, "args": session.pending.args}) if session.pending else "none"
        hist = "\n".join(f"USER: {u}\nASSISTANT: {a}" for u, a in session.history[-3:]) or "(none)"
        user = (f"Requester: employee_id={r.employee_id}, role={r.role}, region={r.region}\n"
                f"Date cheat sheet:\n{D.date_cheatsheet(ctx.today)}\n\nPending proposal: {pending}\n\n"
                f"Recent conversation:\n{hist}\n\nLatest message:\n{message}")
        try:
            plan = self.llm.complete_json(PLANNER_SYSTEM, user, purpose="plan", corpus_tokens=0, request_id=rid)
            if not isinstance(plan, dict):
                raise ValueError("plan is not an object")
        except Exception as e:
            log("planner_failed_using_keyword_fallback", request_id=rid, err=type(e).__name__)
            plan = fallback_plan(message)
        nw = D.deterministic_range(message, ctx.today)  # Section 8.4 "next week"
        if nw:
            for a in plan.get("actions") or []:
                if isinstance(a, dict) and a.get("tool") in ("apply_leave", "get_team_leave"):
                    a["start_date"], a["end_date"], a["date_problem"] = nw[0].isoformat(), nw[1].isoformat(), None
        # FIX: runs once, after the loop (it was inside the loop before)
        trace.step(
            "plan",
            category=plan.get("category", "hr"),
            needs_documents=bool(plan.get("needs_documents")),
            actions=[a.get("tool") for a in plan.get("actions") or []
                     if isinstance(a, dict)],
            search_query=plan.get("search_query") or "",
        )
        return plan

    @staticmethod
    def _reference_date(plan: dict, today: date) -> date:
        try:
            ref = date.fromisoformat(plan.get("reference_date") or "")
            return ref if ref < today else today
        except (ValueError, TypeError):
            return today

    # ------------------------------------------------------------------ read tools
    def _resolve_employee(self, ctx: T.ToolContext, who: Optional[str]) -> str | Turn:
        r = ctx.requester
        if not who or who.strip().lower() in ("self", "me", "my", "myself", "i", r.employee_id.lower()):
            return r.employee_id
        who = who.strip()
        if _EMP_ID.match(who):
            return who.upper()  # permission checked inside the tool
        # names: search only employees the requester may see (never reveals others' existence)
        if r.role == "hr":
            visible = db.all_employees(ctx.db_path)
        elif r.role == "manager":
            visible = [r] + db.get_direct_reports(r.employee_id, ctx.db_path)
        else:
            visible = [r]
        q = re.sub(r"'s$", "", who.lower()).strip()
        matches = [e for e in visible if e.full_name.lower() == q] or \
                  [e for e in visible if set(q.split()) <= set(e.full_name.lower().split())]
        if len(matches) == 1:
            return matches[0].employee_id
        if len(matches) > 1:
            return Turn("Which employee do you mean: " +
                        ", ".join(f"{e.full_name} ({e.employee_id})" for e in matches[:5]) + "?", "clarify")
        return Turn(T.NOT_ALLOWED, "refuse")

    def _run_read(self, ctx: T.ToolContext, a: dict) -> T.ToolResult | Turn:
        tool = a["tool"]
        if tool in ("get_leave_balance", "list_leave_requests"):
            emp = self._resolve_employee(ctx, a.get("employee"))
            if isinstance(emp, Turn):
                return emp
            res = (T.get_leave_balance(ctx, emp, _lt(a.get("leave_type"))) if tool == "get_leave_balance" else
                   T.list_leave_requests(ctx, emp, a.get("status") if a.get("status") in T.STATUSES else None))
        elif tool == "get_team_leave":
            if ctx.requester.role != "manager":
                return Turn("Team leave information is only available to managers for their direct reports.", "refuse")
            if not a.get("start_date") or not a.get("end_date"):
                return Turn("Which dates should I check your team's leave for?", "clarify")
            res = T.get_team_leave(ctx, a["start_date"], a["end_date"])
        else:
            return Turn(REFUSE_OFF_TOPIC, "refuse")
        if res.status == "error" and res.error_kind == "permission":
            return Turn(res.error or T.NOT_ALLOWED, "refuse")
        if res.status == "error":
            return Turn(f"I couldn't do that: {res.error}", "tool", tool_calls=[self._tc(res)])
        return res

    # ------------------------------------------------------------------ write tools (two-step)
    def _handle_write(self, session: Session, ctx: T.ToolContext, a: dict) -> Turn:
        if a["tool"] == "apply_leave":
            lt, s, e = _lt(a.get("leave_type")), a.get("start_date"), a.get("end_date")
            missing = []
            if not lt:
                missing.append("whether this is annual or sick leave")
            if a.get("date_problem") == "past":
                missing.append(f"new dates (the date you mentioned has already passed; today is {ctx.today})")
            elif a.get("date_problem") == "ambiguous" or not s:
                missing.append("the exact start and end dates")
            if missing:
                return Turn("To set up your leave request, please tell me " + " and ".join(missing) + ".", "clarify")
            e = e or s
            try:
                sd = date.fromisoformat(s)
            except (TypeError, ValueError):
                return Turn("Please give the dates as calendar dates, e.g. 12 November.", "clarify")
            if sd < ctx.today and sd.year == ctx.today.year:
                return Turn(f"{s} has already passed (today is {ctx.today}). Which dates did you mean?", "clarify")
            v = T.validate_apply(ctx, lt, s, e)
            if v.status == "error":
                return Turn(f"I can't submit that leave request: {v.error}", "tool", tool_calls=[self._tc(v)])
            session.pending = Proposal("apply_leave", v.args)
            hol = v.data["holidays_excluded"]
            ans = (f"Please confirm this leave request:\n- Type: {lt}\n- From {s} to {e}\n"
                   f"- Working days: {v.data['working_days']}"
                   + (f" (public holidays excluded: {', '.join(hol)})" if hol else "") +
                   f"\n- Balance: {v.data['balance_before']:g} → "
                   f"{v.data['balance_before'] - v.data['working_days']:g} days\n"
                   "Reply \"yes\" to submit it, or tell me what to change.")
            return Turn(ans, "tool", tool_calls=[self._tc(v)], pending_confirmation=True)

        rid = a.get("request_id")
        if not rid:
            cands = T.cancellable_requests(ctx)
            if a.get("date_hint"):
                cands = [c for c in cands if c["start_date"] <= a["date_hint"] <= c["end_date"]]
            if _lt(a.get("leave_type")):
                cands = [c for c in cands if c["leave_type"] == _lt(a.get("leave_type"))]
            if not cands:
                lr = T.list_leave_requests(ctx, ctx.requester.employee_id)
                return Turn("I couldn't find a matching leave request of yours that can still be cancelled (only "
                            "pending or approved requests starting after today can be cancelled).", "tool",
                            tool_calls=[self._tc(lr)])
            if len(cands) > 1:
                return Turn("You have several requests that can be cancelled. Which one?\n" + "\n".join(
                    f"- {c['request_id']}: {c['leave_type']} {c['start_date']} to {c['end_date']} ({c['status']})"
                    for c in cands), "clarify")
            rid = cands[0]["request_id"]
        v = T.validate_cancel(ctx, rid)
        if v.status == "error":
            return Turn(f"I can't cancel that: {v.error}", "tool", tool_calls=[self._tc(v)])
        session.pending = Proposal("cancel_leave", v.args)
        d = v.data
        return Turn(f"Please confirm you want to cancel this request:\n- {d['request_id']}: {d['leave_type']} leave "
                    f"{d['start_date']} to {d['end_date']} ({d['working_days']} working days, {d['status']})\n"
                    f"{d['working_days']} days will be returned to your {d['leave_type']} balance. Reply \"yes\" to "
                    "confirm.", "tool", tool_calls=[self._tc(v)], pending_confirmation=True)

    def _execute(self, ctx: T.ToolContext, prop: Proposal) -> Turn:
        if prop.tool == "apply_leave":
            r = T.apply_leave(ctx, **prop.args)
            msg = (f"Done. Your {prop.args['leave_type']} leave from {prop.args['start_date']} to "
                   f"{prop.args['end_date']} ({r.data['working_days']} working days) has been submitted with status "
                   f"pending. Request ID: {r.data['request_id']}. Remaining {prop.args['leave_type']} balance: "
                   f"{r.data['new_balance']:g} days." if r.status == "success"
                   else f"I couldn't submit the request: {r.error}")
        else:
            r = T.cancel_leave(ctx, **prop.args)
            msg = (f"Done. Request {prop.args['request_id']} is cancelled and {r.data['working_days']} days were "
                   f"returned to your {r.data['leave_type']} balance." if r.status == "success"
                   else f"I couldn't cancel the request: {r.error}")
        return Turn(msg, "tool", tool_calls=[self._tc(r)])

    # ------------------------------------------------------------------ documents
    def _answer_from_docs(self, ctx, ref: date, query: str, message: str, results, plan, rid):
        r = ctx.requester
        c = build_context(self.kb, r.role, r.region, ref, query)
        holiday_block, hmap = "", {}
        if plan.get("holiday_lookup") or _HOLIDAY_WORDS.search(message):
            holiday_block, hmap = self._holiday_records(ctx, ref)
        if not c.sources and not hmap:
            return "no_answer", "", []
        tool_data = json.dumps([{"tool": x.name, "data": scrub(x.data)} for x in results], default=str) \
            if results else "none"
        user = (f"Requester region: {r.region}. Reference date: {ref} (today is {ctx.today}).\n\n"
                f"PRECEDENCE:\n{c.precedence_notes}\n\n"
                f"STRUCTURED HOLIDAY RECORDS (extracted at ingestion):\n{holiday_block or 'none'}\n\n"
                f"TOOL DATA:\n{tool_data}\n\nSOURCES:\n{c.rendered}\n\nQUESTION:\n{message}")
        try:
            out = self.llm.complete_json(ANSWER_SYSTEM, user, purpose="answer", corpus_tokens=c.corpus_tokens,
                                         request_id=rid)
        except Exception as e:
            log("answer_failed", request_id=rid, err=type(e).__name__, corpus_tokens=c.corpus_tokens)
            return "no_answer", "", []
        smap = {s.sid: (s.doc_id, s.locator) for s in c.sources} | hmap
        cites, seen = [], set()
        for sid in out.get("used_sources") or []:
            if sid in smap and smap[sid] not in seen:  # only ids we sent -> never a restricted doc
                seen.add(smap[sid])
                cites.append({"doc_id": smap[sid][0], "locator": smap[sid][1]})
        # FIX: these two lines were inside the loop before (returned after the first source,
        # and returned None when used_sources was empty)
        answer = _SOURCE_REF.sub("", str(out.get("answer", "")))
        return out.get("status", "answered"), answer, cites

    def _holiday_records(self, ctx, ref: date):
        r = ctx.requester
        year = ref.year
        docs = list(self.kb.docs.values())
        in_use = holidays_for_range(docs, r.role, r.region, date(year, 1, 1), date(year, 12, 31), self.db_path)
        lines, hmap = [], {}
        for cal in (d for d in docs if d.doc_type == "calendar" and P.is_eligible(d, r.role, r.region, ref)):
            rows = [h for h in db.holidays_for_doc(cal.doc_id, self.db_path) if h["holiday_date"].startswith(str(year))]
            if not rows:
                continue
            hid = f"H{len(hmap) + 1}"
            hmap[hid] = (cal.doc_id, rows[0]["locator"] or "p. 1")
            items = "; ".join(f"{h['holiday_date']} {h['name'] or ''}{' (optional)' if h['optional'] else ''}".strip()
                              for h in rows)
            applies = any(date.fromisoformat(h["holiday_date"]) in in_use for h in rows)
            lines.append(f'<records id="{hid}" calendar_effective="{cal.effective_date}" '
                         f'status="{"applies" if applies else "superseded"}">{items}</records>')
        return "\n".join(lines), hmap

    # ------------------------------------------------------------------ rendering
    def _render_reads(self, ctx, results: list[T.ToolResult]) -> str:
        out, me = [], ctx.requester.employee_id
        for r in results:
            d = r.data
            who = "Your" if d.get("employee_id") == me else f"Employee {d.get('employee_id')}'s"
            if r.name == "get_leave_balance":
                out.append(f"{who} remaining leave balance: " + (", ".join(
                    f"{b['leave_type']} {b['balance_days']:g} days" for b in d["balances"]) or "no records") + ".")
            elif r.name == "list_leave_requests":
                out.append(f"{who} leave requests:\n" + ("\n".join(
                    f"- {q['request_id']}: {q['leave_type']} {q['start_date']} to {q['end_date']}, "
                    f"{q['working_days']} working days, {q['status']}" for q in d["requests"]) or "none found."))
            elif r.name == "get_team_leave":
                out.append(f"Team leave between {r.args['start_date']} and {r.args['end_date']}:\n" + ("\n".join(
                    f"- {q['full_name']} ({q['employee_id']}): {q['leave_type']} {q['start_date']} to "
                    f"{q['end_date']} ({q['status']})" for q in d["leave"]) or
                    "none of your direct reports has pending or approved leave in that period."))
        return "\n\n".join(out)

    @staticmethod
    def _tc(r: T.ToolResult) -> dict:
        tc = {"name": r.name, "args": scrub(r.args), "status": r.status}
        trace.step("tool", tool=r.name, args=tc["args"],
                   status=r.status, error=r.error)
        return tc

    @staticmethod
    def _finalise(t: Turn) -> Turn:
        t.answer = scrub_text(t.answer or "")
        if t.route in ("clarify", "no_answer", "refuse"):
            t.citations, t.tool_calls, t.pending_confirmation = [], [], False
        elif t.route == "tool":
            t.citations = []
        elif t.route == "rag":
            t.tool_calls = []
        return t


def fallback_plan(message: str) -> dict:
    """Keyword planner used only when every LLM provider fails."""
    m = message.lower()
    if re.search(r"\b(everyone|all employees|all staff|whole company)\b", m):
        return {"category": "bulk_request", "actions": []}
    actions = []
    if "balance" in m or "how many days" in m:
        actions.append({"tool": "get_leave_balance", "employee": "self",
                        "leave_type": "sick" if "sick" in m else "annual" if "annual" in m else None})
    if re.search(r"\bmy (leave )?requests\b", m):
        actions.append({"tool": "list_leave_requests", "employee": "self"})
    return {"category": "hr", "needs_documents": not actions, "search_query": message, "actions": actions,
            "holiday_lookup": bool(_HOLIDAY_WORDS.search(m))}