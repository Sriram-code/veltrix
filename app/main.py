"""HTTP layer: POST /chat and GET /health, plus a test UI at GET /."""
from __future__ import annotations

import threading
from contextlib import asynccontextmanager
from pathlib import Path

import anyio
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse

from app import db
from app import trace
from app.config import settings
from app.ingestion.pipeline import ingest
from app.llm.client import LLMClient
from app.logging_setup import log, setup_logging
from app.orchestration.orchestrator import Orchestrator
from app.orchestration.sessions import SessionStore

UI_FILE = Path(__file__).parent / "static" / "index.html"


class AppState:
    ready = False
    orchestrator = None
    sessions = SessionStore()


state = AppState()


def _run_ingestion():
    try:
        kb = ingest()
        llm = LLMClient()
        state.orchestrator = Orchestrator(kb, llm, lambda: settings.today, None)
        state.ready = True
    except Exception as e:
        log("startup_failed", err=str(e)[:300])


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging(settings.log_file)
    threading.Thread(target=_run_ingestion, daemon=True).start()
    yield


app = FastAPI(title="Veltrix PeopleOps Assistant", lifespan=lifespan)


@app.get("/", include_in_schema=False)
def ui():
    if UI_FILE.is_file():
        return FileResponse(UI_FILE)
    return JSONResponse({"error": "UI not found"}, status_code=404)


@app.get("/health")
def health():
    if state.ready:
        return {"status": "ok"}
    return JSONResponse({"status": "starting"}, status_code=503)

@app.get("/debug/trace/{request_id}", include_in_schema=False)
def get_trace(request_id: str, request: Request):
    """Monitor panel only. Not part of the API contract."""
    emp = (request.headers.get("X-Employee-Id") or "").strip()
    t = trace.get(request_id, emp) if emp else None
    if t is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    return t


@app.get("/debug/traces", include_in_schema=False)
def list_traces(request: Request):
    emp = (request.headers.get("X-Employee-Id") or "").strip()
    return trace.recent(emp) if emp else []


def _valid_body(body) -> bool:
    if not isinstance(body, dict):
        return False
    sid = body.get("session_id")
    msg = body.get("message")
    if not isinstance(sid, str) or not sid.strip():
        return False
    if not isinstance(msg, str) or not msg.strip():
        return False
    return True


@app.post("/chat")
async def chat(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON body"}, status_code=400)
    if not _valid_body(body):
        return JSONResponse(
            {"error": "body needs non-empty session_id and message"},
            status_code=400,
        )

    emp_id = (request.headers.get("X-Employee-Id") or "").strip()
    requester = db.get_employee(emp_id) if emp_id else None
    if requester is None:
        return JSONResponse(
            {"error": "unknown or missing X-Employee-Id"}, status_code=401
        )

    session = state.sessions.get_or_create(body["session_id"], requester.employee_id)
    if session is None:
        log("session_owner_mismatch", employee_id=requester.employee_id)
        return JSONResponse(
            {"error": "session belongs to another employee"}, status_code=403
        )

    if not state.ready or state.orchestrator is None:
        return JSONResponse({"error": "service is starting"}, status_code=503)

    message = body["message"][:4000]
    return await anyio.to_thread.run_sync(
        state.orchestrator.handle, session, requester, message
    )