"""Ingestion: manifest -> parse every file -> chunks + holiday records -> index."""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from app import db
from app.config import settings
from app.ingestion.chunker import Chunk, chunk_blocks
from app.ingestion.holidays import extract_holidays
from app.ingestion.parsers import parse_any
from app.logging_setup import log
from app.retrieval.index import HybridIndex, VECTOR_BACKEND
from app.retrieval.precedence import DocMeta


@dataclass
class KnowledgeBase:
    docs: dict[str, DocMeta] = field(default_factory=dict)
    chunks: list[Chunk] = field(default_factory=list)
    index: HybridIndex | None = None
    holiday_count: int = 0
    failed: list[str] = field(default_factory=list)
    corpus_hash: str = ""


def corpus_fingerprint(corpus: Path, manifest: Path) -> str:
    """Hash of the manifest and every corpus file. Changes whenever
    the data changes, so cached answers are never reused.
    Files are streamed so large PDFs aren't loaded into memory at once."""
    h = hashlib.sha256(manifest.read_bytes())
    for p in sorted(corpus.rglob("*")):
        if p.is_file():
            h.update(p.name.encode())
            with p.open("rb") as f:
                for block in iter(lambda: f.read(1 << 20), b""):
                    h.update(block)
    return h.hexdigest()[:16]


def _existing_holidays() -> dict[str, list[tuple]]:
    """Holiday rows already in Postgres, grouped by doc_id."""
    if os.getenv("FORCE_REINGEST") == "1":
        return {}
    out: dict[str, list[tuple]] = {}
    for r in db.query(
        "SELECT doc_id, holiday_date, name, locator, optional FROM ingested_holidays"
    ):
        out.setdefault(r["doc_id"], []).append(
            (r["doc_id"], r["holiday_date"], r["name"], r["locator"], r["optional"])
        )
    return out


def ingest(corpus_dir: str | None = None, manifest_path: str | None = None, db_path: str | None = None,
           use_dense: bool | None = None) -> KnowledgeBase:
    t0 = time.time()
    corpus = Path(corpus_dir or settings.corpus_dir)
    manifest = json.loads(Path(manifest_path or settings.manifest_path).read_text(encoding="utf-8"))
    db.ensure_extra_schema(db_path)
    existing = _existing_holidays()
    kb = KnowledgeBase()
    holiday_rows: list[tuple] = []
    for entry in manifest:
        meta = DocMeta.from_manifest(entry)
        kb.docs[meta.doc_id] = meta
        path = corpus / meta.file
        if not path.exists():
            kb.failed.append(meta.doc_id)
            log("ingest_missing_file", doc_id=meta.doc_id)
            continue
        try:
            kb.chunks.extend(chunk_blocks(meta.doc_id, parse_any(path), path.suffix.lower() == ".pdf"))
            if meta.doc_type == "calendar":
                rows = existing.get(meta.doc_id)
                if rows is None:  # not in DB yet -> parse the calendar again
                    rows = extract_holidays(meta.doc_id, parse_any(path, merge_lines=False),
                                            meta.effective_date.year, "US" in meta.regions)
                holiday_rows.extend(rows)
                log("ingest_calendar", doc_id=meta.doc_id, holidays=len(rows),
                    reused=meta.doc_id in existing)
        except Exception as e:
            kb.failed.append(meta.doc_id)
            log("ingest_failed", doc_id=meta.doc_id, err=str(e)[:200])
    for i, c in enumerate(kb.chunks):
        c.chunk_id = f"C{i}"
    db.replace_holidays(holiday_rows, db_path)
    kb.holiday_count = len(holiday_rows)
    kb.corpus_hash = corpus_fingerprint(
        corpus, Path(manifest_path or settings.manifest_path))

    # ------------------------------------------------------------------ #
    # Build the retrieval index
    # ------------------------------------------------------------------ #
    if VECTOR_BACKEND == "pgvector":
        # Chunks and BM25 corpus are loaded from Postgres; no embedding model
        # is instantiated on the server — this is the RAM-saving path.
        log("ingest_using_pgvector_store")
        try:
            kb.index = HybridIndex.from_pg(conn_factory=db.connect)
            # Overwrite chunks so that access-control in retriever.py works
            # (chunk positions must match the index's internal chunk list)
            kb.chunks = kb.index.chunks
        except Exception as exc:
            log("pgvector_load_failed_bm25_only", err=str(exc)[:300])
            dense = settings.embeddings == "fastembed" if use_dense is None else use_dense
            kb.index = HybridIndex(kb.chunks, {d: m.title for d, m in kb.docs.items()},
                                   dense, settings.embed_model)
    else:
        dense = settings.embeddings == "fastembed" if use_dense is None else use_dense
        kb.index = HybridIndex(kb.chunks, {d: m.title for d, m in kb.docs.items()},
                               dense, settings.embed_model)

    log("ingest_done", docs=len(kb.docs), chunks=len(kb.chunks),
        holidays=len(holiday_rows), failed=kb.failed,
        dense=kb.index.embedder is not None, seconds=round(time.time() - t0, 1))
    return kb