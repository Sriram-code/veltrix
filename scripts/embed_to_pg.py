#!/usr/bin/env python3
"""
Pre-embed all corpus chunks locally and push the vectors into PostgreSQL.

Run this ONCE on your local machine (which has enough RAM for fastembed)
BEFORE deploying to Render.  After this, the Render service can do retrieval
with VECTOR_BACKEND=pgvector and EMBEDDINGS=none – no embedding model is ever
loaded on the server.

Usage
-----
    python scripts/embed_to_pg.py [--model BAAI/bge-small-en-v1.5]

Environment variables required (read from .env automatically):
    DATABASE_URL  – PostgreSQL connection string to the Render DB
    CORPUS_DIR    – path to the corpus directory  (default: ./data/corpus)
    MANIFEST_PATH – path to manifest.json         (default: ./data/manifest.json)
    EMBED_MODEL   – model name override (or pass --model on the CLI)

The script is idempotent: it upserts rows by chunk_id, so re-running it after
adding new documents only inserts the new chunks.

pgvector
--------
The script enables the pgvector extension and creates the chunk_embeddings table
automatically.  If pgvector is not available on the server, it falls back to
storing vectors as a plain BYTEA column (numpy-serialised float32 array) and
prints a warning – retrieval will still work, but similarity search will be done
in Python instead of in Postgres.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# --------------------------------------------------------------------------- #
# Resolve project root and bootstrap imports
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import _load_dotenv, settings  # noqa: E402

_load_dotenv()

import numpy as np  # noqa: E402
import psycopg2  # noqa: E402
import psycopg2.extras  # noqa: E402

# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pre-embed corpus chunks into PostgreSQL")
    p.add_argument(
        "--model",
        default=os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5"),
        help="fastembed model name (default: BAAI/bge-small-en-v1.5)",
    )
    p.add_argument(
        "--batch", type=int, default=64,
        help="embedding batch size (default: 64)",
    )
    p.add_argument(
        "--force", action="store_true",
        help="re-embed and overwrite all chunks even if already in DB",
    )
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Ingestion (reuse existing pipeline, dense=False to skip embedding)
# --------------------------------------------------------------------------- #

def _ingest_chunks():
    """Load and chunk all corpus documents without embedding."""
    from app.ingestion.pipeline import ingest
    print(">>> Ingesting corpus (parse + chunk, no embedding) ...")
    kb = ingest(use_dense=False)
    print(f"  {len(kb.chunks)} chunks from {len(kb.docs)} documents"
          f"  ({len(kb.failed)} failed)")
    return kb


# --------------------------------------------------------------------------- #
# Postgres helpers
# --------------------------------------------------------------------------- #

_CREATE_EXTENSION = "CREATE EXTENSION IF NOT EXISTS vector"

_CREATE_TABLE_PGVECTOR = """
CREATE TABLE IF NOT EXISTS chunk_embeddings (
    chunk_id   TEXT PRIMARY KEY,
    doc_id     TEXT NOT NULL,
    locator    TEXT NOT NULL,
    text       TEXT NOT NULL,
    tokens     INTEGER NOT NULL,
    embedding  vector(%s)
)
"""

_CREATE_TABLE_BYTEA = """
CREATE TABLE IF NOT EXISTS chunk_embeddings (
    chunk_id   TEXT PRIMARY KEY,
    doc_id     TEXT NOT NULL,
    locator    TEXT NOT NULL,
    text       TEXT NOT NULL,
    tokens     INTEGER NOT NULL,
    embedding  BYTEA NOT NULL,
    embed_dim  INTEGER NOT NULL
)
"""

_CREATE_INDEX_PGVECTOR = """
CREATE INDEX IF NOT EXISTS idx_chunk_emb_cosine
    ON chunk_embeddings USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 10)
"""


def _try_enable_pgvector(cur) -> bool:
    """Return True if pgvector is available, False otherwise."""
    try:
        cur.execute(_CREATE_EXTENSION)
        return True
    except Exception as exc:
        print(f"  WARNING: pgvector not available ({exc}); using BYTEA fallback.")
        return False


def _ensure_table(conn, dim: int) -> bool:
    """Create chunk_embeddings table. Returns True if pgvector was used."""
    with conn.cursor() as cur:
        has_pgvector = _try_enable_pgvector(cur)
        conn.commit()

    with conn.cursor() as cur:
        if has_pgvector:
            cur.execute(_CREATE_TABLE_PGVECTOR, (dim,))
            # Attempt IVFFlat index (silently skip if not enough rows yet)
            try:
                cur.execute(_CREATE_INDEX_PGVECTOR)
            except Exception:
                pass
        else:
            cur.execute(_CREATE_TABLE_BYTEA)
        conn.commit()
    return has_pgvector


def _existing_chunk_ids(conn) -> set[str]:
    with conn.cursor() as cur:
        try:
            cur.execute("SELECT chunk_id FROM chunk_embeddings")
            return {r[0] for r in cur.fetchall()}
        except Exception:
            conn.rollback()
            return set()


# --------------------------------------------------------------------------- #
# Embedding
# --------------------------------------------------------------------------- #

def _load_embedder(model_name: str):
    from fastembed import TextEmbedding
    print(f">>> Loading embedding model: {model_name} ...")
    embedder = TextEmbedding(model_name=model_name,
                             cache_dir=os.getenv("FASTEMBED_CACHE_PATH"))
    # warm-up
    list(embedder.embed(["warm up"]))
    print("  Model ready.")
    return embedder


def _embed_batch(embedder, texts: list[str]) -> np.ndarray:
    vecs = np.array(list(embedder.embed(texts)), dtype="float32")
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs /= norms + 1e-9
    return vecs


# --------------------------------------------------------------------------- #
# Upsert
# --------------------------------------------------------------------------- #

def _upsert_pgvector(conn, rows):
    """rows = list of (chunk_id, doc_id, locator, text, tokens, vec_list)"""
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(
            cur,
            """
            INSERT INTO chunk_embeddings
                (chunk_id, doc_id, locator, text, tokens, embedding)
            VALUES %s
            ON CONFLICT (chunk_id) DO UPDATE SET
                doc_id    = EXCLUDED.doc_id,
                locator   = EXCLUDED.locator,
                text      = EXCLUDED.text,
                tokens    = EXCLUDED.tokens,
                embedding = EXCLUDED.embedding
            """,
            rows,
            template="(%s, %s, %s, %s, %s, %s::vector)",
        )
    conn.commit()


def _upsert_bytea(conn, rows_with_vecs: list[tuple]):
    """rows_with_vecs: (chunk_id, doc_id, locator, text, tokens, np.ndarray)"""
    data = [
        (cid, did, loc, txt, tok, vec.tobytes(), len(vec))
        for cid, did, loc, txt, tok, vec in rows_with_vecs
    ]
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(
            cur,
            """
            INSERT INTO chunk_embeddings
                (chunk_id, doc_id, locator, text, tokens, embedding, embed_dim)
            VALUES %s
            ON CONFLICT (chunk_id) DO UPDATE SET
                doc_id    = EXCLUDED.doc_id,
                locator   = EXCLUDED.locator,
                text      = EXCLUDED.text,
                tokens    = EXCLUDED.tokens,
                embedding = EXCLUDED.embedding,
                embed_dim = EXCLUDED.embed_dim
            """,
            data,
        )
    conn.commit()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    args = _parse_args()

    database_url = settings.database_url or os.getenv("DATABASE_URL", "")
    if not database_url:
        sys.exit("ERROR: DATABASE_URL is not set. Check your .env file.")

    # 1. Ingest corpus (no embedding)
    kb = _ingest_chunks()
    chunks = kb.chunks
    if not chunks:
        sys.exit("No chunks found -- check CORPUS_DIR / MANIFEST_PATH.")

    # 2. Load embedding model
    embedder = _load_embedder(args.model)

    # 3. Connect to Postgres
    print(">>> Connecting to Postgres ...")
    conn = psycopg2.connect(database_url)
    conn.autocommit = False

    # 4. Create table
    dim = next(len(v) for v in embedder.embed(["dim probe"]))
    has_pgvector = _ensure_table(conn, dim)
    mode = "pgvector" if has_pgvector else "BYTEA (numpy fallback)"
    print(f"  chunk_embeddings table ready  [{mode}]  dim={dim}")

    # 5. Filter chunks that need embedding
    if args.force:
        todo = chunks
        print(f"  --force: re-embedding all {len(todo)} chunks")
    else:
        existing = _existing_chunk_ids(conn)
        todo = [c for c in chunks if c.chunk_id not in existing]
        print(f"  {len(existing)} already in DB, {len(todo)} to embed")

    if not todo:
        print("Nothing to do -- all chunks are already embedded.")
        conn.close()
        return

    # 6. Build the text strings exactly as HybridIndex does
    titles = {d: m.title for d, m in kb.docs.items()}
    texts = [f"{titles.get(c.doc_id, '')} - {c.locator}\n{c.text}" for c in todo]

    # 7. Embed in batches and upsert
    total = len(todo)
    batch_size = args.batch
    t0 = time.time()
    print(f">>> Embedding {total} chunks in batches of {batch_size} ...")

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_chunks = todo[start:end]
        batch_texts = texts[start:end]

        vecs = _embed_batch(embedder, batch_texts)

        rows = [
            (c.chunk_id, c.doc_id, c.locator, c.text, c.tokens, vecs[i].tolist())
            for i, c in enumerate(batch_chunks)
        ]

        if has_pgvector:
            _upsert_pgvector(conn, rows)
        else:
            rows_np = [
                (c.chunk_id, c.doc_id, c.locator, c.text, c.tokens, vecs[i])
                for i, c in enumerate(batch_chunks)
            ]
            _upsert_bytea(conn, rows_np)

        elapsed = time.time() - t0
        pct = end / total * 100
        rate = end / elapsed
        print(f"  [{end:>4}/{total}]  {pct:5.1f}%  {rate:.1f} chunks/s", end="\r")

    print(f"\nDone! {total} chunks embedded and stored in {time.time() - t0:.1f}s")
    print(f"  --> Set VECTOR_BACKEND=pgvector and EMBEDDINGS=none in render.yaml / Render dashboard")
    conn.close()


if __name__ == "__main__":
    main()
