"""PostgreSQL-backed vector store for retrieval.

Two sub-modes, chosen automatically at startup:
  - pgvector  : uses the  <=>  operator (cosine distance) via the pgvector
                extension.  Fast even for thousands of chunks.
  - bytea     : loads all vectors from the BYTEA column into a numpy array
                at startup and does exact cosine search in Python.
                Falls back to this when the pgvector extension is absent.

Either way the interface is identical to NumpyStore / QdrantStore in index.py:
    store.search(qv, allowed_set, limit) -> list[(chunk_index, score)]

chunk_index here maps to the position of the chunk in kb.chunks (same order
as returned by load_chunks_from_pg).

Usage (from index.py):
    VECTOR_BACKEND=pgvector
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np

logger = logging.getLogger("veltrix.pg_store")


# ---------------------------------------------------------------------------
# Load all pre-embedded chunks from Postgres
# ---------------------------------------------------------------------------

def load_chunks_from_pg(conn_factory) -> tuple[list[dict], Optional[np.ndarray]]:
    """Return (rows, vecs_or_None).

    rows  – list of dicts with keys: chunk_id, doc_id, locator, text, tokens
    vecs  – float32 numpy array shaped (N, dim), normalised; or None if the
            table uses BYTEA storage (caller will load vecs separately).
    """
    import psycopg2.extras

    conn = conn_factory()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # Detect storage mode
            cur.execute("""
                SELECT data_type
                FROM information_schema.columns
                WHERE table_name = 'chunk_embeddings'
                  AND column_name = 'embedding'
                  AND table_schema = current_schema()
            """)
            row = cur.fetchone()
            if row is None:
                raise RuntimeError("chunk_embeddings table not found. "
                                   "Run scripts/embed_to_pg.py first.")
            dtype = (row["data_type"] or "").lower()
            has_pgvector = "user-defined" in dtype or "vector" in dtype

            if has_pgvector:
                cur.execute(
                    "SELECT chunk_id, doc_id, locator, text, tokens, "
                    "       embedding::text AS emb_text "
                    "FROM chunk_embeddings ORDER BY chunk_id"
                )
            else:
                cur.execute(
                    "SELECT chunk_id, doc_id, locator, text, tokens, "
                    "       embedding, embed_dim "
                    "FROM chunk_embeddings ORDER BY chunk_id"
                )
            db_rows = cur.fetchall()
    finally:
        conn.rollback()
        conn.close()

    if not db_rows:
        return [], None

    rows = [{"chunk_id": r["chunk_id"], "doc_id": r["doc_id"],
             "locator": r["locator"], "text": r["text"],
             "tokens": r["tokens"]} for r in db_rows]

    if has_pgvector:
        # Parse pgvector text format: "[0.1,0.2,...]"
        vecs = np.array(
            [list(map(float, r["emb_text"].strip("[]").split(",")))
             for r in db_rows],
            dtype="float32",
        )
    else:
        # Deserialise BYTEA
        dim = db_rows[0]["embed_dim"]
        vecs = np.stack(
            [np.frombuffer(bytes(r["embedding"]), dtype="float32").reshape(dim)
             for r in db_rows]
        )

    return rows, vecs


# ---------------------------------------------------------------------------
# PgVectorStore (pgvector cosine operator)
# ---------------------------------------------------------------------------

class PgVectorStore:
    """Uses the  <=>  operator for server-side cosine nearest-neighbour."""

    name = "pgvector (postgres, cosine)"

    def __init__(self, chunk_id_to_index: dict[str, int], conn_factory):
        self._c2i = chunk_id_to_index
        self._conn_factory = conn_factory

    def search(self, qv: np.ndarray, allowed: set[int], limit: int):
        import psycopg2.extras

        # Build the allowed chunk_id list from the allowed *index* set
        i2c = {v: k for k, v in self._c2i.items()}
        allowed_ids = [i2c[i] for i in allowed if i in i2c]
        if not allowed_ids:
            return []

        vec_str = "[" + ",".join(f"{x:.8f}" for x in qv.tolist()) + "]"

        conn = self._conn_factory()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT chunk_id,
                           1 - (embedding <=> %s::vector) AS score
                    FROM chunk_embeddings
                    WHERE chunk_id = ANY(%s)
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (vec_str, allowed_ids, vec_str, limit),
                )
                results = cur.fetchall()
        finally:
            conn.rollback()
            conn.close()

        return [(self._c2i[r["chunk_id"]], float(r["score"]))
                for r in results if r["chunk_id"] in self._c2i]


# ---------------------------------------------------------------------------
# NumpyPgStore (BYTEA fallback — vectors already loaded into RAM as numpy)
# ---------------------------------------------------------------------------

class NumpyPgStore:
    """Exact in-memory cosine search; vectors loaded once from Postgres BYTEA."""

    name = "numpy (loaded from postgres BYTEA)"

    def __init__(self, vecs: np.ndarray):
        self.vecs = vecs

    def search(self, qv: np.ndarray, allowed: set[int], limit: int):
        idx = np.array(sorted(allowed))
        sims = self.vecs[idx] @ qv
        order = np.argsort(-sims)[:limit]
        return [(int(idx[o]), float(sims[o])) for o in order]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_pg_store(
    rows: list[dict],
    vecs: Optional[np.ndarray],
    conn_factory,
) -> "PgVectorStore | NumpyPgStore":
    """Return the right store based on what the DB supports."""
    chunk_id_to_index = {r["chunk_id"]: i for i, r in enumerate(rows)}

    # Try pgvector operator — if we have vecs via text parsing, the DB has it
    # but we choose server-side search for scalability.
    # Probe the DB once to confirm the <=> operator works.
    try:
        conn = conn_factory()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM chunk_embeddings "
                    "ORDER BY embedding <=> '[0]'::vector LIMIT 1"
                )
        finally:
            conn.rollback()
            conn.close()
        logger.info("pg_store_mode=pgvector")
        return PgVectorStore(chunk_id_to_index, conn_factory)
    except Exception as exc:
        logger.warning("pgvector_unavailable_using_numpy_fallback",
                       extra={"fields": {"err": str(exc)[:150]}})

    # Fall back to in-memory numpy search
    if vecs is None:
        raise RuntimeError("No vectors available and pgvector is not working.")
    logger.info("pg_store_mode=numpy_bytea_fallback")
    return NumpyPgStore(vecs)
