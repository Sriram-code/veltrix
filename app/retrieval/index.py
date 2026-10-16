"""Hybrid retrieval index.

Pipeline for one question (only over chunks the requester may read):
  1. BM25 keyword ranking
  2. Dense vector ranking (bge-small embeddings, cosine)
  3. Reciprocal Rank Fusion of 1 and 2
  4. Cross-encoder reranking of the top RERANK_TOP_N fused hits

Vector store (VECTOR_BACKEND):
  qdrant    - Qdrant. Embedded in-process by default (no extra
              container); set QDRANT_URL to use a Qdrant server.
  numpy     - exact brute-force search in memory.
  pgvector  - pre-embedded vectors stored in Postgres (chunk_embeddings
              table).  No embedding model is loaded on the server;
              run scripts/embed_to_pg.py locally first.
If Qdrant cannot start, numpy is used. If embeddings cannot load,
search is BM25 only. If the reranker cannot load, step 4 is skipped.
Access control: every backend searches ONLY the allowed chunk ids.
"""
from __future__ import annotations

import logging
import math
import os
import re
from collections import Counter

from app.ingestion.chunker import Chunk

logger = logging.getLogger("veltrix.index")

VECTOR_BACKEND = os.getenv("VECTOR_BACKEND", "qdrant").lower()
QDRANT_URL = os.getenv("QDRANT_URL", "").strip()
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "veltrix_chunks")
RERANKER = os.getenv("RERANKER", "none").lower()  # none (default) | fastembed
RERANK_MODEL = os.getenv("RERANK_MODEL",
                         "Xenova/ms-marco-MiniLM-L-6-v2")
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "30"))
_min = os.getenv("RERANK_MIN_SCORE", "").strip()
RERANK_MIN_SCORE = float(_min) if _min else None

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = set(
    "the a an of to and or in on for is are be by with as at it this "
    "that from your you i my me what how do does can".split())


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


class BM25:
    def __init__(self, docs, k1=1.4, b=0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(d) for d in docs]
        self.lens = [len(d) for d in docs]
        self.avg = (sum(self.lens) / len(self.lens)) if self.lens else 1.0
        df = Counter()
        for d in self.docs:
            df.update(d.keys())
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5))
                    for t, f in df.items()}

    def score(self, q, i) -> float:
        d, L, s = self.docs[i], self.lens[i], 0.0
        for t in q:
            if t in d:
                tf = d[t]
                s += self.idf[t] * tf * (self.k1 + 1) / (
                    tf + self.k1 * (1 - self.b + self.b * L / self.avg))
        return s


# ------------------------------------------------------------ stores
class NumpyStore:
    name = "numpy (exact, in memory)"

    def __init__(self, vecs):
        self.vecs = vecs

    def search(self, qv, allowed, limit):
        import numpy as np

        idx = np.array(sorted(allowed))
        sims = self.vecs[idx] @ qv
        order = np.argsort(-sims)[:limit]
        return [(int(idx[o]), float(sims[o])) for o in order]


class QdrantStore:
    def __init__(self, vecs, doc_ids):
        from qdrant_client import QdrantClient, models

        self.models = models
        if QDRANT_URL:
            self.client = QdrantClient(url=QDRANT_URL, timeout=10)
            self.name = "qdrant server (" + QDRANT_URL + ")"
        else:
            self.client = QdrantClient(location=":memory:")
            self.name = "qdrant (embedded, in memory)"
        col = QDRANT_COLLECTION
        if self.client.collection_exists(col):
            self.client.delete_collection(col)
        self.client.create_collection(
            col, vectors_config=models.VectorParams(
                size=int(vecs.shape[1]),
                distance=models.Distance.COSINE))
        points = [models.PointStruct(id=i, vector=vecs[i].tolist(),
                                     payload={"doc_id": doc_ids[i]})
                  for i in range(len(doc_ids))]
        for s in range(0, len(points), 256):
            self.client.upsert(col, points=points[s:s + 256])
        self.col = col

    def search(self, qv, allowed, limit):
        m = self.models
        flt = m.Filter(must=[m.HasIdCondition(has_id=sorted(allowed))])
        res = self.client.query_points(
            self.col, query=[float(x) for x in qv],
            query_filter=flt, limit=limit)
        return [(int(p.id), float(p.score)) for p in res.points]


def make_store(vecs, doc_ids):
    if VECTOR_BACKEND == "qdrant":
        try:
            return QdrantStore(vecs, doc_ids)
        except Exception as e:
            logger.warning("qdrant_unavailable_using_numpy",
                           extra={"fields": {"err": str(e)[:150]}})
    return NumpyStore(vecs)


def make_pg_store_from_db(rows, vecs, conn_factory):
    """Thin wrapper so index.py doesn't import pg_store at module level."""
    from app.retrieval.pg_store import make_pg_store
    return make_pg_store(rows, vecs, conn_factory)


# ------------------------------------------------------------ index
class HybridIndex:
    def __init__(self, chunks: list[Chunk], titles: dict[str, str],
                 use_dense: bool, embed_model: str, use_rerank=None):
        self.chunks = chunks
        self.texts = [f"{titles.get(c.doc_id, '')} - {c.locator}\n"
                      f"{c.text}" for c in chunks]
        self.bm25 = BM25([tokenize(t) for t in self.texts])
        self.embedder = None
        self.store = None
        self.reranker = None
        if use_dense and chunks:
            self._load_dense(embed_model)
        if use_rerank is None:
            use_rerank = use_dense and RERANKER == "fastembed"
        if use_rerank and chunks:
            self._load_reranker()

    @classmethod
    def from_pg(cls, conn_factory, use_rerank=None) -> "HybridIndex":
        """Build a HybridIndex whose dense store reads pre-embedded vectors
        from the chunk_embeddings table in Postgres.

        No embedding model is loaded — suitable for low-RAM deployments.
        """
        from app.ingestion.chunker import Chunk as _Chunk
        from app.retrieval.pg_store import load_chunks_from_pg

        logger.info("pgvector_store_loading_chunks_from_postgres")
        rows, vecs = load_chunks_from_pg(conn_factory)
        if not rows:
            raise RuntimeError(
                "chunk_embeddings table is empty. "
                "Run 'python scripts/embed_to_pg.py' locally first."
            )

        chunks = [
            _Chunk(
                chunk_id=r["chunk_id"],
                doc_id=r["doc_id"],
                locator=r["locator"],
                text=r["text"],
                tokens=r["tokens"],
            )
            for r in rows
        ]

        obj = cls.__new__(cls)
        obj.chunks = chunks
        obj.texts = [f"{r['doc_id']} - {r['locator']}\n{r['text']}" for r in rows]
        obj.bm25 = BM25([tokenize(t) for t in obj.texts])
        obj.embedder = _PgQueryEmbedder(conn_factory)
        obj.store = make_pg_store_from_db(rows, vecs, conn_factory)
        obj.reranker = None
        if use_rerank is None:
            use_rerank = RERANKER == "fastembed"
        if use_rerank:
            obj._load_reranker()
        logger.info("pgvector_store_ready", extra={"fields": {"chunks": len(chunks)}})
        return obj

    def _load_dense(self, embed_model):
        try:
            import numpy as np
            from fastembed import TextEmbedding

            self.embedder = TextEmbedding(
                model_name=embed_model,
                cache_dir=os.getenv("FASTEMBED_CACHE_PATH"))
            v = np.array(list(self.embedder.embed(self.texts)),
                         dtype="float32")
            v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
            self.store = make_store(v, [c.doc_id for c in self.chunks])
        except Exception as e:
            logger.warning("dense_disabled_bm25_only",
                           extra={"fields": {"err": str(e)[:150]}})
            self.embedder = None
            self.store = None

    def _load_reranker(self):
        try:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            self.reranker = TextCrossEncoder(
                model_name=RERANK_MODEL,
                cache_dir=os.getenv("FASTEMBED_CACHE_PATH"))
            list(self.reranker.rerank("warm up", ["warm up"]))
        except Exception as e:
            logger.warning("reranker_disabled",
                           extra={"fields": {"err": str(e)[:150]}})
            self.reranker = None

    def describe(self) -> str:
        parts = ["BM25"]
        if self.store is not None:
            parts.append("dense via " + self.store.name)
        if self.reranker is not None:
            parts.append("rerank (" + RERANK_MODEL.split("/")[-1] + ")")
        return " + ".join(parts)

    def _dense(self, query, allowed, limit):
        import numpy as np

        qv = np.array(list(self.embedder.query_embed([query]))[0],
                      dtype="float32")
        qv /= np.linalg.norm(qv) + 1e-9
        return self.store.search(qv, allowed, limit)

    def search(self, query: str, allowed: set[int], k: int = 30):
        """Ranked [(chunk_index, score)] from `allowed` only."""
        if not allowed:
            return []
        q = tokenize(query)
        bm = sorted(((i, self.bm25.score(q, i)) for i in allowed),
                    key=lambda t: -t[1])
        ranks: dict[int, float] = {}
        for r, (i, s) in enumerate(bm[:k * 2]):
            if s > 0:
                ranks[i] = ranks.get(i, 0) + 1 / (60 + r)
        if self.store is not None and self.embedder is not None:
            for r, (i, _) in enumerate(self._dense(query, allowed,
                                                   k * 2)):
                if i in allowed:  # defence in depth
                    ranks[i] = ranks.get(i, 0) + 1 / (60 + r)
        fused = sorted(ranks.items(), key=lambda t: -t[1])[:k]
        return self._rerank(query, fused)

    def _rerank(self, query, fused):
        if self.reranker is None or not fused:
            return fused
        top, rest = fused[:RERANK_TOP_N], fused[RERANK_TOP_N:]
        try:
            scores = list(self.reranker.rerank(
                query, [self.texts[i] for i, _ in top]))
        except Exception as e:
            logger.warning("rerank_failed",
                           extra={"fields": {"err": str(e)[:150]}})
            return fused
        ranked = sorted(((i, float(s)) for (i, _), s in zip(top, scores)),
                        key=lambda t: -t[1])
        if RERANK_MIN_SCORE is not None:
            ranked = [t for t in ranked if t[1] >= RERANK_MIN_SCORE]
            rest = []
        return ranked + rest


# ---------------------------------------------------------------------------
# Thin query embedder wrapper for the pgvector path
# (embeds only the QUERY at search time – no model loaded at startup)
# ---------------------------------------------------------------------------

class _PgQueryEmbedder:
    """Lazy-loads the embedding model ONLY when the first query arrives.
    On a RAM-constrained server with EMBEDDINGS=none this is never called
    (PgVectorStore does server-side cosine search); it is here as a safety
    net in case NumpyPgStore (BYTEA fallback) is used instead.
    """

    def __init__(self, conn_factory):
        self._conn_factory = conn_factory
        self._model = None

    def _get_model(self):
        if self._model is None:
            from fastembed import TextEmbedding
            model_name = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
            self._model = TextEmbedding(
                model_name=model_name,
                cache_dir=os.getenv("FASTEMBED_CACHE_PATH"),
            )
        return self._model

    def query_embed(self, texts):
        return self._get_model().query_embed(texts)