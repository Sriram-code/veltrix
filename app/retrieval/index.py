"""Local hybrid index: BM25 (always) + dense open-source embeddings via fastembed on CPU
(BAAI/bge-small-en-v1.5, no API key), fused with Reciprocal Rank Fusion.
If the embedding model can't load, we fall back to BM25 only."""
from __future__ import annotations

import logging
import math
import os
import re
from collections import Counter

from app.ingestion.chunker import Chunk

logger = logging.getLogger("veltrix.index")
_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = set("the a an of to and or in on for is are be by with as at it this that from your you i my me what how "
            "do does can".split())


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.4, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(d) for d in docs]
        self.lens = [len(d) for d in docs]
        self.avg = (sum(self.lens) / len(self.lens)) if self.lens else 1.0
        df = Counter()
        for d in self.docs:
            df.update(d.keys())
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def score(self, q: list[str], i: int) -> float:
        d, L, s = self.docs[i], self.lens[i], 0.0
        for t in q:
            if t in d:
                tf = d[t]
                s += self.idf[t] * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * L / self.avg))
        return s


class HybridIndex:
    def __init__(self, chunks: list[Chunk], titles: dict[str, str], use_dense: bool, embed_model: str):
        self.chunks = chunks
        self.bm25 = BM25([tokenize(titles.get(c.doc_id, "") + " " + c.locator + " " + c.text) for c in chunks])
        self.embedder = None
        self.vecs = None
        if use_dense and chunks:
            try:
                import numpy as np
                from fastembed import TextEmbedding

                self.embedder = TextEmbedding(model_name=embed_model, cache_dir=os.getenv("FASTEMBED_CACHE_PATH"))
                texts = [f"{titles.get(c.doc_id, '')} - {c.locator}\n{c.text}" for c in chunks]
                v = np.array(list(self.embedder.embed(texts)), dtype="float32")
                self.vecs = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
            except Exception as e:
                logger.warning("dense_disabled_bm25_only", extra={"fields": {"err": str(e)[:120]}})
                self.embedder = None

    def search(self, query: str, allowed: set[int], k: int = 30) -> list[tuple[int, float]]:
        """Search ONLY within `allowed` chunk indices (access filtering happens before scoring)."""
        if not allowed:
            return []
        q = tokenize(query)
        bm = sorted(((i, self.bm25.score(q, i)) for i in allowed), key=lambda t: -t[1])
        ranks: dict[int, float] = {}
        for r, (i, s) in enumerate(bm[:k * 2]):
            if s > 0:
                ranks[i] = ranks.get(i, 0) + 1 / (60 + r)
        if self.embedder is not None and self.vecs is not None:
            import numpy as np

            qv = np.array(list(self.embedder.query_embed([query]))[0], dtype="float32")
            qv /= np.linalg.norm(qv) + 1e-9
            idx = np.array(sorted(allowed))
            order = idx[np.argsort(-(self.vecs[idx] @ qv))]
            for r, i in enumerate(order[: k * 2]):
                ranks[int(i)] = ranks.get(int(i), 0) + 1 / (60 + r)
        return sorted(ranks.items(), key=lambda t: -t[1])[:k]
