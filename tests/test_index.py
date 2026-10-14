"""Hybrid index: access filter in every stage, reranker order."""
import numpy as np
import pytest

from app.ingestion.chunker import Chunk
from app.retrieval import index as IX


def chunks():
    data = [("A", "Sick leave", "Employees get 12 sick days."),
            ("A", "Annual leave", "Annual leave accrues monthly."),
            ("B", "Sick leave", "Germany: 30 sick days."),
            ("C", "Travel", "Hotel limit is 150 per night.")]
    return [Chunk(f"C{i}", d, loc, t, 5)
            for i, (d, loc, t) in enumerate(data)]


class FakeEmbedder:
    def query_embed(self, texts):
        return [np.array([1.0, 0.0, 0.0], dtype="float32")]


def index_with_vectors(store_cls):
    ix = IX.HybridIndex(chunks(), {}, use_dense=False, embed_model="x")
    vecs = np.array([[1, 0, 0], [0.5, 0.5, 0], [0.9, 0.1, 0],
                     [0, 0, 1]], dtype="float32")
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    ix.embedder = FakeEmbedder()
    ix.store = (store_cls(vecs) if store_cls is IX.NumpyStore
                else store_cls(vecs, [c.doc_id for c in ix.chunks]))
    return ix


def test_bm25_only_respects_allowed():
    ix = IX.HybridIndex(chunks(), {}, use_dense=False, embed_model="x")
    hits = ix.search("sick days", allowed={0, 1, 3})
    assert hits and all(i in {0, 1, 3} for i, _ in hits)
    assert ix.describe() == "BM25"


def test_numpy_store_respects_allowed():
    ix = index_with_vectors(IX.NumpyStore)
    hits = ix.search("sick", allowed={1, 3})
    assert {i for i, _ in hits} <= {1, 3}
    assert "numpy" in ix.describe()


def test_qdrant_store_respects_allowed():
    pytest.importorskip("qdrant_client")
    ix = index_with_vectors(IX.QdrantStore)
    hits = ix.search("sick", allowed={1, 3})
    assert {i for i, _ in hits} <= {1, 3}
    assert "qdrant" in ix.describe()


def test_reranker_reorders_and_keeps_allowed():
    class FakeReranker:
        def rerank(self, query, docs):
            return [10.0 if "Germany" in d else 0.0 for d in docs]

    ix = IX.HybridIndex(chunks(), {}, use_dense=False, embed_model="x")
    ix.reranker = FakeReranker()
    hits = ix.search("sick leave days", allowed={0, 2})
    assert hits[0][0] == 2
    assert {i for i, _ in hits} <= {0, 2}


def test_empty_allowed_returns_nothing():
    ix = index_with_vectors(IX.NumpyStore)
    assert ix.search("sick", allowed=set()) == []