"""
tests/test_retriever.py
────────────────────────
Unit and integration tests for the retrieval pipeline.
Uses mock embeddings to avoid requiring API keys during testing.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.indexing.chunker import DocumentChunk
from src.indexing.vector_store import FAISSVectorStore, SearchResult
from src.ingestion.base import DocumentType

# ── Helpers ──────────────────────────────────────────────────────

def make_chunk(
    chunk_id: str,
    text: str,
    doc_type: DocumentType = DocumentType.RUNBOOK,
    metadata: dict | None = None,
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id=f"doc_{chunk_id}",
        source_system="test",
        document_type=doc_type,
        text=text,
        token_count=len(text.split()),
        chunk_index=0,
        total_chunks=1,
        metadata=metadata or {},
    )


def random_embedding(dims: int = 64) -> np.ndarray:
    vec = np.random.randn(dims).astype(np.float32)
    return vec / np.linalg.norm(vec)  # L2 normalise


# ── FAISSVectorStore tests ────────────────────────────────────────

class TestFAISSVectorStore:

    DIMS = 64  # small dims for fast testing

    @pytest.fixture
    def store(self, tmp_path):
        config = {
            "index_path": str(tmp_path / "faiss_index"),
            "index_type": "HNSW",
            "hnsw_m": 8,
            "hnsw_ef_construction": 64,
            "hnsw_ef_search": 32,
            "dimensions": self.DIMS,
        }
        return FAISSVectorStore(config)

    def test_empty_store_returns_no_results(self, store):
        query = random_embedding(self.DIMS)
        results = store.search(query, top_k=5)
        assert results == []

    def test_add_and_retrieve(self, store):
        chunk = make_chunk("c1", "database connection pool exhaustion fix")
        embedding = random_embedding(self.DIMS)
        store.add_chunks([(chunk, embedding)])

        query = random_embedding(self.DIMS)
        results = store.search(query, top_k=1)
        assert len(results) == 1
        assert results[0].chunk.chunk_id == "c1"

    def test_top_k_respected(self, store):
        pairs = [
            (make_chunk(f"c{i}", f"document {i}"), random_embedding(self.DIMS))
            for i in range(10)
        ]
        store.add_chunks(pairs)
        results = store.search(random_embedding(self.DIMS), top_k=3)
        assert len(results) <= 3

    def test_metadata_filter(self, store):
        chunk_a = make_chunk("ca", "payments issue", metadata={"service": "payments"})
        chunk_b = make_chunk("cb", "auth issue", metadata={"service": "auth"})
        store.add_chunks([
            (chunk_a, random_embedding(self.DIMS)),
            (chunk_b, random_embedding(self.DIMS)),
        ])
        results = store.search(
            random_embedding(self.DIMS),
            top_k=10,
            filters={"service": "payments"},
        )
        assert all(r.chunk.metadata["service"] == "payments" for r in results)

    def test_rebuild_index(self, store):
        # Add initial chunks
        pairs = [(make_chunk(f"c{i}", f"text {i}"), random_embedding(self.DIMS)) for i in range(5)]
        store.add_chunks(pairs)

        # Rebuild with new chunks
        new_pairs = [(make_chunk(f"new{i}", f"new text {i}"), random_embedding(self.DIMS)) for i in range(3)]
        store.rebuild_index(new_pairs)

        assert store.total_vectors == 3  # only new chunks
        assert store.total_chunks == 3

    def test_duplicate_chunks_not_added(self, store):
        chunk = make_chunk("dup", "duplicate text")
        emb = random_embedding(self.DIMS)
        store.add_chunks([(chunk, emb)])
        store.add_chunks([(chunk, emb)])  # add same chunk again
        assert store.total_vectors == 1

    def test_search_result_has_score(self, store):
        chunk = make_chunk("scored", "some text about kubernetes pod")
        store.add_chunks([(chunk, random_embedding(self.DIMS))])
        results = store.search(random_embedding(self.DIMS), top_k=1)
        assert 0 <= results[0].score <= 1.0

    def test_stats(self, store):
        stats = store.stats()
        assert "total_vectors" in stats
        assert "dimensions" in stats
        assert stats["dimensions"] == self.DIMS

    def test_persistence_and_reload(self, tmp_path):
        config = {
            "index_path": str(tmp_path / "persistent_index"),
            "dimensions": self.DIMS,
            "hnsw_m": 8,
            "hnsw_ef_construction": 64,
            "hnsw_ef_search": 32,
        }
        # Build and save
        store1 = FAISSVectorStore(config)
        chunk = make_chunk("persist1", "persistent document text")
        store1.add_chunks([(chunk, random_embedding(self.DIMS))])
        count = store1.total_vectors

        # Reload from disk
        store2 = FAISSVectorStore(config)
        assert store2.total_vectors == count


# ── SearchResult tests ────────────────────────────────────────────

class TestSearchResult:

    def test_to_dict_has_required_fields(self):
        chunk = make_chunk("c1", "test text")
        result = SearchResult(chunk=chunk, score=0.87, rank=1)
        d = result.to_dict()
        assert "chunk_id" in d
        assert "text" in d
        assert "score" in d
        assert d["score"] == 0.87
        assert d["rank"] == 1


# ── Cross-encoder re-ranker tests (mocked) ───────────────────────

class TestCrossEncoderReranker:

    @patch("sentence_transformers.CrossEncoder")
    def test_rerank_sorts_by_score(self, mock_ce_class):
        from src.retrieval.reranker import CrossEncoderReranker

        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.3, 0.9, 0.1])
        mock_ce_class.return_value = mock_model

        reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
        reranker._model = mock_model
        reranker._model_name = "mock"

        chunks = [make_chunk(f"c{i}", f"text {i}") for i in range(3)]
        results = [SearchResult(chunk=c, score=0.5, rank=i+1) for i, c in enumerate(chunks)]

        ranked = reranker.rerank("query", results)
        scores = [r.score for r in ranked]
        assert scores == sorted(scores, reverse=True)

    @patch("sentence_transformers.CrossEncoder")
    def test_rerank_empty_returns_empty(self, mock_ce_class):
        from src.retrieval.reranker import CrossEncoderReranker

        mock_model = MagicMock()
        mock_ce_class.return_value = mock_model

        reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
        reranker._model = mock_model

        assert reranker.rerank("query", []) == []
