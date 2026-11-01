"""
src/indexing/vector_store.py
─────────────────────────────
FAISS-based vector store with HNSW index.
Implements a blue-green atomic index swap for zero-downtime re-indexing.
Supports hybrid retrieval: dense ANN + structured metadata filters.
"""

from __future__ import annotations

import os
import pickle
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
import structlog

from src.indexing.chunker import DocumentChunk

logger = structlog.get_logger(__name__)


@dataclass
class SearchResult:
    """A single retrieval result from the vector store."""
    chunk: DocumentChunk
    score: float          # Cosine similarity (0-1, higher = more relevant)
    rank: int

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk.chunk_id,
            "document_id": self.chunk.document_id,
            "document_type": self.chunk.document_type.value,
            "text": self.chunk.text,
            "score": round(self.score, 4),
            "rank": self.rank,
            "metadata": self.chunk.metadata,
        }


class FAISSVectorStore:
    """
    FAISS HNSW vector store for semantic similarity search.

    Index configuration:
      - Index type: HNSW (Hierarchical Navigable Small World)
      - M = 32 neighbours per node
      - ef_construction = 400
      - Similarity metric: cosine (inner product on L2-normalised vectors)

    Blue-green swap ensures zero-downtime re-indexing.
    """

    def __init__(self, config: dict):
        self.config = config
        self.index_path = Path(config.get("index_path", "data/faiss_index"))
        self.dimensions = config.get("dimensions", 1536)
        self.hnsw_m = config.get("hnsw_m", 32)
        self.hnsw_ef_construction = config.get("hnsw_ef_construction", 400)
        self.hnsw_ef_search = config.get("hnsw_ef_search", 64)

        self._lock = threading.RLock()
        self._index: faiss.IndexHNSWFlat | None = None
        self._chunks: list[DocumentChunk] = []  # parallel array to FAISS index
        self._id_to_position: dict[str, int] = {}

        os.makedirs(self.index_path, exist_ok=True)

        # Load existing index if present
        if self._index_files_exist():
            self._load_index()
        else:
            self._build_empty_index()

    # ── Index Construction ─────────────────────────────────────

    def _build_empty_index(self) -> None:
        """Create a fresh HNSW index with the configured parameters."""
        # HNSW index using Inner Product (for cosine after L2 normalisation)
        index = faiss.IndexHNSWFlat(self.dimensions, self.hnsw_m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = self.hnsw_ef_construction
        index.hnsw.efSearch = self.hnsw_ef_search
        self._index = index
        self._chunks = []
        self._id_to_position = {}
        logger.info("Created empty FAISS HNSW index", dims=self.dimensions, M=self.hnsw_m)

    def _index_files_exist(self) -> bool:
        return (
            (self.index_path / "index.faiss").exists()
            and (self.index_path / "chunks.pkl").exists()
        )

    def _load_index(self) -> None:
        """Load existing index from disk."""
        index = faiss.read_index(str(self.index_path / "index.faiss"))
        if index.d != self.dimensions:
            # Typically happens after switching embedding provider/model
            # (e.g. OpenAI 1536-dim -> local 768-dim) while the old index is
            # still on disk / in the Docker volume. Searching it would hit a
            # FAISS assertion on every query, so start empty instead and ask
            # for a rebuild.
            logger.error(
                "Persisted FAISS index dimension does not match the embedding "
                "model — ignoring it. Re-run indexing with --rebuild "
                "(or FORCE_REINDEX=true in Docker).",
                index_dims=index.d,
                expected_dims=self.dimensions,
                index_path=str(self.index_path),
            )
            self._build_empty_index()
            return
        with open(self.index_path / "chunks.pkl", "rb") as f:
            chunks = pickle.load(f)
        if len(chunks) != index.ntotal:
            logger.warning(
                "FAISS index and chunk store are out of sync",
                vectors=index.ntotal,
                chunks=len(chunks),
            )
        index.hnsw.efSearch = self.hnsw_ef_search
        self._index = index
        self._chunks = chunks
        self._id_to_position = {c.chunk_id: i for i, c in enumerate(self._chunks)}
        logger.info(
            "Loaded FAISS index",
            vectors=self._index.ntotal,
            chunks=len(self._chunks),
        )

    def _save_index(self, path: Path) -> None:
        """Persist the live index to disk."""
        self._save_index_raw(self._index, self._chunks, path)

    # ── Blue-Green Atomic Index Swap ──────────────────────────

    def rebuild_index(
        self,
        chunk_embedding_pairs: list[tuple[DocumentChunk, np.ndarray]],
    ) -> None:
        """
        Build a new index from scratch and atomically hot-swap it.
        Zero-downtime: existing index serves queries until swap completes.
        """
        logger.info("Starting index rebuild", total_chunks=len(chunk_embedding_pairs))
        green_path = self.index_path / "green"

        # Build new index in background directory
        new_index = faiss.IndexHNSWFlat(self.dimensions, self.hnsw_m, faiss.METRIC_INNER_PRODUCT)
        new_index.hnsw.efConstruction = self.hnsw_ef_construction
        new_index.hnsw.efSearch = self.hnsw_ef_search

        new_chunks = []
        vectors = []

        seen: set[str] = set()
        for chunk, embedding in chunk_embedding_pairs:
            if chunk.chunk_id in seen:
                continue  # keep the index and chunk list 1:1 and unique
            seen.add(chunk.chunk_id)
            normalised = self._normalise(np.asarray(embedding, dtype=np.float32))
            vectors.append(normalised)
            new_chunks.append(chunk)

        if vectors:
            matrix = np.vstack(vectors).astype(np.float32)
            new_index.add(matrix)

        # Save to green directory
        self._save_index_raw(new_index, new_chunks, green_path)

        # Atomic swap under lock
        with self._lock:
            self._index = new_index
            self._chunks = new_chunks
            self._id_to_position = {c.chunk_id: i for i, c in enumerate(new_chunks)}
            # Promote green → live by atomically renaming the files
            for name in ("index.faiss", "chunks.pkl"):
                os.replace(green_path / name, self.index_path / name)

        # Cleanup
        shutil.rmtree(green_path, ignore_errors=True)
        logger.info("Index rebuild complete", total_vectors=new_index.ntotal)

    def add_chunks(
        self,
        chunk_embedding_pairs: list[tuple[DocumentChunk, np.ndarray]],
    ) -> None:
        """Incrementally add chunks to an existing live index."""
        added = 0
        with self._lock:
            for chunk, embedding in chunk_embedding_pairs:
                if chunk.chunk_id in self._id_to_position:
                    continue  # Skip duplicates
                normalised = self._normalise(
                    np.asarray(embedding, dtype=np.float32)
                ).reshape(1, -1).astype(np.float32)
                self._index.add(normalised)
                self._id_to_position[chunk.chunk_id] = len(self._chunks)
                self._chunks.append(chunk)
                added += 1

            self._save_index(self.index_path)

        logger.info(
            "Incremental index update",
            added=added,
            skipped_duplicates=len(chunk_embedding_pairs) - added,
            total=self._index.ntotal,
        )

    # ── Search ─────────────────────────────────────────────────

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        """
        Perform ANN search. Optionally filter results by metadata predicates.

        filters example: {"service": "payments-service", "priority": "P1"}
        """
        with self._lock:
            if self._index is None or self._index.ntotal == 0:
                logger.warning("Search on empty index")
                return []

            query = self._normalise(
                np.asarray(query_embedding, dtype=np.float32)
            ).reshape(1, -1).astype(np.float32)
            if query.shape[1] != self._index.d:
                raise ValueError(
                    f"Query embedding has {query.shape[1]} dimensions but the "
                    f"index expects {self._index.d}. Was the index built with a "
                    "different embedding model? Re-index with --rebuild."
                )
            # Fetch extra candidates if filtering is active
            fetch_k = min(top_k * 3 if filters else top_k, self._index.ntotal)
            scores, indices = self._index.search(query, fetch_k)
            # Snapshot the chunk list that matches *this* index. Reading
            # self._chunks after releasing the lock raced with
            # rebuild_index(), which could swap in a different list and map
            # FAISS positions to the wrong (or missing) chunks.
            chunks = self._chunks

        results = []
        for raw_score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(chunks):
                continue
            chunk = chunks[idx]
            if filters and not self._matches_filters(chunk, filters):
                continue
            # `raw_score` is the FAISS inner product on L2-normalised
            # vectors, i.e. raw cosine similarity in [-1, 1]. Rescale to
            # [0, 1] so SearchResult.score honours its documented contract
            # (callers, e.g. the reranker's threshold filter and the
            # confidence display, assume a [0, 1] relevance score).
            score = max(0.0, min(1.0, (float(raw_score) + 1.0) / 2.0))
            results.append(SearchResult(chunk=chunk, score=score, rank=len(results) + 1))
            if len(results) >= top_k:
                break

        return results

    @staticmethod
    def _matches_filters(chunk: DocumentChunk, filters: dict) -> bool:
        """
        Check if a chunk's metadata satisfies all filter predicates.

        A chunk that doesn't carry a filter key at all is *not* excluded:
        runbooks, post-mortems and docs have no `service`/`environment`
        metadata (only incident tickets do). Treating a missing key as a
        mismatch meant any request with a service or environment set could
        only ever retrieve tickets for that exact service — every runbook
        and post-mortem was silently filtered out.
        """
        for key, value in filters.items():
            if key not in chunk.metadata:
                continue
            chunk_val = chunk.metadata.get(key)
            if isinstance(value, list):
                if chunk_val not in value:
                    return False
            elif chunk_val != value:
                return False
        return True

    @staticmethod
    def _normalise(vec: np.ndarray) -> np.ndarray:
        """L2-normalise a vector for cosine similarity via inner product."""
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    @staticmethod
    def _save_index_raw(
        index: faiss.IndexHNSWFlat,
        chunks: list[DocumentChunk],
        path: Path,
    ) -> None:
        # Write to temp files then os.replace(), so a crash mid-write never
        # leaves a truncated/mismatched index.faiss + chunks.pkl pair behind.
        os.makedirs(path, exist_ok=True)
        tmp_index = path / "index.faiss.tmp"
        tmp_chunks = path / "chunks.pkl.tmp"
        faiss.write_index(index, str(tmp_index))
        with open(tmp_chunks, "wb") as f:
            pickle.dump(chunks, f)
        os.replace(tmp_index, path / "index.faiss")
        os.replace(tmp_chunks, path / "chunks.pkl")

    # ── Stats ──────────────────────────────────────────────────

    @property
    def total_vectors(self) -> int:
        return self._index.ntotal if self._index else 0

    @property
    def total_chunks(self) -> int:
        return len(self._chunks)

    def stats(self) -> dict:
        return {
            "total_vectors": self.total_vectors,
            "total_chunks": self.total_chunks,
            "index_type": "HNSW",
            "dimensions": self.dimensions,
            "hnsw_m": self.hnsw_m,
        }
