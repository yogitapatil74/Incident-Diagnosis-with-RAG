"""
src/retrieval/reranker.py
──────────────────────────
Cross-encoder re-ranking using ms-marco-MiniLM-L6-v2.
Scores query-document pairs more precisely than cosine similarity.
Replaces FAISS scores with cross-encoder scores on the top-k candidates.
"""

from __future__ import annotations

import numpy as np
import structlog

from src.indexing.vector_store import SearchResult

logger = structlog.get_logger(__name__)


class CrossEncoderReranker:
    """
    Re-ranks retrieved candidates using a cross-encoder model.
    ms-marco-MiniLM-L6-v2 is a lightweight (22M param) model that achieves
    strong relevance scoring for query-passage pairs.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder
        logger.info("Loading cross-encoder reranker", model=model_name)
        self._model = CrossEncoder(model_name, max_length=512)
        self._model_name = model_name

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        max_length: int = 512,
    ) -> list[SearchResult]:
        """
        Re-rank results by cross-encoder relevance score.
        Returns results sorted by score descending (mutates score field).
        """
        if not results:
            return results

        pairs = [(query, r.chunk.text[:max_length]) for r in results]
        raw_scores = self._model.predict(pairs)

        # Normalise scores to [0, 1] using sigmoid
        scores = self._sigmoid(raw_scores)

        for result, score in zip(results, scores):
            result.score = float(score)

        results.sort(key=lambda r: r.score, reverse=True)

        logger.debug(
            "Re-ranking complete",
            num_results=len(results),
            top_score=round(results[0].score, 3) if results else 0,
        )
        return results

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float32)))
