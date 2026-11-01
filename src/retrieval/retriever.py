"""
src/retrieval/retriever.py
───────────────────────────
Semantic retrieval engine combining:
  1. FAISS ANN search (dense retrieval)
  2. Cross-encoder re-ranking (ms-marco-MiniLM)
  3. HyDE query expansion (when initial results are sparse)
"""

from __future__ import annotations

import structlog

from src.indexing.embedder import Embedder
from src.indexing.vector_store import FAISSVectorStore, SearchResult
from src.retrieval.query_expander import HyDEQueryExpander
from src.retrieval.reranker import CrossEncoderReranker

logger = structlog.get_logger(__name__)


class SemanticRetriever:
    """
    Main retrieval engine.

    Pipeline per query:
      1. Embed query → query vector
      2. FAISS top-k ANN search (with optional metadata filters)
      3. Cross-encoder re-ranking
      4. Threshold filtering
      5. Optional HyDE expansion if results < min_docs threshold
    """

    def __init__(
        self,
        vector_store: FAISSVectorStore,
        embedder: Embedder,
        reranker: CrossEncoderReranker,
        query_expander: HyDEQueryExpander | None = None,
        config: dict | None = None,
    ):
        self.vector_store = vector_store
        self.embedder = embedder
        self.reranker = reranker
        self.query_expander = query_expander
        cfg = config or {}
        self.top_k = cfg.get("top_k", 10)
        self.rerank_top_k = cfg.get("rerank_top_k", 5)
        self.rerank_threshold = cfg.get("rerank_threshold", 0.45)
        self.min_docs_before_expansion = cfg.get("min_docs_before_expansion", 3)
        self.hyde_enabled = cfg.get("hyde_enabled", True)
        # Widen to an unfiltered search when metadata filters leave too few
        # candidates (e.g. the first incident ever seen for a service).
        self.filter_fallback = cfg.get("filter_fallback", True)

    def retrieve(
        self,
        query: str,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        """
        Full retrieval pipeline for a query string.
        Returns re-ranked, threshold-filtered results.
        """
        logger.info("Starting retrieval", query_preview=query[:100])

        # Step 1: Embed query
        query_embedding = self.embedder.embed_text(query)

        # Step 2: ANN search
        candidates = self.vector_store.search(
            query_embedding, top_k=self.top_k, filters=filters
        )
        if (
            filters
            and self.filter_fallback
            and len(candidates) < self.min_docs_before_expansion
        ):
            # A hard service/environment filter means a new or rarely-seen
            # service gets little or no context — yet similar incidents on
            # *other* services are exactly what helps most. Top up with
            # unfiltered candidates rather than returning near-empty context.
            logger.info(
                "Filtered search sparse — widening to unfiltered search",
                filtered=len(candidates),
                filters=filters,
            )
            candidates = self._merge(
                candidates,
                self.vector_store.search(query_embedding, top_k=self.top_k),
            )
        logger.debug("ANN candidates", count=len(candidates))

        # Step 3: Re-rank
        if candidates:
            candidates = self.reranker.rerank(query, candidates)

        # Step 4: Threshold filter
        passing = [r for r in candidates if r.score >= self.rerank_threshold]
        logger.debug("After threshold filter", count=len(passing), threshold=self.rerank_threshold)

        # Step 5: HyDE expansion if sparse results
        if (
            len(passing) < self.min_docs_before_expansion
            and self.hyde_enabled
            and self.query_expander is not None
        ):
            logger.info(
                "Sparse results — triggering HyDE expansion",
                passing=len(passing),
                min_required=self.min_docs_before_expansion,
            )
            passing = self._hyde_expand(query, filters, existing=passing)

        # Assign final ranks
        for i, result in enumerate(passing[: self.rerank_top_k]):
            result.rank = i + 1

        final = passing[: self.rerank_top_k]
        logger.info("Retrieval complete", results_returned=len(final))
        return final

    def _hyde_expand(
        self,
        query: str,
        filters: dict | None,
        existing: list[SearchResult],
    ) -> list[SearchResult]:
        """
        Hypothetical Document Embedding (HyDE) expansion.
        Generates a synthetic exemplar document and retrieves against it.
        """
        expanded_query = self.query_expander.expand(query)
        expanded_embedding = self.embedder.embed_text(expanded_query)

        new_candidates = self.vector_store.search(
            expanded_embedding, top_k=self.top_k, filters=filters
        )
        merged = self._merge(existing, new_candidates)

        if merged:
            merged = self.reranker.rerank(query, merged)

        return [r for r in merged if r.score >= self.rerank_threshold]

    @staticmethod
    def _merge(
        primary: list[SearchResult], extra: list[SearchResult]
    ) -> list[SearchResult]:
        """Append `extra` results to `primary`, skipping duplicate chunks."""
        seen = {r.chunk.chunk_id for r in primary}
        merged = list(primary)
        for result in extra:
            if result.chunk.chunk_id not in seen:
                merged.append(result)
                seen.add(result.chunk.chunk_id)
        return merged
