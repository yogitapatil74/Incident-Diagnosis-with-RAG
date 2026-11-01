"""
src/indexing/embedder.py
─────────────────────────
Embedding generation for document chunks.
Supports OpenAI text-embedding-ada-002 (1536-dim) and
local sentence-transformers models for offline/cost-sensitive use.
Implements batching and Redis-based caching for efficiency.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Protocol

import numpy as np
import structlog

from src.indexing.chunker import DocumentChunk

logger = structlog.get_logger(__name__)


class EmbeddingProvider(Protocol):
    """Protocol for embedding backend implementations."""
    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...
    @property
    def dimensions(self) -> int: ...


class OpenAIEmbeddingProvider:
    """
    OpenAI embedding provider (text-embedding-ada-002 / text-embedding-3-*).
    Dense vectors, cosine similarity.
    """

    # Native output dimensions of the OpenAI embedding models. Used to size
    # the FAISS index correctly — previously this was hard-coded to 1536,
    # so configuring e.g. text-embedding-3-large (3072-dim) produced a
    # vector-store/embedding dimension mismatch and a FAISS assertion error.
    KNOWN_DIMENSIONS = {
        "text-embedding-ada-002": 1536,
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
    }

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-ada-002",
        dimensions: int | None = None,
    ):
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key)
        self._model = model
        self._dims = self.KNOWN_DIMENSIONS.get(model) or dimensions or 1536

    @property
    def dimensions(self) -> int:
        return self._dims

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Handles OpenAI rate limits with retry."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self._client.embeddings.create(
                    model=self._model, input=texts
                )
                return [item.embedding for item in response.data]
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                wait = 2 ** attempt
                logger.warning(
                    "Embedding API error, retrying",
                    attempt=attempt + 1,
                    wait=wait,
                    error=str(e),
                )
                time.sleep(wait)


class LocalEmbeddingProvider:
    """
    Local sentence-transformers embedding provider.
    No API key required — good for offline PoC runs.
    Default: all-mpnet-base-v2 (768-dim)
    """

    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        from sentence_transformers import SentenceTransformer
        logger.info("Loading local embedding model", model=model_name)
        self._model = SentenceTransformer(model_name)
        self._dims = self._model.get_sentence_embedding_dimension()

    @property
    def dimensions(self) -> int:
        return self._dims

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return embeddings.tolist()


class EmbeddingCache:
    """
    Redis-backed embedding cache using content hash as key.
    Falls back to in-memory dict cache if Redis is unavailable.
    """

    def __init__(self, config: dict):
        self._ttl = config.get("ttl_seconds", 3600)
        self._memory_cache: dict[str, list[float]] = {}
        self._redis = None

        if config.get("enabled", False):
            try:
                import redis
                self._redis = redis.Redis(
                    host=config.get("host", "localhost"),
                    port=config.get("port", 6379),
                    password=config.get("password"),
                    decode_responses=False,
                )
                self._redis.ping()
                logger.info("Redis cache connected")
            except Exception as e:
                logger.warning("Redis unavailable, using memory cache", error=str(e))
                self._redis = None

    def _key(self, text: str) -> str:
        return f"emb:{hashlib.sha256(text.encode()).hexdigest()}"

    def get(self, text: str) -> list[float] | None:
        key = self._key(text)
        # Memory cache first
        if key in self._memory_cache:
            return self._memory_cache[key]
        # Redis fallback
        if self._redis:
            try:
                raw = self._redis.get(key)
                if raw:
                    emb = json.loads(raw)
                    self._memory_cache[key] = emb  # promote to memory
                    return emb
            except Exception:
                pass
        return None

    def set(self, text: str, embedding: list[float]) -> None:
        key = self._key(text)
        self._memory_cache[key] = embedding
        if self._redis:
            try:
                self._redis.setex(key, self._ttl, json.dumps(embedding))
            except Exception:
                pass


class Embedder:
    """
    Main embedding orchestrator.
    Handles batching, caching, and provider selection.
    """

    def __init__(self, config: dict):
        self.config = config
        self._provider = self._build_provider(config)
        self._cache = EmbeddingCache(config.get("cache", {}))
        self._batch_size = config.get("batch_size", 100)

    @classmethod
    def from_app_config(cls, app_config: dict) -> "Embedder":
        """
        Build an Embedder from the full application config (config.yaml).

        The embedding section must take precedence over the LLM section.
        Callers previously did ``Embedder({**cfg["embedding"], **cfg["llm"]})``,
        which let ``llm.provider`` / ``llm.model`` overwrite the embedding
        provider and model. Consequences: ``--local-embeddings`` was silently
        ignored, OpenAI was asked to embed with the chat model
        (``gpt-4-turbo``), and ``llm.provider: anthropic`` crashed startup
        with "Unknown embedding provider: anthropic". Only the provider
        credentials are borrowed from the LLM section now.
        """
        llm_cfg = app_config.get("llm", {}) or {}
        return cls({
            **(app_config.get("embedding", {}) or {}),
            "openai": llm_cfg.get("openai", {}) or {},
            "cache": app_config.get("cache", {}) or {},
        })

    def _build_provider(self, config: dict) -> EmbeddingProvider:
        provider = config.get("provider", "openai")
        if provider == "openai":
            return OpenAIEmbeddingProvider(
                api_key=(config.get("openai") or {}).get("api_key", ""),
                model=config.get("model", "text-embedding-ada-002"),
                dimensions=config.get("dimensions"),
            )
        elif provider == "local":
            return LocalEmbeddingProvider(
                model_name=config.get("local_model", "sentence-transformers/all-mpnet-base-v2")
            )
        else:
            raise ValueError(f"Unknown embedding provider: {provider}")

    @property
    def dimensions(self) -> int:
        return self._provider.dimensions

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text string."""
        cached = self._cache.get(text)
        if cached:
            return np.array(cached, dtype=np.float32)
        embedding = self._provider.embed_batch([text])[0]
        self._cache.set(text, embedding)
        return np.array(embedding, dtype=np.float32)

    def embed_chunks(self, chunks: list[DocumentChunk]) -> list[tuple[DocumentChunk, np.ndarray]]:
        """
        Embed all chunks in batches.
        Returns list of (chunk, embedding_vector) tuples.
        """
        results: list[tuple[DocumentChunk, np.ndarray]] = []
        # Separate cached from uncached
        to_embed: list[tuple[int, DocumentChunk]] = []
        for i, chunk in enumerate(chunks):
            cached = self._cache.get(chunk.text)
            if cached:
                results.append((chunk, np.array(cached, dtype=np.float32)))
            else:
                to_embed.append((i, chunk))

        # Embed uncached in batches
        for batch_start in range(0, len(to_embed), self._batch_size):
            batch = to_embed[batch_start: batch_start + self._batch_size]
            texts = [chunk.text for _, chunk in batch]

            logger.info(
                "Embedding batch",
                batch_num=batch_start // self._batch_size + 1,
                batch_size=len(texts),
                total_remaining=len(to_embed) - batch_start,
            )

            embeddings = self._provider.embed_batch(texts)
            for (_, chunk), emb in zip(batch, embeddings):
                self._cache.set(chunk.text, emb)
                results.append((chunk, np.array(emb, dtype=np.float32)))

        logger.info(
            "Embedding complete",
            total_chunks=len(chunks),
            cache_hits=len(chunks) - len(to_embed),
        )
        return results
