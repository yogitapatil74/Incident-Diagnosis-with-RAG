"""
src/indexing/chunker.py
────────────────────────
Document chunking strategies for the knowledge indexing pipeline.
Three strategies:
  1. SlidingWindowChunker  — fixed window with overlap (runbooks, docs)
  2. SemanticChunker       — section/paragraph-aware (tickets, post-mortems)
  3. SentenceChunker       — sentence-level (alerts, short-form notes)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol

import structlog
import tiktoken

from src.ingestion.base import DocumentType, OperationalDocument

logger = structlog.get_logger(__name__)

# Tokeniser used for chunk size measurement (same model as ada-002).
# Loaded lazily (see `_get_tokeniser`) rather than at import time.
_TOKENISER: "tiktoken.Encoding | None" = None


def _get_tokeniser() -> "tiktoken.Encoding":
    """Lazily load and cache the tiktoken encoding.

    tiktoken downloads its BPE merge file from a remote CDN the first time
    an encoding is requested, then caches it locally (respects
    TIKTOKEN_CACHE_DIR). Loading it at import time meant simply importing
    this module — even for code paths that never chunk text — required
    network access and failed hard in offline/sandboxed environments. Now
    the encoding is only fetched the first time it's actually needed.
    """
    global _TOKENISER
    if _TOKENISER is None:
        try:
            _TOKENISER = tiktoken.get_encoding("cl100k_base")
        except Exception as exc:
            raise RuntimeError(
                "Failed to load the tiktoken 'cl100k_base' encoding, which "
                "requires network access on first use (the result is then "
                "cached locally). If you're running offline, pre-warm the "
                "cache on a machine with internet access, or set "
                "TIKTOKEN_CACHE_DIR to a directory containing the cached "
                "encoding."
            ) from exc
    return _TOKENISER


@dataclass
class DocumentChunk:
    """A single chunk derived from an OperationalDocument."""
    chunk_id: str
    document_id: str
    source_system: str
    document_type: DocumentType
    text: str
    token_count: int
    chunk_index: int
    total_chunks: int
    # Inherited from parent document
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "source_system": self.source_system,
            "document_type": self.document_type.value,
            "text": self.text,
            "token_count": self.token_count,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "metadata": self.metadata,
        }


class ChunkingStrategy(Protocol):
    """Protocol for all chunking strategy implementations."""
    def chunk(self, document: OperationalDocument) -> list[DocumentChunk]: ...


def _count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(_get_tokeniser().encode(text))


def _make_chunk_id(document_id: str, index: int) -> str:
    return f"{document_id}_chunk_{index:04d}"


class SlidingWindowChunker:
    """
    Fixed sliding-window chunking with configurable size and overlap.
    Best for long-form documents: runbooks, documentation.
    """

    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        # overlap >= chunk_size makes the window step <= 0, which previously
        # caused an infinite loop (step 0) or an endless backwards walk.
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if not 0 <= overlap < chunk_size:
            raise ValueError(
                f"overlap must be >= 0 and < chunk_size ({chunk_size}), got {overlap}"
            )
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, document: OperationalDocument) -> list[DocumentChunk]:
        return self._to_chunk_objects(document, self._chunk_text(document.content_text))

    def _chunk_text(self, text: str) -> list[str]:
        """Split raw text into overlapping token windows."""
        tokeniser = _get_tokeniser()
        tokens = tokeniser.encode(text)
        chunks: list[str] = []
        start = 0
        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunks.append(tokeniser.decode(tokens[start:end]))
            if end == len(tokens):
                break
            start += self.chunk_size - self.overlap
        return chunks

    @staticmethod
    def _to_chunk_objects(
        document: OperationalDocument, texts: list[str]
    ) -> list[DocumentChunk]:
        total = len(texts)
        return [
            DocumentChunk(
                chunk_id=_make_chunk_id(document.document_id, i),
                document_id=document.document_id,
                source_system=document.source_system,
                document_type=document.document_type,
                text=text,
                token_count=_count_tokens(text),
                chunk_index=i,
                total_chunks=total,
                metadata={
                    **document.metadata,
                    "title": document.title,
                    "created_at": document.created_at.isoformat(),
                    "last_modified": document.last_modified.isoformat(),
                    "chunking_strategy": "sliding_window",
                },
            )
            for i, text in enumerate(texts)
        ]


class SemanticChunker:
    """
    Section-aware chunking that treats Markdown headings as hard semantic
    boundaries: content under different top-level headings is never merged
    into the same chunk, so retrieval doesn't blend unrelated sections of
    an incident ticket or post-mortem together.

    Within a single heading-delimited section, paragraphs are packed
    together up to `max_chunk_size`, and any section too large to fit in
    one chunk is split with a sliding window. `min_chunk_size` only
    controls folding an undersized trailing paragraph fragment into its
    neighbour *within* a section — it never merges across a heading
    boundary and never drops content.

    Best for incident tickets and post-mortems.
    """

    # Hard boundary: split immediately before a Markdown heading.
    HEADING_PATTERN = re.compile(r"(?=^#{1,3}\s)", re.MULTILINE)
    # Soft boundary used only when splitting an oversized section.
    PARAGRAPH_PATTERN = re.compile(r"\n{2,}")

    def __init__(self, max_chunk_size: int = 512, min_chunk_size: int = 50):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size

    def chunk(self, document: OperationalDocument) -> list[DocumentChunk]:
        sections = self.HEADING_PATTERN.split(document.content_text)
        sections = [s.strip() for s in sections if s.strip()]
        if not sections:
            return []

        pieces: list[str] = []
        for section in sections:
            pieces.extend(self._split_section(section))

        return SlidingWindowChunker._to_chunk_objects(document, pieces)

    def _split_section(self, section: str) -> list[str]:
        """Return one chunk for `section`, or several if it's oversized."""
        if _count_tokens(section) <= self.max_chunk_size:
            return [section]

        paragraphs = [
            p.strip() for p in self.PARAGRAPH_PATTERN.split(section) if p.strip()
        ]
        packed = self._merge_and_split(paragraphs)

        # Fold an undersized trailing fragment into the previous chunk
        # instead of emitting a near-empty final chunk.
        if len(packed) > 1 and _count_tokens(packed[-1]) < self.min_chunk_size:
            candidate = f"{packed[-2]}\n\n{packed[-1]}"
            if _count_tokens(candidate) <= self.max_chunk_size:
                packed[-2:] = [candidate]

        return packed

    def _merge_and_split(self, sections: list[str]) -> list[str]:
        """Greedily pack `sections` up to max_chunk_size, splitting any
        individually-oversized section with a sliding window. Never drops
        content (unlike the packing loop's earlier revision, which
        silently discarded any trailing fragment under min_chunk_size)."""
        result: list[str] = []
        buffer = ""

        for section in sections:
            section_tokens = _count_tokens(section)

            if section_tokens > self.max_chunk_size:
                if buffer:
                    result.append(buffer.strip())
                    buffer = ""
                result.extend(
                    SlidingWindowChunker(
                        chunk_size=self.max_chunk_size,
                        overlap=min(30, self.max_chunk_size // 4),
                    )._chunk_text(section)
                )
                continue

            candidate = f"{buffer}\n\n{section}".strip() if buffer else section
            if _count_tokens(candidate) <= self.max_chunk_size:
                buffer = candidate
            else:
                if buffer:
                    result.append(buffer.strip())
                buffer = section

        if buffer:
            result.append(buffer.strip())

        return result


class SentenceChunker:
    """
    Sentence-level chunking for short-form content such as alerts and notes.
    Groups consecutive sentences until chunk_size is reached.
    """

    # Simple sentence boundary pattern
    SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")

    def __init__(self, chunk_size: int = 256):
        self.chunk_size = chunk_size

    def chunk(self, document: OperationalDocument) -> list[DocumentChunk]:
        sentences = self.SENTENCE_BOUNDARY.split(document.content_text)
        sentences = [s.strip() for s in sentences if s.strip()]

        groups = []
        current = []
        current_tokens = 0

        for sentence in sentences:
            st = _count_tokens(sentence)
            if current_tokens + st > self.chunk_size and current:
                groups.append(" ".join(current))
                current = [sentence]
                current_tokens = st
            else:
                current.append(sentence)
                current_tokens += st

        if current:
            groups.append(" ".join(current))

        return SlidingWindowChunker._to_chunk_objects(document, groups)


class DocumentChunker:
    """
    Main chunker that dispatches to the appropriate strategy
    based on document type.
    """

    STRATEGY_MAP: dict[DocumentType, type] = {
        DocumentType.INCIDENT_TICKET: SemanticChunker,
        DocumentType.POST_MORTEM: SemanticChunker,
        DocumentType.RUNBOOK: SlidingWindowChunker,
        DocumentType.DOCUMENTATION: SlidingWindowChunker,
        DocumentType.ALERT: SentenceChunker,
        DocumentType.UNKNOWN: SlidingWindowChunker,
    }

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        chunk_size = self.config.get("chunk_size", 512)
        overlap = self.config.get("chunk_overlap", 50)

        self._strategies = {
            DocumentType.INCIDENT_TICKET: SemanticChunker(max_chunk_size=chunk_size),
            DocumentType.POST_MORTEM: SemanticChunker(max_chunk_size=chunk_size),
            DocumentType.RUNBOOK: SlidingWindowChunker(chunk_size=chunk_size, overlap=overlap),
            DocumentType.DOCUMENTATION: SlidingWindowChunker(chunk_size=chunk_size, overlap=overlap),
            DocumentType.ALERT: SentenceChunker(chunk_size=256),
            DocumentType.UNKNOWN: SlidingWindowChunker(chunk_size=chunk_size, overlap=overlap),
        }

    def chunk_document(self, document: OperationalDocument) -> list[DocumentChunk]:
        strategy = self._strategies.get(
            document.document_type,
            self._strategies[DocumentType.UNKNOWN]
        )
        chunks = strategy.chunk(document)
        logger.debug(
            "Document chunked",
            document_id=document.document_id,
            doc_type=document.document_type.value,
            strategy=type(strategy).__name__,
            num_chunks=len(chunks),
        )
        return chunks

    def chunk_documents(self, documents: list[OperationalDocument]) -> list[DocumentChunk]:
        all_chunks = []
        for doc in documents:
            all_chunks.extend(self.chunk_document(doc))
        logger.info(
            "Chunking complete",
            documents=len(documents),
            chunks=len(all_chunks),
        )
        return all_chunks
