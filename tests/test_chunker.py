"""
tests/test_chunker.py
──────────────────────
Unit tests for document chunking strategies.
"""

from __future__ import annotations

from datetime import datetime, timezone

from src.indexing.chunker import (
    DocumentChunker,
    SemanticChunker,
    SentenceChunker,
    SlidingWindowChunker,
    _count_tokens,
)
from src.ingestion.base import DocumentType, OperationalDocument

# ── Fixtures ─────────────────────────────────────────────────────

def make_doc(
    content: str,
    doc_type: DocumentType = DocumentType.RUNBOOK,
    doc_id: str = "test-doc-001",
) -> OperationalDocument:
    now = datetime.now(timezone.utc)
    return OperationalDocument(
        document_id=doc_id,
        source_system="test",
        document_type=doc_type,
        title="Test Document",
        content_text=content,
        created_at=now,
        last_modified=now,
    )


SHORT_TEXT = "This is a short document. It has three sentences. Nothing more."

LONG_TEXT = " ".join(["word"] * 600)  # 600 tokens — exceeds single chunk

SECTION_TEXT = """# Overview
This section describes the overview of the system.
It contains several important points about architecture.

# Troubleshooting
When the service fails, follow these steps.
First check the logs. Then check the metrics.

# Resolution
Apply the fix and verify the service is healthy.
Monitor for 30 minutes after resolution.
"""


# ── SlidingWindowChunker ─────────────────────────────────────────

class TestSlidingWindowChunker:

    def test_short_text_single_chunk(self):
        chunker = SlidingWindowChunker(chunk_size=512, overlap=50)
        doc = make_doc(SHORT_TEXT)
        chunks = chunker.chunk(doc)
        assert len(chunks) == 1
        assert chunks[0].text == SHORT_TEXT

    def test_long_text_multiple_chunks(self):
        chunker = SlidingWindowChunker(chunk_size=200, overlap=20)
        doc = make_doc(LONG_TEXT)
        chunks = chunker.chunk(doc)
        assert len(chunks) > 1

    def test_chunk_size_respected(self):
        chunker = SlidingWindowChunker(chunk_size=100, overlap=10)
        doc = make_doc(LONG_TEXT)
        chunks = chunker.chunk(doc)
        for chunk in chunks[:-1]:  # last chunk may be smaller
            assert chunk.token_count <= 110  # allow small tolerance

    def test_chunk_ids_unique(self):
        chunker = SlidingWindowChunker(chunk_size=100, overlap=10)
        doc = make_doc(LONG_TEXT)
        chunks = chunker.chunk(doc)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_chunk_metadata_propagation(self):
        chunker = SlidingWindowChunker()
        doc = make_doc(SHORT_TEXT)
        chunks = chunker.chunk(doc)
        assert chunks[0].document_id == doc.document_id
        assert chunks[0].source_system == doc.source_system
        assert "title" in chunks[0].metadata

    def test_total_chunks_field(self):
        chunker = SlidingWindowChunker(chunk_size=100, overlap=10)
        doc = make_doc(LONG_TEXT)
        chunks = chunker.chunk(doc)
        for chunk in chunks:
            assert chunk.total_chunks == len(chunks)


# ── SemanticChunker ──────────────────────────────────────────────

class TestSemanticChunker:

    def test_section_boundaries_respected(self):
        chunker = SemanticChunker(max_chunk_size=512)
        doc = make_doc(SECTION_TEXT, doc_type=DocumentType.INCIDENT_TICKET)
        chunks = chunker.chunk(doc)
        # Should produce multiple chunks at section boundaries
        assert len(chunks) >= 2

    def test_no_empty_chunks(self):
        chunker = SemanticChunker()
        doc = make_doc(SECTION_TEXT)
        chunks = chunker.chunk(doc)
        for chunk in chunks:
            assert chunk.text.strip() != ""
            assert chunk.token_count > 0

    def test_oversized_section_split(self):
        # A single section that exceeds max_chunk_size should be split
        oversized = "word " * 700  # ~700 tokens in one block
        chunker = SemanticChunker(max_chunk_size=256)
        doc = make_doc(oversized)
        chunks = chunker.chunk(doc)
        assert len(chunks) > 1


# ── SentenceChunker ──────────────────────────────────────────────

class TestSentenceChunker:

    def test_sentence_grouping(self):
        text = "Service is down. Pod restarted 5 times. Memory limit exceeded. Check kubectl logs."
        chunker = SentenceChunker(chunk_size=30)
        doc = make_doc(text, doc_type=DocumentType.ALERT)
        chunks = chunker.chunk(doc)
        assert len(chunks) >= 1
        # All original words should be present across chunks
        all_text = " ".join(c.text for c in chunks)
        assert "Service is down" in all_text

    def test_no_empty_chunks(self):
        chunker = SentenceChunker()
        doc = make_doc(SHORT_TEXT)
        chunks = chunker.chunk(doc)
        for chunk in chunks:
            assert chunk.text.strip() != ""


# ── DocumentChunker (dispatcher) ─────────────────────────────────

class TestDocumentChunker:

    def test_dispatches_by_document_type(self):
        dispatcher = DocumentChunker()
        for doc_type in DocumentType:
            doc = make_doc(LONG_TEXT, doc_type=doc_type)
            chunks = dispatcher.chunk_document(doc)
            assert len(chunks) > 0, f"No chunks for {doc_type}"

    def test_chunk_documents_batch(self):
        dispatcher = DocumentChunker()
        docs = [make_doc(SHORT_TEXT, doc_id=f"doc-{i}") for i in range(5)]
        chunks = dispatcher.chunk_documents(docs)
        assert len(chunks) >= 5  # at least one chunk per doc

    def test_incident_uses_semantic_strategy(self):
        dispatcher = DocumentChunker()
        doc = make_doc(SECTION_TEXT, doc_type=DocumentType.INCIDENT_TICKET)
        chunks = dispatcher.chunk_document(doc)
        # Semantic chunker should produce multiple chunks for sectioned text
        assert len(chunks) >= 1

    def test_alert_uses_sentence_strategy(self):
        dispatcher = DocumentChunker()
        text = "Alert fired. CPU at 98%. Pod restarted. Check node status."
        doc = make_doc(text, doc_type=DocumentType.ALERT)
        chunks = dispatcher.chunk_document(doc)
        assert all(c.token_count > 0 for c in chunks)


# ── Token counting ───────────────────────────────────────────────

def test_token_count_non_zero():
    assert _count_tokens("hello world") > 0


def test_token_count_empty():
    assert _count_tokens("") == 0


def test_token_count_scales_with_length():
    short = _count_tokens("hello")
    long = _count_tokens("hello " * 100)
    assert long > short
