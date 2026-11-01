"""
src/ingestion/base.py
─────────────────────
Abstract base class for all data source ingestion connectors.
Each connector normalises source documents into the canonical
OperationalDocument schema used throughout the pipeline.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Iterator

import structlog

logger = structlog.get_logger(__name__)


class DocumentType(str, Enum):
    INCIDENT_TICKET = "incident_ticket"
    RUNBOOK = "runbook"
    POST_MORTEM = "post_mortem"
    DOCUMENTATION = "documentation"
    ALERT = "alert"
    UNKNOWN = "unknown"


@dataclass
class OperationalDocument:
    """
    Canonical document schema used across all pipeline stages.
    Every ingestion source normalises to this format.
    """
    document_id: str
    source_system: str
    document_type: DocumentType
    title: str
    content_text: str
    created_at: datetime
    last_modified: datetime
    metadata: dict = field(default_factory=dict)
    # Computed fields
    content_hash: str = field(default="", init=False)
    word_count: int = field(default=0, init=False)

    def __post_init__(self):
        self.content_hash = hashlib.sha256(
            self.content_text.encode("utf-8")
        ).hexdigest()
        self.word_count = len(self.content_text.split())

    def to_dict(self) -> dict:
        return {
            "document_id": self.document_id,
            "source_system": self.source_system,
            "document_type": self.document_type.value,
            "title": self.title,
            "content_text": self.content_text,
            "created_at": self.created_at.isoformat(),
            "last_modified": self.last_modified.isoformat(),
            "metadata": self.metadata,
            "content_hash": self.content_hash,
            "word_count": self.word_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "OperationalDocument":
        doc = cls(
            document_id=data["document_id"],
            source_system=data["source_system"],
            document_type=DocumentType(data["document_type"]),
            title=data["title"],
            content_text=data["content_text"],
            created_at=datetime.fromisoformat(data["created_at"]),
            last_modified=datetime.fromisoformat(data["last_modified"]),
            metadata=data.get("metadata", {}),
        )
        return doc


class BaseIngester(ABC):
    """
    Abstract base class for all ingestion connectors.

    Subclasses implement:
      - ingest()  → yields OperationalDocument instances
      - validate_connection() → confirms source is reachable
    """

    def __init__(self, config: dict):
        self.config = config
        self.logger = structlog.get_logger(self.__class__.__name__)
        self._ingested_count = 0
        self._error_count = 0

    @abstractmethod
    def ingest(self) -> Iterator[OperationalDocument]:
        """
        Yield OperationalDocument instances from the data source.
        Implementations should handle pagination and rate limiting.
        """
        ...

    @abstractmethod
    def validate_connection(self) -> bool:
        """Return True if the source system is reachable."""
        ...

    def run(self) -> list[OperationalDocument]:
        """
        Execute ingestion and return all documents.
        Wraps ingest() with logging and error handling.
        """
        self.logger.info("Starting ingestion", source=self.__class__.__name__)
        documents = []

        for doc in self.ingest():
            try:
                documents.append(doc)
                self._ingested_count += 1
                if self._ingested_count % 100 == 0:
                    self.logger.info(
                        "Ingestion progress",
                        count=self._ingested_count,
                        errors=self._error_count,
                    )
            except Exception as exc:
                self._error_count += 1
                self.logger.error(
                    "Failed to process document",
                    error=str(exc),
                    exc_info=True,
                )

        self.logger.info(
            "Ingestion complete",
            total=self._ingested_count,
            errors=self._error_count,
        )
        return documents

    @staticmethod
    def _stable_hash(text: str, length: int = 6) -> str:
        """
        Deterministic short content hash for building document IDs.

        Python's built-in hash() is salted per process (PYTHONHASHSEED), so
        IDs built from it changed on every run. Re-running the indexer then
        treated every document as new: incremental indexing appended a full
        duplicate copy of the knowledge base each time (e.g. on every
        FORCE_REINDEX container start), skewing retrieval towards repeats.
        """
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]

    @staticmethod
    def _clean_text(text: str) -> str:
        """
        Basic text cleaning shared across ingesters.

        Normalises stray whitespace while preserving line and paragraph
        structure — downstream, `SemanticChunker` splits on Markdown
        headings and blank lines, and `_extract_title` matches heading
        lines with `re.MULTILINE`. Collapsing all whitespace (including
        newlines) to single spaces, as an earlier version of this method
        did, silently destroyed that structure: every document became a
        single line, so heading/section detection only ever matched (at
        most) the very first line and swallowed the rest of the document
        into its capture group.
        """
        import re
        # Remove null bytes
        text = text.replace("\x00", "")
        # Collapse runs of horizontal whitespace (spaces/tabs), not newlines
        text = re.sub(r"[ \t]+", " ", text)
        # Trim trailing whitespace on each line
        text = "\n".join(line.rstrip() for line in text.split("\n"))
        # Collapse 3+ consecutive blank lines down to a single paragraph break
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @staticmethod
    def _detect_language(text: str) -> str:
        """Detect document language using langdetect."""
        try:
            from langdetect import detect
            return detect(text[:500])
        except Exception:
            return "en"

    @staticmethod
    def _redact_sensitive(text: str) -> str:
        """Redact common sensitive patterns before indexing."""
        import re
        patterns = {
            "ip_address": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
            "api_key": r"(?i)(api[_-]?key|token|secret|password)\s*[=:]\s*\S+",
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        }
        for label, pattern in patterns.items():
            text = re.sub(pattern, f"[REDACTED_{label.upper()}]", text)
        return text
