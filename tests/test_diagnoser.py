"""
tests/test_diagnoser.py
────────────────────────
Unit tests for the IncidentDiagnoser and related components.
All LLM and retrieval calls are mocked for deterministic testing.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.diagnosis.diagnoser import (
    DiagnosisResult,
    IncidentContext,
    IncidentDiagnoser,
    _build_user_message,
)
from src.indexing.chunker import DocumentChunk
from src.indexing.vector_store import SearchResult
from src.ingestion.base import DocumentType

# ── Fixtures ─────────────────────────────────────────────────────

def make_chunk(chunk_id: str, text: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id=f"doc_{chunk_id}",
        source_system="test",
        document_type=DocumentType.INCIDENT_TICKET,
        text=text,
        token_count=10,
        chunk_index=0,
        total_chunks=1,
        metadata={"title": "Sample Document", "service": "payments"},
    )


def make_search_result(chunk_id: str, text: str, score: float = 0.85) -> SearchResult:
    return SearchResult(
        chunk=make_chunk(chunk_id, text),
        score=score,
        rank=1,
    )


LLM_RESPONSE = {
    "root_causes": [
        {
            "hypothesis": "Database connection pool exhaustion due to slow query",
            "confidence": 0.92,
            "evidence": "Retrieved incident INC-2024-001 describes identical HikariPool timeout pattern",
        }
    ],
    "confidence_score": 0.89,
    "remediation_steps": [
        "Kill long-running queries: SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE duration > interval '2 minutes'",
        "Roll back to previous deployment version",
        "Increase HikariCP pool size from 10 to 25",
        "Add composite index on (customer_id, created_at)",
    ],
    "requires_escalation": False,
    "escalation_reason": "",
    "related_docs": [
        {
            "title": "Runbook: Database Connection Pool Exhaustion",
            "document_id": "runbook_database-connection-pool",
            "relevance": "Direct match — covers HikariPool timeout remediation",
        }
    ],
    "diagnostic_summary": (
        "The payments-service is experiencing connection pool exhaustion. "
        "A slow query introduced in the recent deployment is holding connections "
        "for 8-15s instead of <50ms. Immediate rollback recommended."
    ),
}


# ── IncidentContext tests ────────────────────────────────────────

class TestIncidentContext:

    def test_to_query_string_includes_title(self):
        incident = IncidentContext(
            incident_id="INC-001",
            title="Payment timeout",
            description="Service is returning 503",
            service="payments",
        )
        query = incident.to_query_string()
        assert "Payment timeout" in query
        assert "503" in query
        assert "payments" in query

    def test_to_query_string_handles_missing_fields(self):
        incident = IncidentContext(
            incident_id="INC-002",
            title="Some issue",
            description="Something broke",
        )
        query = incident.to_query_string()
        assert "Some issue" in query


# ── User message building ────────────────────────────────────────

class TestBuildUserMessage:

    def test_includes_incident_details(self):
        incident = IncidentContext(
            incident_id="INC-001",
            title="DB timeout",
            description="Connection pool exhausted",
            service="payments",
            priority="P1",
        )
        results = [make_search_result("r1", "HikariPool timeout runbook")]
        msg = _build_user_message(incident, results)

        assert "DB timeout" in msg
        assert "payments" in msg
        assert "P1" in msg
        assert "HikariPool timeout runbook" in msg

    def test_includes_retrieved_context(self):
        incident = IncidentContext("INC-001", "title", "desc")
        results = [
            make_search_result("r1", "context one", score=0.9),
            make_search_result("r2", "context two", score=0.8),
        ]
        msg = _build_user_message(incident, results)
        assert "context one" in msg
        assert "context two" in msg
        assert "RETRIEVED KNOWLEDGE" in msg

    def test_empty_retrieved_context(self):
        incident = IncidentContext("INC-001", "title", "desc")
        msg = _build_user_message(incident, [])
        assert "INCIDENT" in msg


# ── IncidentDiagnoser tests ──────────────────────────────────────

class TestIncidentDiagnoser:

    @pytest.fixture
    def mock_retriever(self):
        retriever = MagicMock()
        retriever.retrieve.return_value = [
            make_search_result("r1", "Database connection pool runbook"),
            make_search_result("r2", "HikariPool configuration guide"),
        ]
        return retriever

    @pytest.fixture
    def mock_llm(self):
        llm = MagicMock()
        llm.complete_json.return_value = LLM_RESPONSE
        return llm

    @pytest.fixture
    def diagnoser(self, mock_retriever, mock_llm):
        return IncidentDiagnoser(
            retriever=mock_retriever,
            llm_client=mock_llm,
        )

    def test_diagnose_returns_result(self, diagnoser):
        incident = IncidentContext(
            incident_id="TEST-001",
            title="Payment 503",
            description="HikariPool timeout errors",
            service="payments",
            priority="P1",
        )
        result = diagnoser.diagnose(incident)
        assert isinstance(result, DiagnosisResult)

    def test_diagnose_parses_root_causes(self, diagnoser):
        incident = IncidentContext("TEST-001", "Payment 503", "HikariPool errors")
        result = diagnoser.diagnose(incident)
        assert len(result.root_causes) == 1
        assert "exhaustion" in result.root_causes[0].hypothesis.lower()
        assert result.root_causes[0].confidence == 0.92

    def test_diagnose_parses_remediation_steps(self, diagnoser):
        incident = IncidentContext("TEST-001", "Payment 503", "HikariPool errors")
        result = diagnoser.diagnose(incident)
        assert len(result.remediation_steps) == 4
        assert "pg_terminate_backend" in result.remediation_steps[0]

    def test_diagnose_confidence_score(self, diagnoser):
        incident = IncidentContext("TEST-001", "Payment 503", "HikariPool errors")
        result = diagnoser.diagnose(incident)
        assert 0.0 <= result.confidence_score <= 1.0
        assert result.confidence_score == 0.89

    def test_diagnose_not_escalated(self, diagnoser):
        incident = IncidentContext("TEST-001", "Payment 503", "HikariPool errors")
        result = diagnoser.diagnose(incident)
        assert result.requires_escalation is False

    def test_diagnose_includes_related_docs(self, diagnoser):
        incident = IncidentContext("TEST-001", "Payment 503", "HikariPool errors")
        result = diagnoser.diagnose(incident)
        assert len(result.related_docs) == 1
        assert "Runbook" in result.related_docs[0].title

    def test_diagnose_latency_recorded(self, diagnoser):
        incident = IncidentContext("TEST-001", "Payment 503", "HikariPool errors")
        result = diagnoser.diagnose(incident)
        assert result.latency_ms > 0

    def test_to_dict_serialisable(self, diagnoser):
        import json
        incident = IncidentContext("TEST-001", "Payment 503", "HikariPool errors")
        result = diagnoser.diagnose(incident)
        d = result.to_dict()
        json.dumps(d)  # should not raise
        assert "incident_id" in d
        assert "root_causes" in d

    def test_llm_failure_uses_fallback(self, mock_retriever):
        llm = MagicMock()
        llm.complete_json.side_effect = Exception("LLM API timeout")
        diagnoser = IncidentDiagnoser(mock_retriever, llm)
        incident = IncidentContext("TEST-002", "Unknown failure", "Something broke")
        result = diagnoser.diagnose(incident)
        # Should return fallback, not raise
        assert result.requires_escalation is True
        assert result.confidence_score == 0.0

    def test_retriever_called_with_filters(self, mock_retriever, mock_llm):
        diagnoser = IncidentDiagnoser(mock_retriever, mock_llm)
        incident = IncidentContext(
            incident_id="TEST-003",
            title="Service down",
            description="Error occurring",
            service="checkout",
            environment="production",
        )
        diagnoser.diagnose(incident)
        _, kwargs = mock_retriever.retrieve.call_args
        assert kwargs.get("filters") is not None
