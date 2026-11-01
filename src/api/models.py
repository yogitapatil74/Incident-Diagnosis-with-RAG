"""
src/api/models.py
──────────────────
Pydantic request/response models for the FastAPI layer.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class DiagnoseRequest(BaseModel):
    incident_id: str = Field(..., examples=["INC-20240115-001"])
    title: str = Field(..., examples=["Payment service returning 503 errors"])
    description: str = Field(..., examples=["The payments-service pod is returning HTTP 503..."])
    service: str | None = Field(None, examples=["payments-service"])
    priority: str | None = Field(None, examples=["P1"])
    environment: str | None = Field(None, examples=["production"])
    error_message: str | None = Field(None, examples=["ConnectionPool timeout after 30s"])
    affected_components: list[str] | None = Field(None, examples=[["payments-db", "payments-api"]])
    recent_deployments: list[str] | None = Field(None, examples=[["payments-service v2.3.1"]])

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "incident_id": "INC-20240115-001",
                "title": "Payment service returning 503 errors",
                "description": "The payments-service pods are returning HTTP 503 errors. Pod restarts observed in the last 10 minutes. Database connection errors in logs.",
                "service": "payments-service",
                "priority": "P1",
                "environment": "production",
                "error_message": "HikariPool-1 - Connection is not available, request timed out after 30000ms",
                "affected_components": ["payments-db", "payments-api"],
                "recent_deployments": ["payments-service v2.3.1 (2h ago)"]
            }]
        }
    }


class RootCauseModel(BaseModel):
    hypothesis: str
    confidence: float
    evidence: str


class RelatedDocModel(BaseModel):
    title: str
    document_id: str
    relevance: str


class DiagnoseResponse(BaseModel):
    incident_id: str
    timestamp: str
    confidence_score: float
    diagnostic_summary: str
    root_causes: list[RootCauseModel]
    remediation_steps: list[str]
    requires_escalation: bool
    escalation_reason: str
    related_docs: list[RelatedDocModel]
    retrieved_context_count: int
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    version: str
    vector_store_size: int
    index_loaded: bool


class StatsResponse(BaseModel):
    total_vectors: int
    total_chunks: int
    index_type: str
    dimensions: int
    hnsw_m: int


class IndexRequest(BaseModel):
    source: str = Field(..., examples=["local_files"])
    path: str | None = Field(None, examples=["data/samples"])


class IndexResponse(BaseModel):
    status: str
    documents_ingested: int
    chunks_indexed: int
    message: str
