"""
src/diagnosis/diagnoser.py
───────────────────────────
Core incident diagnosis orchestrator.
Implements the full 8-step diagnosis workflow:
  1. Error context extraction
  2. Query formulation
  3. Query embedding
  4. Semantic retrieval
  5. Context construction
  6. LLM prompt assembly
  7. LLM reasoning & generation
  8. Structured output + feedback loop

DiagnosisResult contains: root_causes, confidence_score,
remediation_steps, requires_escalation, related_docs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

import structlog

from src.diagnosis.llm_client import LLMClient
from src.indexing.vector_store import SearchResult
from src.retrieval.retriever import SemanticRetriever

logger = structlog.get_logger(__name__)


# ── Prompt Templates ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert Site Reliability Engineer (SRE) with deep knowledge \
of cloud-native architectures, Kubernetes, microservices, and production incident management.

You will receive:
1. An incident description with structured metadata
2. Retrieved knowledge from the organisational knowledge base (similar past incidents, runbooks, documentation)

Your task is to analyse the incident and provide a structured diagnosis.

CRITICAL INSTRUCTIONS:
- Respond ONLY with valid JSON. No preamble, no explanation outside JSON.
- Base your diagnosis on the retrieved knowledge whenever possible.
- Assign confidence_score based on how well the retrieved context matches the incident.
- List root_causes in order of probability (most likely first).
- remediation_steps must be specific, actionable, and ordered.

OUTPUT SCHEMA:
{
  "root_causes": [
    {
      "hypothesis": "Brief root cause description",
      "confidence": 0.0-1.0,
      "evidence": "Which retrieved document supports this"
    }
  ],
  "confidence_score": 0.0-1.0,
  "remediation_steps": [
    "Step 1: ...",
    "Step 2: ..."
  ],
  "requires_escalation": true|false,
  "escalation_reason": "Only if requires_escalation is true",
  "related_docs": [
    {
      "title": "Document title",
      "document_id": "doc_id",
      "relevance": "Why this document is relevant"
    }
  ],
  "diagnostic_summary": "2-3 sentence narrative summary for the engineer"
}"""


def _build_user_message(incident: "IncidentContext", context_chunks: list[SearchResult]) -> str:
    """Assemble the full LLM user message from incident + retrieved context."""
    parts = [
        "═══ INCIDENT ═══",
        f"ID: {incident.incident_id}",
        f"Title: {incident.title}",
        f"Service: {incident.service or 'unknown'}",
        f"Priority: {incident.priority or 'unknown'}",
        f"Environment: {incident.environment or 'production'}",
        f"Description:\n{incident.description}",
    ]
    if incident.error_message:
        parts.append(f"Error Message:\n{incident.error_message}")
    if incident.affected_components:
        parts.append(f"Affected Components: {', '.join(incident.affected_components)}")
    if incident.recent_deployments:
        parts.append(f"Recent Deployments: {', '.join(incident.recent_deployments)}")

    parts.append("\n═══ RETRIEVED KNOWLEDGE BASE CONTEXT ═══")
    for i, result in enumerate(context_chunks, 1):
        doc_type = result.chunk.document_type.value.replace("_", " ").title()
        title = result.chunk.metadata.get("title", result.chunk.document_id)
        parts.append(
            f"\n[Context {i} | {doc_type} | Relevance: {result.score:.2f}]\n"
            f"Source: {title}\n"
            f"{result.chunk.text[:800]}"
        )

    parts.append("\n═══ DIAGNOSIS REQUEST ═══")
    parts.append(
        "Based on the incident details and retrieved context above, "
        "provide your structured JSON diagnosis."
    )
    return "\n".join(parts)


# ── Data Classes ──────────────────────────────────────────────────────────────

@dataclass
class IncidentContext:
    """Structured representation of an incoming incident."""
    incident_id: str
    title: str
    description: str
    service: str | None = None
    priority: str | None = None
    environment: str | None = None
    error_message: str | None = None
    error_code: str | None = None
    affected_components: list[str] = field(default_factory=list)
    recent_deployments: list[str] = field(default_factory=list)
    alert_labels: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_query_string(self) -> str:
        """Convert incident to a dense query string for embedding."""
        parts = [self.title, self.description]
        if self.service:
            parts.append(f"service: {self.service}")
        if self.error_message:
            parts.append(f"error: {self.error_message[:300]}")
        if self.affected_components:
            parts.append(f"components: {' '.join(self.affected_components)}")
        return " ".join(parts)


@dataclass
class RootCause:
    hypothesis: str
    confidence: float
    evidence: str


@dataclass
class RelatedDoc:
    title: str
    document_id: str
    relevance: str


@dataclass
class DiagnosisResult:
    """
    Structured output from the RAG diagnosis pipeline.
    """
    incident_id: str
    root_causes: list[RootCause]
    confidence_score: float
    remediation_steps: list[str]
    requires_escalation: bool
    escalation_reason: str
    related_docs: list[RelatedDoc]
    diagnostic_summary: str
    retrieved_chunks: list[SearchResult]
    latency_ms: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        return {
            "incident_id": self.incident_id,
            "timestamp": self.timestamp.isoformat(),
            "confidence_score": self.confidence_score,
            "diagnostic_summary": self.diagnostic_summary,
            "root_causes": [
                {
                    "hypothesis": rc.hypothesis,
                    "confidence": rc.confidence,
                    "evidence": rc.evidence,
                }
                for rc in self.root_causes
            ],
            "remediation_steps": self.remediation_steps,
            "requires_escalation": self.requires_escalation,
            "escalation_reason": self.escalation_reason,
            "related_docs": [
                {
                    "title": rd.title,
                    "document_id": rd.document_id,
                    "relevance": rd.relevance,
                }
                for rd in self.related_docs
            ],
            "retrieved_context_count": len(self.retrieved_chunks),
            "latency_ms": round(self.latency_ms, 1),
        }


# ── Core Diagnoser ─────────────────────────────────────────────────────────

class IncidentDiagnoser:
    """
    Orchestrates the full incident diagnosis pipeline.
    """

    def __init__(
        self,
        retriever: SemanticRetriever,
        llm_client: LLMClient,
        config: dict | None = None,
    ):
        self.retriever = retriever
        self.llm = llm_client
        self.config = config or {}

    def diagnose(self, incident: IncidentContext) -> DiagnosisResult:
        """
        Execute the full 8-step diagnosis workflow.
        Returns a structured DiagnosisResult.
        """
        start_time = time.perf_counter()
        logger.info("Starting diagnosis", incident_id=incident.incident_id)

        # Step 1-3: Context extraction, query formulation, embedding + retrieval
        query = incident.to_query_string()
        filters = self._build_filters(incident)
        retrieved = self.retriever.retrieve(query, filters=filters)

        logger.info(
            "Retrieval complete",
            incident_id=incident.incident_id,
            chunks_retrieved=len(retrieved),
        )

        # Step 5-6: Context construction + prompt assembly
        user_message = _build_user_message(incident, retrieved)

        # Step 7: LLM reasoning
        try:
            raw_output = self.llm.complete_json(SYSTEM_PROMPT, user_message)
        except Exception as e:
            logger.error("LLM diagnosis failed", error=str(e))
            raw_output = self._fallback_output(incident, retrieved)

        # Step 8: Parse and validate structured output
        result = self._parse_output(
            incident_id=incident.incident_id,
            raw=raw_output,
            retrieved=retrieved,
            latency_ms=(time.perf_counter() - start_time) * 1000,
        )

        logger.info(
            "Diagnosis complete",
            incident_id=incident.incident_id,
            confidence=result.confidence_score,
            root_causes=len(result.root_causes),
            latency_ms=round(result.latency_ms, 1),
        )
        return result

    def _build_filters(self, incident: IncidentContext) -> dict | None:
        """Build metadata filter predicates from incident context."""
        filters: dict = {}
        if incident.service:
            filters["service"] = incident.service
        if incident.environment:
            filters["environment"] = incident.environment
        return filters if filters else None

    @staticmethod
    def _parse_output(
        incident_id: str,
        raw: dict,
        retrieved: list[SearchResult],
        latency_ms: float,
    ) -> DiagnosisResult:
        """
        Convert the LLM's JSON into a DiagnosisResult, tolerating the ways
        real model output deviates from the schema. Previously:
          - "escalation_reason": null (the natural reply when there is no
            escalation) became None and failed DiagnoseResponse validation,
            turning a successful diagnosis into an HTTP 500;
          - "requires_escalation": "false" was coerced to True by bool();
          - a non-numeric confidence ("high") raised ValueError;
          - out-of-range confidences (e.g. 85) were passed through;
          - non-dict list items crashed with AttributeError.
        """
        if not isinstance(raw, dict):
            raw = {}

        root_causes = [
            RootCause(
                hypothesis=_as_str(rc.get("hypothesis")) or "Unknown",
                confidence=_as_confidence(rc.get("confidence"), default=0.5),
                evidence=_as_str(rc.get("evidence")),
            )
            for rc in _as_list(raw.get("root_causes"))
            if isinstance(rc, dict)
        ]
        related_docs = [
            RelatedDoc(
                title=_as_str(rd.get("title")),
                document_id=_as_str(rd.get("document_id")),
                relevance=_as_str(rd.get("relevance")),
            )
            for rd in _as_list(raw.get("related_docs"))
            if isinstance(rd, dict)
        ]
        remediation_steps = [
            _as_str(step) for step in _as_list(raw.get("remediation_steps"))
            if _as_str(step)
        ]
        return DiagnosisResult(
            incident_id=incident_id,
            root_causes=root_causes,
            confidence_score=_as_confidence(raw.get("confidence_score"), default=0.0),
            remediation_steps=remediation_steps,
            requires_escalation=_as_bool(raw.get("requires_escalation")),
            escalation_reason=_as_str(raw.get("escalation_reason")),
            related_docs=related_docs,
            diagnostic_summary=_as_str(raw.get("diagnostic_summary")),
            retrieved_chunks=retrieved,
            latency_ms=latency_ms,
        )

    @staticmethod
    def _fallback_output(
        incident: IncidentContext, retrieved: list[SearchResult]
    ) -> dict:
        """Minimal structured fallback when LLM call fails."""
        return {
            "root_causes": [
                {
                    "hypothesis": "LLM reasoning unavailable — manual investigation required",
                    "confidence": 0.0,
                    "evidence": "LLM error",
                }
            ],
            "confidence_score": 0.0,
            "remediation_steps": [
                "Check service health in monitoring dashboard",
                "Review recent deployment logs",
                "Escalate to on-call engineer",
            ],
            "requires_escalation": True,
            "escalation_reason": "Automated diagnosis unavailable",
            "related_docs": [],
            "diagnostic_summary": (
                "Automated diagnosis could not be completed. "
                f"{len(retrieved)} related knowledge base entries were retrieved "
                "and are available for manual review."
            ),
        }


# ── Output coercion helpers ──────────────────────────────────────────────────

def _as_str(value) -> str:
    if value is None:
        return ""
    return value if isinstance(value, str) else str(value)


def _as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1", "y"}
    if isinstance(value, (int, float)):
        return value != 0
    return False


def _as_confidence(value, default: float) -> float:
    """Coerce to a float in [0, 1]. Accepts 0.85, "0.85", 85, "85%"."""
    try:
        if isinstance(value, str):
            value = value.strip().rstrip("%")
        conf = float(value)
    except (TypeError, ValueError):
        return default
    if conf != conf:  # NaN
        return default
    if conf > 1.0:
        conf = conf / 100.0 if conf <= 100.0 else 1.0
    return max(0.0, min(1.0, conf))
