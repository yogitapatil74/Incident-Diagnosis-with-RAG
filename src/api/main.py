"""
src/api/main.py
────────────────
FastAPI application entry point.
Provides REST endpoints for incident diagnosis, knowledge base
management, and system health monitoring.
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager

import structlog
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.responses import Response

from src import __version__
from src.api.models import (
    DiagnoseRequest,
    DiagnoseResponse,
    HealthResponse,
    StatsResponse,
)
from src.diagnosis.diagnoser import IncidentContext, IncidentDiagnoser
from src.indexing.chunker import DocumentChunker
from src.indexing.embedder import Embedder
from src.indexing.vector_store import FAISSVectorStore
from src.pipeline import build_pipeline

logger = structlog.get_logger(__name__)

CONFIG_PATH_ENV = "AIOPS_CONFIG"
DEFAULT_CONFIG_PATH = "config/config.yaml"

# ── Prometheus metrics ──────────────────────────────────────────
REQUEST_COUNT = Counter(
    "aiops_requests_total", "Total diagnosis requests", ["status"]
)
REQUEST_LATENCY = Histogram(
    "aiops_request_latency_seconds", "Diagnosis request latency",
    buckets=[0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 30.0]
)
RETRIEVAL_DOCS = Histogram(
    "aiops_retrieval_docs", "Number of documents retrieved per query",
    buckets=[0, 1, 2, 3, 5, 8, 10]
)


def load_config(path: str | None = None) -> dict:
    path = path or os.environ.get(CONFIG_PATH_ENV, DEFAULT_CONFIG_PATH)
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _cors_origins() -> list[str]:
    """Read allowed CORS origins from config (api.cors_origins).

    Previously the configured value was ignored and "*" was always used.
    Falls back to "*" if the config can't be read at import time.
    """
    try:
        origins = load_config().get("api", {}).get("cors_origins")
    except Exception:
        origins = None
    return list(origins) if origins else ["*"]


# ── App State ───────────────────────────────────────────────────

class AppState:
    config: dict | None = None
    diagnoser: IncidentDiagnoser | None = None
    chunker: DocumentChunker | None = None
    embedder: Embedder | None = None
    vector_store: FAISSVectorStore | None = None


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise all components on startup."""
    logger.info("Initialising RAG-AIOps Framework")
    try:
        state.config = load_config()
        pipeline = build_pipeline(state.config)

        state.embedder = pipeline.embedder
        state.vector_store = pipeline.vector_store
        state.diagnoser = pipeline.diagnoser
        state.chunker = DocumentChunker(state.config.get("chunking", {}))

        logger.info(
            "Framework initialised",
            index_size=state.vector_store.total_vectors,
        )
    except Exception as e:
        logger.error("Startup failed", error=str(e), exc_info=True)
        raise

    yield  # App running

    logger.info("Shutting down")


# ── FastAPI App ─────────────────────────────────────────────────

app = FastAPI(
    title="RAG-AIOps Incident Diagnosis API",
    description="REST API for the RAG-based AIOps incident diagnosis framework.",
    version=__version__,
    lifespan=lifespan,
)

_origins = _cors_origins()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _require_ready() -> None:
    if state.diagnoser is None or state.vector_store is None:
        raise HTTPException(status_code=503, detail="Service is still initialising")


# ── Endpoints ───────────────────────────────────────────────────
# NOTE: these are plain `def` (not `async def`) on purpose. The diagnosis
# pipeline is fully synchronous and blocking (embedding API call, FAISS
# search, cross-encoder inference, LLM call — several seconds). Declared as
# `async def`, it ran directly on the event loop and blocked every other
# request (including /v1/health, which made the container healthcheck fail
# under load). FastAPI runs `def` endpoints in a worker threadpool.

@app.post("/v1/diagnose", response_model=DiagnoseResponse, tags=["Diagnosis"])
def diagnose_incident(request: DiagnoseRequest):
    """
    **Primary endpoint** — Run the full RAG diagnosis pipeline on an incident.

    Returns structured root cause analysis, remediation steps, and
    confidence score.
    """
    _require_ready()
    start = time.perf_counter()
    try:
        incident = IncidentContext(
            incident_id=request.incident_id,
            title=request.title,
            description=request.description,
            service=request.service,
            priority=request.priority,
            environment=request.environment,
            error_message=request.error_message,
            affected_components=request.affected_components or [],
            recent_deployments=request.recent_deployments or [],
        )
        result = state.diagnoser.diagnose(incident)

        REQUEST_COUNT.labels(status="success").inc()
        REQUEST_LATENCY.observe(time.perf_counter() - start)
        RETRIEVAL_DOCS.observe(len(result.retrieved_chunks))

        return DiagnoseResponse(**result.to_dict())

    except Exception as e:
        REQUEST_COUNT.labels(status="error").inc()
        logger.error("Diagnosis failed", error=str(e), exc_info=True)
        # Don't leak internal exception text (may include URLs, keys, paths).
        raise HTTPException(status_code=500, detail="Diagnosis failed — see server logs")


@app.get("/v1/health", response_model=HealthResponse, tags=["Operations"])
def health_check():
    """System health check including vector store status."""
    size = state.vector_store.total_vectors if state.vector_store else 0
    return HealthResponse(
        status="healthy" if state.diagnoser is not None else "initialising",
        version=__version__,
        vector_store_size=size,
        index_loaded=size > 0,
    )


@app.get("/v1/stats", response_model=StatsResponse, tags=["Operations"])
def get_stats():
    """Return knowledge base and index statistics."""
    _require_ready()
    return StatsResponse(**state.vector_store.stats())


@app.get("/metrics", include_in_schema=False)
def metrics():
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
