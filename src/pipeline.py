"""
src/pipeline.py
────────────────
Single place that wires the RAG pipeline components together from the
application config. Used by the API and the CLI scripts so they can't
drift apart (previously each one assembled the components by hand, and
they disagreed on how to size the vector store and how to configure the
embedder).
"""

from __future__ import annotations

from dataclasses import dataclass

from src.diagnosis.diagnoser import IncidentDiagnoser
from src.diagnosis.llm_client import LLMClient
from src.indexing.embedder import Embedder
from src.indexing.vector_store import FAISSVectorStore
from src.retrieval.query_expander import HyDEQueryExpander
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.retriever import SemanticRetriever


@dataclass
class Pipeline:
    embedder: Embedder
    vector_store: FAISSVectorStore
    retriever: SemanticRetriever
    llm_client: LLMClient
    diagnoser: IncidentDiagnoser


def build_vector_store(config: dict, embedder: Embedder) -> FAISSVectorStore:
    """Size the index from the embedder actually in use, not from config,
    so a stale/incorrect ``embedding.dimensions`` value can't cause a
    dimension mismatch."""
    return FAISSVectorStore({
        **config.get("vector_store", {}),
        "dimensions": embedder.dimensions,
    })


def build_pipeline(config: dict) -> Pipeline:
    retrieval_cfg = config.get("retrieval", {})

    embedder = Embedder.from_app_config(config)
    vector_store = build_vector_store(config, embedder)
    reranker = CrossEncoderReranker(
        retrieval_cfg.get("reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    )
    llm_client = LLMClient(config["llm"])
    query_expander = (
        HyDEQueryExpander(llm_client) if retrieval_cfg.get("hyde_enabled", True) else None
    )
    retriever = SemanticRetriever(
        vector_store=vector_store,
        embedder=embedder,
        reranker=reranker,
        query_expander=query_expander,
        config=retrieval_cfg,
    )
    diagnoser = IncidentDiagnoser(retriever, llm_client, config)
    return Pipeline(
        embedder=embedder,
        vector_store=vector_store,
        retriever=retriever,
        llm_client=llm_client,
        diagnoser=diagnoser,
    )
