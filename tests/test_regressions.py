"""
tests/test_regressions.py
──────────────────────────
Regression tests for bugs fixed in the code-review pass. Each test pins a
specific defect so it can't silently come back. No network, API keys or
model downloads are needed.
"""

from __future__ import annotations

import sys
import types
from datetime import datetime, timezone
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.diagnosis.diagnoser import DiagnosisResult, IncidentDiagnoser, RootCause
from src.diagnosis.llm_client import LLMClient
from src.indexing.chunker import DocumentChunk, SemanticChunker, SlidingWindowChunker
from src.indexing.embedder import Embedder, OpenAIEmbeddingProvider
from src.indexing.vector_store import FAISSVectorStore, SearchResult
from src.ingestion.base import BaseIngester, DocumentType
from src.ingestion.runbook_ingester import LocalRunbookIngester
from src.ingestion.ticket_ingester import LocalTicketIngester, ServiceNowIngester
from src.retrieval.retriever import SemanticRetriever

DIMS = 16


def make_chunk(chunk_id: str, metadata: dict | None = None,
               doc_type: DocumentType = DocumentType.RUNBOOK) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id=f"doc_{chunk_id}",
        source_system="test",
        document_type=doc_type,
        text=f"text for {chunk_id}",
        token_count=3,
        chunk_index=0,
        total_chunks=1,
        metadata=metadata or {},
    )


def vec(seed: int, dims: int = DIMS) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dims).astype(np.float32)
    return v / np.linalg.norm(v)


def store_config(path, dims: int = DIMS) -> dict:
    return {
        "index_path": str(path),
        "dimensions": dims,
        "hnsw_m": 8,
        "hnsw_ef_construction": 32,
        "hnsw_ef_search": 16,
    }


@pytest.fixture
def fake_openai(monkeypatch):
    """Install a fake `openai` module so clients can be constructed offline."""
    module = types.ModuleType("openai")
    module.OpenAI = MagicMock(name="OpenAI")
    module.AzureOpenAI = MagicMock(name="AzureOpenAI")
    monkeypatch.setitem(sys.modules, "openai", module)
    return module


# ── Embedder configuration ───────────────────────────────────────

class TestEmbedderConfig:

    APP_CONFIG = {
        "llm": {
            "provider": "anthropic",
            "model": "gpt-4-turbo",
            "openai": {"api_key": "sk-test"},
            "anthropic": {"api_key": "sk-ant"},
        },
        "embedding": {
            "provider": "local",
            "model": "text-embedding-ada-002",
            "local_model": "sentence-transformers/all-MiniLM-L6-v2",
        },
        "cache": {"enabled": False},
    }

    def test_embedding_section_wins_over_llm_section(self, monkeypatch):
        captured = {}

        def fake_build(self, config):
            captured.update(config)
            return MagicMock(dimensions=384)

        monkeypatch.setattr(Embedder, "_build_provider", fake_build)
        Embedder.from_app_config(self.APP_CONFIG)

        # Previously llm.provider ("anthropic") / llm.model ("gpt-4-turbo")
        # overwrote these, crashing startup or embedding with a chat model.
        assert captured["provider"] == "local"
        assert captured["model"] == "text-embedding-ada-002"
        assert captured["openai"]["api_key"] == "sk-test"

    def test_openai_dimensions_follow_model(self, fake_openai):
        assert OpenAIEmbeddingProvider("k", "text-embedding-3-large").dimensions == 3072
        assert OpenAIEmbeddingProvider("k", "text-embedding-3-small").dimensions == 1536
        assert OpenAIEmbeddingProvider("k", "custom-model", dimensions=256).dimensions == 256


# ── Vector store ─────────────────────────────────────────────────

class TestVectorStore:

    def test_filter_keeps_chunks_without_the_key(self, tmp_path):
        store = FAISSVectorStore(store_config(tmp_path / "idx"))
        store.add_chunks([
            (make_chunk("ticket_pay", {"service": "payments"},
                        DocumentType.INCIDENT_TICKET), vec(1)),
            (make_chunk("ticket_auth", {"service": "auth"},
                        DocumentType.INCIDENT_TICKET), vec(2)),
            (make_chunk("runbook", {"title": "DB pool runbook"}), vec(3)),
        ])
        ids = {r.chunk.chunk_id for r in store.search(
            vec(4), top_k=10, filters={"service": "payments"})}
        assert ids == {"ticket_pay", "runbook"}

    def test_dimension_mismatch_on_load_starts_empty(self, tmp_path):
        path = tmp_path / "idx"
        old = FAISSVectorStore(store_config(path, dims=DIMS))
        old.add_chunks([(make_chunk("c1"), vec(1))])

        reloaded = FAISSVectorStore(store_config(path, dims=DIMS * 2))
        assert reloaded.total_vectors == 0
        # And a rebuild with the new dimension works.
        reloaded.rebuild_index([(make_chunk("c2"), vec(2, DIMS * 2))])
        assert reloaded.total_vectors == 1

    def test_query_dimension_mismatch_raises_clear_error(self, tmp_path):
        store = FAISSVectorStore(store_config(tmp_path / "idx"))
        store.add_chunks([(make_chunk("c1"), vec(1))])
        with pytest.raises(ValueError, match="dimensions"):
            store.search(vec(1, DIMS + 1), top_k=1)

    def test_rebuild_dedupes_and_leaves_no_temp_files(self, tmp_path):
        path = tmp_path / "idx"
        store = FAISSVectorStore(store_config(path))
        store.rebuild_index([
            (make_chunk("dup"), vec(1)),
            (make_chunk("dup"), vec(2)),
            (make_chunk("other"), vec(3)),
        ])
        assert store.total_vectors == store.total_chunks == 2
        assert sorted(p.name for p in path.iterdir()) == ["chunks.pkl", "index.faiss"]

        reloaded = FAISSVectorStore(store_config(path))
        assert reloaded.total_vectors == 2


# ── Chunker ──────────────────────────────────────────────────────

class TestChunkerValidation:

    @pytest.mark.parametrize("overlap", [100, 150, -1])
    def test_invalid_overlap_rejected(self, overlap):
        # overlap >= chunk_size used to loop forever.
        with pytest.raises(ValueError):
            SlidingWindowChunker(chunk_size=100, overlap=overlap)

    def test_small_semantic_chunk_size_does_not_hang(self):
        chunker = SemanticChunker(max_chunk_size=20)
        assert len(chunker._merge_and_split(["word " * 100])) > 1


# ── Ingestion ────────────────────────────────────────────────────

class TestIngestion:

    def test_stable_hash_is_deterministic(self):
        # Must not depend on PYTHONHASHSEED (built-in hash() does).
        assert BaseIngester._stable_hash("abc") == "ba7816"

    def test_runbook_ids_stable_across_runs(self, tmp_path):
        rb = tmp_path / "runbooks"
        rb.mkdir()
        (rb / "db-pool.md").write_text("# DB Pool\n\nRestart the pool.\n")
        cfg = {"path": str(tmp_path), "patterns": ["*.md"]}
        first = [d.document_id for d in LocalRunbookIngester(cfg).run()]
        second = [d.document_id for d in LocalRunbookIngester(cfg).run()]
        assert first == second
        expected = BaseIngester._stable_hash(BaseIngester._clean_text(
            "# DB Pool\n\nRestart the pool.\n"))
        assert first == [f"runbook_db-pool_{expected}"]

    def test_runbook_mentioning_post_mortem_stays_runbook(self, tmp_path):
        rb = tmp_path / "runbooks"
        rb.mkdir()
        f = rb / "oom.md"
        f.write_text("# OOMKilled\n\nSee the post-mortem for INC-42.\n")
        ingester = LocalRunbookIngester({"path": str(tmp_path)})
        assert ingester._classify_document(f, f.read_text()) == DocumentType.RUNBOOK

    def test_ticket_without_id_gets_deterministic_id(self, tmp_path):
        ingester = LocalTicketIngester({"path": str(tmp_path)})
        ticket = {"title": "Disk full", "description": "No space left"}
        a = ingester._ticket_to_document(ticket).document_id
        b = ingester._ticket_to_document(dict(ticket)).document_id
        assert a == b and a.startswith("ticket_")

    @pytest.mark.parametrize("record", [
        {   # default ServiceNow output: plain strings
            "sys_id": "abc", "number": "INC1", "short_description": "DB down",
            "priority": "1 - Critical", "state": "Resolved",
            "sys_created_on": "2024-01-15 10:00:00",
        },
        {   # sysparm_display_value=all output: value/display_value dicts
            "sys_id": {"value": "abc", "display_value": "abc"},
            "number": {"value": "INC1", "display_value": "INC1"},
            "short_description": {"value": "DB down", "display_value": "DB down"},
            "priority": {"value": "1", "display_value": "1 - Critical"},
            "state": {"value": "6", "display_value": "Resolved"},
            "sys_created_on": {"value": "2024-01-15 10:00:00",
                               "display_value": "15/01/2024 10:00"},
        },
    ])
    def test_servicenow_record_parsing(self, record):
        doc = ServiceNowIngester({})._record_to_document(record)
        assert doc.document_id == "snow_abc"
        assert doc.title == "DB down"
        assert doc.metadata["priority"] == "1 - Critical"
        assert doc.metadata["state"] == "Resolved"
        assert doc.created_at == datetime(2024, 1, 15, 10, tzinfo=timezone.utc)


# ── LLM client ───────────────────────────────────────────────────

class TestLLMClient:

    def test_parse_json_with_surrounding_prose(self):
        raw = 'Here is the diagnosis:\n{"confidence_score": 0.7}\nHope this helps!'
        assert LLMClient.parse_json(raw) == {"confidence_score": 0.7}

    def test_parse_json_with_fences(self):
        assert LLMClient.parse_json('```json\n{"a": 1}\n```') == {"a": 1}

    def test_parse_json_rejects_non_object(self):
        with pytest.raises(ValueError):
            LLMClient.parse_json("[1, 2, 3]")

    def test_parse_json_rejects_garbage(self):
        with pytest.raises(ValueError):
            LLMClient.parse_json("no json here")

    def test_azure_uses_deployment_name(self, fake_openai):
        client = LLMClient({
            "provider": "azure_openai",
            "model": "gpt-4-turbo",
            "azure_openai": {
                "api_key": "k", "endpoint": "https://x.openai.azure.com/",
                "deployment_name": "my-gpt4-deployment", "api_version": "2024-02-01",
            },
        })
        assert client.model == "my-gpt4-deployment"

    def test_none_content_does_not_crash(self, fake_openai):
        client = LLMClient({"provider": "openai", "openai": {"api_key": "k"}})
        response = MagicMock()
        response.choices = [MagicMock(message=MagicMock(content=None))]
        client._client.chat.completions.create.return_value = response
        assert client.complete("sys", "user") == ""


# ── Diagnoser output parsing ─────────────────────────────────────

class TestParseOutput:

    MESSY = {
        "root_causes": [
            {"hypothesis": "Pool exhaustion", "confidence": "high", "evidence": None},
            {"hypothesis": "Slow query", "confidence": 85},
            "not a dict",
        ],
        "confidence_score": "0.8",
        "remediation_steps": None,
        "requires_escalation": "false",
        "escalation_reason": None,
        "related_docs": None,
        "diagnostic_summary": None,
    }

    def parse(self, raw):
        return IncidentDiagnoser._parse_output("INC-1", raw, [], 12.0)

    def test_messy_llm_output_is_coerced(self):
        result = self.parse(self.MESSY)
        assert [rc.hypothesis for rc in result.root_causes] == ["Pool exhaustion", "Slow query"]
        assert result.root_causes[0].confidence == 0.5   # unparseable → default
        assert result.root_causes[0].evidence == ""
        assert result.root_causes[1].confidence == 0.85  # percentage → fraction
        assert result.confidence_score == 0.8
        assert result.requires_escalation is False        # bool("false") was True
        assert result.escalation_reason == ""
        assert result.remediation_steps == []

    def test_messy_output_passes_api_response_validation(self):
        from src.api.models import DiagnoseResponse

        # Used to raise a ValidationError (→ HTTP 500) on escalation_reason=None.
        DiagnoseResponse(**self.parse(self.MESSY).to_dict())

    def test_non_dict_output(self):
        result = self.parse(["unexpected"])
        assert result.root_causes == [] and result.confidence_score == 0.0


# ── Retriever filter fallback ────────────────────────────────────

class TestRetrieverFilterFallback:

    def build(self, filtered, unfiltered):
        store = MagicMock()
        store.search.side_effect = lambda emb, top_k, filters=None: (
            list(filtered) if filters else list(unfiltered)
        )
        embedder = MagicMock()
        embedder.embed_text.return_value = vec(0)
        reranker = MagicMock()
        reranker.rerank.side_effect = lambda q, results: results
        retriever = SemanticRetriever(
            vector_store=store, embedder=embedder, reranker=reranker,
            config={"rerank_threshold": 0.0, "min_docs_before_expansion": 2,
                    "hyde_enabled": False},
        )
        return retriever, store

    def test_sparse_filtered_results_are_topped_up(self):
        a = SearchResult(make_chunk("a"), 0.9, 1)
        b = SearchResult(make_chunk("b"), 0.8, 1)
        retriever, store = self.build(filtered=[], unfiltered=[a, b])
        results = retriever.retrieve("q", filters={"service": "new-service"})
        assert [r.chunk.chunk_id for r in results] == ["a", "b"]
        assert store.search.call_count == 2

    def test_enough_filtered_results_no_fallback(self):
        a = SearchResult(make_chunk("a"), 0.9, 1)
        b = SearchResult(make_chunk("b"), 0.8, 1)
        retriever, store = self.build(filtered=[a, b], unfiltered=[])
        retriever.retrieve("q", filters={"service": "payments"})
        assert store.search.call_count == 1


# ── CLI scripts ──────────────────────────────────────────────────

class TestScripts:

    def test_root_cause_match_ignores_stopwords(self):
        pytest.importorskip("rich")
        from scripts.evaluate import root_cause_matches

        gt = "Database connection pool exhaustion"
        assert root_cause_matches("Connection pool exhaustion, caused by a slow query", gt)
        # Shares only stopwords / one weak term with the label.
        assert not root_cause_matches("Memory leak due to the cache", "Pool exhaustion due to the load")

    def test_print_result_does_not_crash(self):
        pytest.importorskip("rich")
        from scripts.diagnose_incident import print_result

        result = DiagnosisResult(
            incident_id="INC-1",
            root_causes=[RootCause("Pool exhaustion", 0.9, "runbook")],
            confidence_score=0.9,
            remediation_steps=["Restart pods"],
            requires_escalation=False,
            escalation_reason="",
            related_docs=[],
            diagnostic_summary="summary",
            retrieved_chunks=[],
            latency_ms=10.0,
        )
        print_result(result)  # used to raise AttributeError (retrieved_context_count)
