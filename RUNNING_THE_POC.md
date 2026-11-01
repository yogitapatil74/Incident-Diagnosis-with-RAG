# Running the PoC — RAG-AIOps Incident Diagnosis

---

## Prerequisites

- Python 3.10+
- Git
- Docker & Docker Compose *(optional — for full stack only)*
- OpenAI API key *(optional — local embeddings available as free alternative)*

---

## Step 1 — Clone & Setup

```bash
git clone https://github.com/yogitapatil74/Incident-Diagnosis-with-RAG.git
cd Incident-Diagnosis-with-RAG

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Mac/Linux
# .venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Step 2 — Configure

```bash
cp config/config.example.yaml config/config.yaml
```

Open `config/config.yaml` and set your OpenAI key:

```yaml
llm:
  provider: "openai"
  openai:
    api_key: "sk-YOUR_KEY_HERE"
```

### No OpenAI key? Use local embeddings (free, fully offline)

```yaml
embedding:
  provider: "local"
  local_model: "sentence-transformers/all-mpnet-base-v2"
```

Then pass `--local-embeddings` to all scripts below.

---

## Step 3 — Index the Knowledge Base

```bash
python scripts/index_knowledge_base.py --data-dir data/samples

# Without OpenAI key:
python scripts/index_knowledge_base.py --data-dir data/samples --local-embeddings
```

**Expected output:**

```
─── RAG-AIOps Knowledge Base Indexer ───
Step 1: Ingesting documents...
  ✓ Incident tickets:              5
  ✓ Runbooks/docs/post-mortems:    3
  Total documents ingested:        8

Step 2: Chunking documents...
  ✓ Total chunks: 47

Step 3: Generating embeddings...
  Embedding chunks... ████████████ 47/47

Step 4: Building FAISS HNSW index...

─── Indexing Complete ───
  Documents ingested      8
  Chunks created          47
  Vectors indexed         47
  Index type              FAISS HNSW
  Embedding dimensions    1536
```

### Re-index from scratch (blue-green swap)

```bash
python scripts/index_knowledge_base.py --rebuild
```

---

## Step 4 — Diagnose a Single Incident

### Using the sample incident file

```bash
python scripts/diagnose_incident.py \
  --incident data/samples/incidents/sample_incidents.json
```

### Or pass details directly on the command line

```bash
python scripts/diagnose_incident.py \
  --title "Payment service returning 503 errors" \
  --description "HikariPool connection timeout. Pod restarts observed after v2.3.1 deploy." \
  --service "payments-service" \
  --priority "P1"
```

### Save result to JSON

```bash
python scripts/diagnose_incident.py \
  --incident data/samples/incidents/sample_incidents.json \
  --output data/eval_results/diagnosis_output.json
```

**Expected output:**

```
─── RAG-AIOps Incident Diagnosis ───

╭─── RAG-AIOps Diagnosis Result ──────────────────────────────╮
│ Incident:    INC-2024-001                                    │
│ Confidence:  89%                                             │
│ ✓ No escalation needed                                       │
│                                                              │
│ The payments-service is experiencing connection pool         │
│ exhaustion. A slow query introduced in v2.3.1 holds          │
│ connections for 8-15s. Immediate rollback recommended.       │
╰──────────────────────────────────────────────────────────────╯

Root Cause Hypotheses:
  1. Database connection pool exhaustion due to slow query
     Confidence: ████████████████████ 92%
     Evidence: Retrieved INC-2024-001 — identical HikariPool pattern

Remediation Steps:
  1. Kill long-running queries: SELECT pg_terminate_backend(pid)...
  2. Roll back to previous deployment version
  3. Increase HikariCP pool size from 10 to 25
  4. Add composite index on (customer_id, created_at)

Related Knowledge Base Documents:
  → Runbook: Database Connection Pool Exhaustion
    Direct match — covers HikariPool timeout remediation

Retrieved 5 knowledge base chunks | Latency: 2847ms
```

---

## Step 5 — Run Evaluation

```bash
python scripts/evaluate.py \
  --dataset data/samples/eval_dataset.json \
  --output data/eval_results/results.json
```

### Run ablation study

```bash
python scripts/evaluate.py \
  --dataset data/samples/eval_dataset.json \
  --ablation \
  --output data/eval_results/ablation_results.json
```

**Expected output:**

```
─── RAG-AIOps Evaluation Harness ───
Loaded 8 evaluation samples

  Evaluating: Full System (8 samples)
    Processed 8/8...

┌──────────────────┬───────────┬───────────┬────────────┬─────────┬─────────┐
│ Configuration    │ Accuracy  │ Prec@1    │ Recall@3   │ MDT (s) │ Samples │
├──────────────────┼───────────┼───────────┼────────────┼─────────┼─────────┤
│ Full System      │   87.5%   │   87.5%   │   100.0%   │   3.21  │       8 │
└──────────────────┴───────────┴───────────┴────────────┴─────────┴─────────┘

Results saved to data/eval_results/results.json
```

> **Note:** The sample dataset has 8 incidents, meant to demonstrate the evaluation
> harness end-to-end. For meaningful metrics at scale, populate
> `data/samples/incidents/` with your own incident dataset following the schema in
> `data/samples/incidents/sample_incidents.json`.

---

## Step 6 — Start the REST API

```bash
uvicorn src.api.main:app --reload --port 8000
```

Open **http://localhost:8000/docs** in your browser — Swagger UI with all endpoints.

### Test via curl

```bash
curl -X POST http://localhost:8000/v1/diagnose \
  -H "Content-Type: application/json" \
  -d '{
    "incident_id": "INC-001",
    "title": "Payment service 503 errors",
    "description": "HikariPool connection timeout after v2.3.1 deployment",
    "service": "payments-service",
    "priority": "P1",
    "error_message": "HikariPool-1 - Connection is not available, request timed out after 30000ms"
  }'
```

### Check system health

```bash
curl http://localhost:8000/v1/health
```

### Check knowledge base stats

```bash
curl http://localhost:8000/v1/stats
```

---

## Step 7 — Run Tests

```bash
# All tests (no API key needed — all external calls are mocked)
pytest tests/ -v

# With coverage report
pytest tests/ --cov=src --cov-report=html
open htmlcov/index.html      # Mac
# xdg-open htmlcov/index.html  # Linux
```

**Expected output:**

```
tests/test_chunker.py::TestSlidingWindowChunker::test_short_text_single_chunk   PASSED
tests/test_chunker.py::TestSlidingWindowChunker::test_long_text_multiple_chunks  PASSED
tests/test_chunker.py::TestSemanticChunker::test_section_boundaries_respected   PASSED
tests/test_retriever.py::TestFAISSVectorStore::test_add_and_retrieve             PASSED
tests/test_retriever.py::TestFAISSVectorStore::test_metadata_filter              PASSED
tests/test_retriever.py::TestFAISSVectorStore::test_persistence_and_reload       PASSED
tests/test_diagnoser.py::TestIncidentDiagnoser::test_diagnose_returns_result     PASSED
tests/test_diagnoser.py::TestIncidentDiagnoser::test_diagnose_parses_root_causes PASSED
...

30 passed in 12.4s
```

---

## Step 8 — Full Docker Stack (Optional)

Starts API + Redis + Prometheus + Grafana in one command.

```bash
docker-compose up --build
```

| Service    | URL                          | Credentials  |
|------------|------------------------------|--------------|
| API + Docs | http://localhost:8000/docs   | —            |
| Prometheus | http://localhost:9090        | —            |
| Grafana    | http://localhost:3000        | admin / admin |

### Stop the stack

```bash
docker-compose down
```

### Stop and remove all data volumes

```bash
docker-compose down -v
```

---

## Quick Reference

| Goal                          | Command                                                         |
|-------------------------------|-----------------------------------------------------------------|
| Index knowledge base          | `python scripts/index_knowledge_base.py --data-dir data/samples`|
| Re-index from scratch         | `python scripts/index_knowledge_base.py --rebuild`              |
| Diagnose from file            | `python scripts/diagnose_incident.py --incident <file>`         |
| Diagnose from CLI args        | `python scripts/diagnose_incident.py --title "..." --description "..."` |
| Run evaluation                | `python scripts/evaluate.py --dataset data/samples/eval_dataset.json` |
| Run ablation study            | `python scripts/evaluate.py --ablation`                         |
| Start REST API                | `uvicorn src.api.main:app --reload --port 8000`                 |
| Run unit tests                | `pytest tests/ -v`                                              |
| Run tests with coverage       | `pytest tests/ --cov=src --cov-report=html`                     |
| Full Docker stack             | `docker-compose up --build`                                     |

---

## Project Structure

```
Incident-Diagnosis-with-RAG/
├── src/
│   ├── ingestion/          # Data source connectors
│   │   ├── base.py         # OperationalDocument schema + BaseIngester
│   │   ├── ticket_ingester.py
│   │   └── runbook_ingester.py
│   ├── indexing/           # Knowledge indexing pipeline
│   │   ├── chunker.py      # Sliding window / semantic / sentence strategies
│   │   ├── embedder.py     # OpenAI + local embeddings with Redis cache
│   │   └── vector_store.py # FAISS HNSW index with blue-green swap
│   ├── retrieval/          # Semantic retrieval engine
│   │   ├── retriever.py    # ANN → rerank → threshold → HyDE expansion
│   │   ├── reranker.py     # Cross-encoder (ms-marco-MiniLM-L6-v2)
│   │   └── query_expander.py # HyDE query expansion
│   ├── diagnosis/          # LLM reasoning module
│   │   ├── llm_client.py   # OpenAI / Anthropic / Azure OpenAI
│   │   └── diagnoser.py    # 8-step diagnosis workflow
│   └── api/                # FastAPI REST service
│       ├── main.py
│       └── models.py
├── config/
│   ├── config.example.yaml # Template — copy to config.yaml
│   └── prometheus.yml
├── data/samples/
│   ├── incidents/          # Sample incident JSON files
│   ├── runbooks/           # Sample runbook Markdown files
│   ├── post_mortems/       # Sample post-mortem reports
│   └── eval_dataset.json   # 8-sample evaluation dataset
├── scripts/
│   ├── index_knowledge_base.py
│   ├── diagnose_incident.py
│   └── evaluate.py
├── tests/
│   ├── test_chunker.py
│   ├── test_retriever.py
│   └── test_diagnoser.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` inside your venv |
| `FileNotFoundError: config/config.yaml` | Run `cp config/config.example.yaml config/config.yaml` |
| `AuthenticationError: OpenAI` | Check your API key in `config/config.yaml` |
| `faiss not found` | Run `pip install faiss-cpu` |
| `No documents found` | Check `--data-dir` points to a directory containing `.json` or `.md` files |
| Empty retrieval results | Run `index_knowledge_base.py` first before diagnosing |
| Docker port conflict | Change port mapping in `docker-compose.yml` (e.g. `"8001:8000"`) |

---

*For issues or questions, open a GitHub Issue at
[github.com/yogitapatil74/Incident-Diagnosis-with-RAG](https://github.com/yogitapatil74/Incident-Diagnosis-with-RAG)*