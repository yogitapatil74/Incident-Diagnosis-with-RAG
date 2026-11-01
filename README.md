# RAG-Based AIOps Framework for Automated Incident Diagnosis

[![CI](https://github.com/yogitapatil74/Incident-Diagnosis-with-RAG/actions/workflows/ci.yml/badge.svg)](https://github.com/yogitapatil74/Incident-Diagnosis-with-RAG/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

This repository contains a Proof-of-Concept (PoC) implementation of a RAG-based AIOps
framework for automated incident diagnosis. It demonstrates:

- **5-layer architecture**: Data Sources → Ingestion → Indexing → Retrieval → LLM Reasoning
- **Knowledge indexing pipeline**: Chunking, embedding generation, FAISS vector indexing
- **Real-time incident diagnosis**: Semantic retrieval + LLM-generated root cause analysis
- **REST API**: FastAPI-based service for integration with incident management tools
- **Evaluation harness**: Accuracy, precision/recall, and ablation-study metrics

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     RAG-AIOps Framework                         │
├──────────────┬──────────────┬──────────────┬────────────────────┤
│  Ingestion   │   Indexing   │  Retrieval   │   LLM Reasoning    │
│              │              │              │                    │
│ • Tickets    │ • Chunking   │ • FAISS ANN  │ • GPT-4 / Claude   │
│ • Runbooks   │ • Embeddings │ • Re-ranking │ • Chain-of-thought  │
│ • Post-mort. │ • Metadata   │ • HyDE expan │ • Structured output │
│ • Alerts     │ • FAISS idx  │ • Filtering  │ • Confidence score  │
└──────────────┴──────────────┴──────────────┴────────────────────┘
```

---

## Quick Start

### Prerequisites
- Python 3.10+
- Docker & Docker Compose (optional, for full stack)
- OpenAI API key (or Azure OpenAI endpoint)

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure

```bash
cp config/config.example.yaml config/config.yaml
# Edit config/config.yaml with your API keys and settings
```

### 3. Index sample knowledge base

```bash
python scripts/index_knowledge_base.py --data-dir data/samples
```

### 4. Run diagnosis on a sample incident

```bash
python scripts/diagnose_incident.py --incident data/samples/incidents/sample_incidents.json
```

### 5. Start the REST API

```bash
uvicorn src.api.main:app --reload --port 8000
# API docs: http://localhost:8000/docs
```

### 6. Run evaluation

```bash
python scripts/evaluate.py --dataset data/samples/eval_dataset.json
```

---

## Docker Deployment

```bash
docker-compose up --build
```

Services started:
- `aiops-api` — FastAPI service on port 8000
- `redis` — Query embedding cache on port 6379
- `prometheus` — Metrics on port 9090
- `grafana` — Dashboards on port 3000 (admin / admin)

---

## Project Structure

```
Incident-Diagnosis-with-RAG/
├── src/
│   ├── ingestion/          # Data source connectors
│   │   ├── base.py         # OperationalDocument schema + BaseIngester
│   │   ├── ticket_ingester.py   # Local JSON/CSV + ServiceNow
│   │   └── runbook_ingester.py  # Local Markdown + Confluence
│   ├── indexing/           # Knowledge indexing pipeline
│   │   ├── chunker.py      # Document chunking strategies
│   │   ├── embedder.py     # Embedding generation
│   │   └── vector_store.py # FAISS vector store wrapper
│   ├── retrieval/          # Semantic retrieval engine
│   │   ├── retriever.py    # Main retrieval logic
│   │   ├── reranker.py     # Cross-encoder re-ranking
│   │   └── query_expander.py # HyDE query expansion
│   ├── diagnosis/          # LLM reasoning module
│   │   ├── llm_client.py
│   │   └── diagnoser.py
│   └── api/                # FastAPI REST interface
│       ├── main.py
│       └── models.py
├── config/
│   ├── config.example.yaml
│   └── config.yaml         # (gitignored — contains secrets)
├── data/
│   └── samples/            # Sample incidents, runbooks, tickets
├── scripts/
│   ├── index_knowledge_base.py
│   ├── diagnose_incident.py
│   ├── evaluate.py
│   └── generate_config.py
├── tests/
│   ├── test_chunker.py
│   ├── test_retriever.py
│   └── test_diagnoser.py
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Contributing

Contributions are welcome — bug fixes, tests, docs, and new ingestion connectors alike.
See [CONTRIBUTING.md](CONTRIBUTING.md) for setup instructions and PR guidelines, and
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for community expectations.

---

## License

MIT License — see [LICENSE](LICENSE) for details.
