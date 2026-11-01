#!/usr/bin/env python3
"""
scripts/generate_config.py
───────────────────────────
Generates config/config.yaml at container startup from environment variables.
This enables single-command docker-compose runs without manually editing config files.

Environment variables (all optional — sensible defaults provided):

  LLM / Embeddings:
    OPENAI_API_KEY          OpenAI API key
    ANTHROPIC_API_KEY       Anthropic API key (alternative LLM)
    LLM_PROVIDER            openai | anthropic | azure_openai  (default: openai)
    LLM_MODEL               default: gpt-4-turbo
    EMBEDDING_PROVIDER      openai | local  (default: openai, falls back to local)
    EMBEDDING_MODEL         default: text-embedding-ada-002

  Infrastructure:
    REDIS_HOST              default: redis
    REDIS_PORT              default: 6379

  App:
    API_PORT                default: 8000
    LOG_LEVEL               default: INFO
    FORCE_REINDEX           default: false
"""

import os
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent


def env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def build_config() -> dict:
    embedding_provider = env("EMBEDDING_PROVIDER", "openai")
    # Auto-fallback to local if no OpenAI key
    if embedding_provider == "openai" and not env("OPENAI_API_KEY"):
        print("  No OPENAI_API_KEY found — using local embeddings")
        embedding_provider = "local"

    return {
        "app": {
            "name": "RAG-AIOps Incident Diagnosis Framework",
            "version": "1.0.0",
            "environment": env("APP_ENV", "production"),
            "log_level": env("LOG_LEVEL", "INFO"),
        },

        "llm": {
            "provider": env("LLM_PROVIDER", "openai"),
            "model": env("LLM_MODEL", "gpt-4-turbo"),
            "max_tokens": 1500,
            "temperature": 0.1,
            "timeout_seconds": 30,
            "openai": {
                "api_key": env("OPENAI_API_KEY", ""),
                "organization": env("OPENAI_ORG", ""),
            },
            "anthropic": {
                "api_key": env("ANTHROPIC_API_KEY", ""),
                "model": env("ANTHROPIC_MODEL", "claude-3-opus-20240229"),
            },
            "azure_openai": {
                "api_key": env("AZURE_OPENAI_KEY", ""),
                "endpoint": env("AZURE_OPENAI_ENDPOINT", ""),
                "deployment_name": env("AZURE_OPENAI_DEPLOYMENT", "gpt-4-turbo"),
                "api_version": "2024-02-01",
            },
        },

        "embedding": {
            "provider": embedding_provider,
            "model": env("EMBEDDING_MODEL", "text-embedding-ada-002"),
            "dimensions": 1536 if embedding_provider == "openai" else 768,
            "batch_size": 100,
            "local_model": env(
                "LOCAL_EMBEDDING_MODEL",
                "sentence-transformers/all-mpnet-base-v2"
            ),
        },

        "vector_store": {
            "index_path": "data/faiss_index",
            "index_type": "HNSW",
            "hnsw_m": 32,
            "hnsw_ef_construction": 400,
            "hnsw_ef_search": 64,
            "similarity_metric": "cosine",
        },

        "retrieval": {
            "top_k": 10,
            "rerank_top_k": 5,
            "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "rerank_threshold": 0.45,
            "min_docs_before_expansion": 3,
            "hyde_enabled": True,
            "filter_fallback": True,
        },

        "chunking": {
            "default_strategy": "sliding_window",
            "chunk_size": 512,
            "chunk_overlap": 50,
        },

        "cache": {
            "enabled": True,
            "host": env("REDIS_HOST", "redis"),
            "port": int(env("REDIS_PORT", "6379")),
            "password": env("REDIS_PASSWORD", ""),
            "ttl_seconds": 3600,
            "max_memory": "256mb",
        },

        "ingestion": {
            "local_files": {
                "enabled": True,
                "paths": ["data/samples"],
                "recursive": True,
            },
            "servicenow": {"enabled": False},
            "pagerduty": {"enabled": False},
            "confluence": {"enabled": False},
            "github": {"enabled": False},
        },

        "api": {
            "host": "0.0.0.0",
            "port": int(env("API_PORT", "8000")),
            "workers": int(env("API_WORKERS", "2")),
            "cors_origins": ["*"],
            "rate_limit_per_minute": 60,
        },

        "evaluation": {
            "dataset_path": "data/samples/eval_dataset.json",
            "metrics": ["accuracy", "precision_at_1", "recall_at_3", "mdt"],
            "output_dir": "data/eval_results",
        },
    }


def main():
    config_path = ROOT / "config" / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)

    config = build_config()

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"  Config written to {config_path}")
    print(f"  LLM provider:        {config['llm']['provider']}")
    print(f"  Embedding provider:  {config['embedding']['provider']}")
    print(f"  Embedding dims:      {config['embedding']['dimensions']}")
    print(f"  Redis:               {config['cache']['host']}:{config['cache']['port']}")


if __name__ == "__main__":
    main()
