#!/bin/bash
set -e

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║       RAG-AIOps Incident Diagnosis Framework             ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# ── Validate required env vars ─────────────────────────────────
if [ -z "$OPENAI_API_KEY" ] && [ "$EMBEDDING_PROVIDER" != "local" ]; then
  echo "⚠  OPENAI_API_KEY not set — switching to local embeddings"
  export EMBEDDING_PROVIDER="local"
fi

# ── Wait for Redis ──────────────────────────────────────────────
echo "⏳ Waiting for Redis..."
until redis-cli -h "${REDIS_HOST:-redis}" -p "${REDIS_PORT:-6379}" ping > /dev/null 2>&1; do
  sleep 1
done
echo "✓  Redis ready"

# ── Generate config.yaml from environment ──────────────────────
echo "⚙️  Generating config from environment variables..."
python /app/scripts/generate_config.py
echo "✓  Config generated"

# ── Index knowledge base (skip if already indexed) ─────────────
INDEX_PATH="data/faiss_index/index.faiss"
if [ ! -f "$INDEX_PATH" ] || [ "${FORCE_REINDEX:-false}" = "true" ]; then
  echo ""
  echo "📚 Indexing knowledge base..."
  # Always do a full (blue-green) rebuild here. An incremental add on
  # FORCE_REINDEX kept stale chunks from edited/removed documents and could
  # not recover from an index built with a different embedding dimension.
  INDEX_ARGS="--data-dir data/samples --rebuild"
  if [ "$EMBEDDING_PROVIDER" = "local" ]; then
    INDEX_ARGS="$INDEX_ARGS --local-embeddings"
  fi
  python scripts/index_knowledge_base.py $INDEX_ARGS
  echo "✓  Knowledge base indexed"
else
  echo "✓  Knowledge base already indexed (set FORCE_REINDEX=true to rebuild)"
fi

# ── Start API ───────────────────────────────────────────────────
echo ""
echo "🚀 Starting RAG-AIOps API on port ${API_PORT:-8000}..."
echo "   Swagger UI → http://localhost:${API_PORT:-8000}/docs"
echo ""

exec uvicorn src.api.main:app \
  --host 0.0.0.0 \
  --port "${API_PORT:-8000}" \
  --workers "${API_WORKERS:-2}"