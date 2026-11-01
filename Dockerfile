# ── Build stage ────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# ── Runtime stage ───────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Runtime deps: libgomp (FAISS), redis-tools (redis-cli health check)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    redis-tools \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /root/.local /root/.local

# Copy application
COPY src/          ./src/
COPY scripts/      ./scripts/
COPY config/       ./config/
COPY data/samples/ ./data/samples/
COPY entrypoint.sh ./entrypoint.sh

# Create runtime directories
RUN mkdir -p data/faiss_index data/eval_results config

# Make scripts executable
RUN chmod +x entrypoint.sh scripts/*.py

ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/v1/health')" || exit 1

ENTRYPOINT ["./entrypoint.sh"]