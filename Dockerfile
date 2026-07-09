# AMD Hackathon Track 1 — AI Agent
# ─────────────────────────────────────────────────────────────
# Base: slim Python 3.11 for small image size
FROM python:3.11-slim

# ── System deps ──────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        zstd \
    && rm -rf /var/lib/apt/lists/*

# ── Install Ollama ────────────────────────────────────────────
# Official install script — places binary at /usr/local/bin/ollama
RUN curl -fsSL https://ollama.ai/install.sh | sh

# ── Python dependencies ───────────────────────────────────────
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy agent code ───────────────────────────────────────────
COPY agent/ /app/agent/

# ── Entrypoint ────────────────────────────────────────────────
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Ensure output dir exists (it will also be mounted at runtime)
RUN mkdir -p /input /output

ENTRYPOINT ["/app/entrypoint.sh"]
