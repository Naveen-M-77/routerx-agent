#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# Entrypoint for AMD Hackathon Track 1 Agent
# 1. Starts Ollama server in background
# 2. Pulls the local model (or uses cached layers)
# 3. Runs the Python agent
# ─────────────────────────────────────────────────────────────
set -euo pipefail

echo "[entrypoint] Starting Ollama server..."
ollama serve &
OLLAMA_PID=$!

# Wait for Ollama to be ready (up to 30s)
echo "[entrypoint] Waiting for Ollama to be ready..."
for i in $(seq 1 15); do
    if curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "[entrypoint] Ollama is ready."
        break
    fi
    sleep 2
done

# Pull local model (fast if layers already cached in image)
LOCAL_MODEL="${LOCAL_MODEL:-qwen2.5:1.5b}"
echo "[entrypoint] Pulling model: ${LOCAL_MODEL}"
ollama pull "${LOCAL_MODEL}" || echo "[entrypoint] WARNING: Failed to pull local model — will use Fireworks only."

echo "[entrypoint] Starting agent..."
python -m agent.main
EXIT_CODE=$?

# Gracefully stop Ollama
kill "${OLLAMA_PID}" 2>/dev/null || true

echo "[entrypoint] Agent exited with code ${EXIT_CODE}."
exit "${EXIT_CODE}"
