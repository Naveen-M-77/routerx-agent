#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# Entrypoint for AMD Hackathon Track 1 Agent
# 1. Starts Ollama server in background
# 2. Waits for it to be ready (model already baked into image)
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

# Model is pre-baked in the image — no pull needed at runtime
LOCAL_MODEL="${LOCAL_MODEL:-qwen2.5:7b}"
echo "[entrypoint] Using pre-loaded model: ${LOCAL_MODEL}"

echo "[entrypoint] Starting agent..."
python -m agent.main
EXIT_CODE=$?

# Gracefully stop Ollama
kill "${OLLAMA_PID}" 2>/dev/null || true

echo "[entrypoint] Agent exited with code ${EXIT_CODE}."
exit "${EXIT_CODE}"
