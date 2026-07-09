# AMD Hackathon Track 1 — General-Purpose AI Agent

A smart-routing AI agent that handles 8 task categories using local models (zero cost) and Fireworks AI API (minimal tokens).

## Architecture

```
Task → Classifier (regex, no API) → Router
                                      ├─ EASY  → Ollama local model (0 tokens)
                                      ├─ MED   → Smallest Fireworks model
                                      └─ HARD  → Larger Fireworks model (fallback)
```

### Capability Categories
| # | Category | Routing |
|---|----------|---------|
| 1 | Factual knowledge | Small Fireworks |
| 2 | Mathematical reasoning | Small Fireworks |
| 3 | Sentiment classification | **Local (free)** |
| 4 | Text summarization | **Local (free)** |
| 5 | Named entity recognition | **Local (free)** |
| 6 | Code debugging | Small Fireworks |
| 7 | Logical/deductive reasoning | Small Fireworks |
| 8 | Code generation | Large Fireworks |

## Local Development

### Prerequisites
- Docker Desktop
- A Fireworks AI API key (for end-to-end testing)

### Setup

```bash
# Clone / enter the project
cd amd-hackathon

# Copy and edit your env file
cp .env.example .env
# Edit .env with your real FIREWORKS_API_KEY and ALLOWED_MODELS

# Install Python deps (for local testing without Docker)
pip install -r requirements.txt

# Run the classifier smoke test (no API needed)
python tests/test_classifier.py
```

### Run locally without Docker

```bash
# Set env vars
set -a; source .env; set +a   # Linux/Mac
# OR on Windows: set FIREWORKS_API_KEY=... etc.

# Override paths to local dirs
export INPUT_PATH=./input/tasks.json
export OUTPUT_PATH=./output/results.json

python -m agent.main
cat output/results.json
```

### Build & run with Docker

```bash
# Build for linux/amd64 (required by hackathon harness)
docker build --platform linux/amd64 -t amd-hackathon-agent .

# Run with practice tasks
docker run --platform linux/amd64 \
  --env-file .env \
  -v "$(pwd)/input:/input" \
  -v "$(pwd)/output:/output" \
  amd-hackathon-agent

# Check results
cat output/results.json
```

### Push to registry

```bash
# Docker Hub
docker tag amd-hackathon-agent <your-dockerhub-username>/amd-hackathon-agent:latest
docker push <your-dockerhub-username>/amd-hackathon-agent:latest

# GitHub Container Registry
docker tag amd-hackathon-agent ghcr.io/<your-github-username>/amd-hackathon-agent:latest
docker push ghcr.io/<your-github-username>/amd-hackathon-agent:latest
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `FIREWORKS_API_KEY` | Injected by harness — do not hardcode |
| `FIREWORKS_BASE_URL` | Base URL for all Fireworks calls |
| `ALLOWED_MODELS` | Comma-separated list of permitted model IDs |
| `LOCAL_MODEL` | Ollama model name (default: `qwen2.5:1.5b`) |
| `LOCAL_TIMEOUT_SEC` | Timeout for local model calls in seconds (default: `45`) |
| `DISABLE_LOCAL_MODEL` | Set to `1` to skip Ollama entirely |
| `INPUT_PATH` | Override input file path (default: `/input/tasks.json`) |
| `OUTPUT_PATH` | Override output file path (default: `/output/results.json`) |

## Token Optimization

- **Easy tasks** (sentiment, NER, summarization) → Ollama local model → **0 Fireworks tokens**
- **System prompts** are kept minimal per category
- **`max_tokens`** is capped per category (50–600) to limit output cost
- **Model selection** at runtime: smallest model attempted first, escalates on failure

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `PULL_ERROR` | Make sure image is public and has `linux/amd64` manifest |
| `RUNTIME_ERROR` | Check container logs: `docker logs <container>` |
| `TIMEOUT` | Increase `LOCAL_TIMEOUT_SEC` or set `DISABLE_LOCAL_MODEL=1` |
| `OUTPUT_MISSING` | Check `/output` volume mount |
| `ACCURACY_GATE_FAILED` | Tune routing — route more tasks to Fireworks |
| `MODEL_VIOLATION` | Never hardcode model IDs — read from `ALLOWED_MODELS` |
