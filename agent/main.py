"""
Entry point for the AMD Hackathon Track 1 AI Agent.

Strategy: 100% LOCAL — zero Fireworks tokens.
  All tasks go to local Ollama model (qwen2.5:7b).
  No Fireworks API fallback. Retries locally on failure.

Reads:  /input/tasks.json  (or INPUT_PATH env var)
Writes: /output/results.json (or OUTPUT_PATH env var)
Exits:  0 on success, 1 on failure
"""

import json
import logging
import os
import sys
import time

# ── Load .env for local development (no-op in Docker where env is injected) ──
try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except ImportError:
    pass

import requests

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
INPUT_PATH = os.environ.get("INPUT_PATH", "/input/tasks.json")
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "/output/results.json")

# ---------------------------------------------------------------------------
# Local model config (Ollama)
# ---------------------------------------------------------------------------
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
LOCAL_MODEL = os.environ.get("LOCAL_MODEL", "qwen2.5:7b")
LOCAL_TIMEOUT = int(os.environ.get("LOCAL_TIMEOUT_SEC", "300"))
MAX_RETRIES = 3

# ---------------------------------------------------------------------------
# Universal system prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a highly capable AI assistant. "
    "Follow the user's instructions precisely and completely. "
    "Output ONLY what is asked for — no preamble, no thinking steps, "
    "no meta-commentary, no restating the question. "
    "If the task asks for a specific format, use that exact format. "
    "Be concise and accurate."
)

MAX_TOKENS = 1024


# ---------------------------------------------------------------------------
# Local Ollama inference
# ---------------------------------------------------------------------------
def _wait_for_ollama(max_wait: int = 120) -> bool:
    """Poll until Ollama server is ready."""
    deadline = time.time() + max_wait
    while time.time() < deadline:
        try:
            r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def _warmup_model() -> None:
    """
    Send a tiny request to force the model to load into memory.
    First inference is always slow (model loading); this ensures
    actual tasks don't time out waiting for the cold start.
    """
    logger.info("Warming up local model '%s'...", LOCAL_MODEL)
    try:
        resp = requests.post(
            f"{OLLAMA_HOST}/api/chat",
            json={
                "model": LOCAL_MODEL,
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": False,
                "options": {"num_predict": 1, "temperature": 0.0},
            },
            timeout=300,  # model loading can take a while on CPU
        )
        resp.raise_for_status()
        logger.info("Model warmup complete.")
    except Exception as exc:
        logger.warning("Model warmup failed: %s (will retry on first task)", exc)


def call_local(prompt: str) -> str:
    """
    Call local Ollama model with retries. Never falls back to Fireworks.
    Raises RuntimeError only if all retries exhausted.
    """
    payload = {
        "model": LOCAL_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {
            "num_predict": MAX_TOKENS,
            "temperature": 0.0,
        },
    }

    last_exc: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(
                f"{OLLAMA_HOST}/api/chat",
                json=payload,
                timeout=LOCAL_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            content = data["message"]["content"].strip()
            if content:
                return content
            logger.warning("Attempt %d: local model returned empty content, retrying...", attempt)
        except requests.Timeout:
            logger.warning("Attempt %d: local model timed out after %ss, retrying...", attempt, LOCAL_TIMEOUT)
            last_exc = TimeoutError(f"Timeout after {LOCAL_TIMEOUT}s")
        except Exception as exc:
            logger.warning("Attempt %d: local model error: %s, retrying...", attempt, exc)
            last_exc = exc

        # Brief pause before retry
        if attempt < MAX_RETRIES:
            time.sleep(2)

    raise RuntimeError(f"Local model failed after {MAX_RETRIES} attempts. Last error: {last_exc}")


# ---------------------------------------------------------------------------
# Task processing
# ---------------------------------------------------------------------------
def load_tasks(path: str) -> list[dict]:
    logger.info("Loading tasks from %s", path)
    with open(path, "r", encoding="utf-8") as f:
        tasks = json.load(f)
    logger.info("Loaded %d task(s).", len(tasks))
    return tasks


def save_results(results: list[dict], path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("Results written to %s", path)


def process_task(task: dict) -> dict:
    task_id = task["task_id"]
    prompt = task["prompt"]

    logger.info("--- Processing task: %s ---", task_id)

    start = time.monotonic()
    answer = call_local(prompt)
    elapsed = time.monotonic() - start
    logger.info("Task '%s' answered LOCALLY in %.1fs (0 tokens).", task_id, elapsed)

    return {"task_id": task_id, "answer": answer}


def main() -> int:
    # Wait for Ollama to be ready
    logger.info("Waiting for Ollama server...")
    if not _wait_for_ollama(max_wait=120):
        logger.error("Ollama server not available after 120s. Cannot proceed.")
        return 1

    # Warm up model (load into memory)
    _warmup_model()

    try:
        tasks = load_tasks(INPUT_PATH)
    except FileNotFoundError:
        logger.error("Input file not found: %s", INPUT_PATH)
        return 1
    except json.JSONDecodeError as exc:
        logger.error("Invalid JSON in input: %s", exc)
        return 1

    results: list[dict] = []
    failed: list[str] = []

    for task in tasks:
        try:
            result = process_task(task)
            results.append(result)
        except Exception as exc:
            task_id = task.get("task_id", "?")
            logger.error("Task '%s' failed: %s", task_id, exc, exc_info=True)
            failed.append(task_id)
            results.append({"task_id": task_id, "answer": ""})

    try:
        save_results(results, OUTPUT_PATH)
    except Exception as exc:
        logger.error("Failed to write output: %s", exc)
        return 1

    if failed:
        logger.warning("%d task(s) failed: %s", len(failed), failed)

    logger.info("All %d task(s) completed. %d failed.", len(tasks), len(failed))
    return 0


if __name__ == "__main__":
    sys.exit(main())
