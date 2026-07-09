"""
Entry point for the AMD Hackathon Track 1 AI Agent.

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
    load_dotenv(override=False)   # don't override vars already in the environment
except ImportError:
    pass  # python-dotenv not installed — running in Docker, env already set

from agent.classifier import classify
from agent.router import route

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
# Paths — defaults match Docker mount points; override via env / .env
# ---------------------------------------------------------------------------
INPUT_PATH = os.environ.get("INPUT_PATH", "/input/tasks.json")
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "/output/results.json")


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
    category = classify(prompt)
    logger.info("Classified '%s' as: %s", task_id, category)

    start = time.monotonic()
    answer = route(category, prompt)
    elapsed = time.monotonic() - start
    logger.info("Task '%s' answered in %.1fs (category=%s).", task_id, elapsed, category)

    return {"task_id": task_id, "answer": answer}


def main() -> int:
    # Validate required env vars early
    missing = [v for v in ("FIREWORKS_API_KEY", "FIREWORKS_BASE_URL", "ALLOWED_MODELS")
               if not os.environ.get(v)]
    if missing:
        logger.warning(
            "Missing env vars: %s. Fireworks calls may fail. "
            "For local testing, set these in your .env file.",
            ", ".join(missing),
        )

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
            # Still emit a placeholder so the output schema is valid
            results.append({"task_id": task_id, "answer": f"[ERROR: {exc}]"})

    try:
        save_results(results, OUTPUT_PATH)
    except Exception as exc:
        logger.error("Failed to write output: %s", exc)
        return 1

    if failed:
        logger.warning("%d task(s) failed: %s", len(failed), failed)
        return 1

    logger.info("All %d task(s) completed successfully.", len(tasks))
    return 0


if __name__ == "__main__":
    sys.exit(main())
