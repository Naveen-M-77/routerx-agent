"""
Entry point for the AMD Hackathon Track 1 AI Agent.

Reads:  /input/tasks.json  (or INPUT_PATH env var)
Writes: /output/results.json (or OUTPUT_PATH env var)
Exits:  0 on success, 1 on failure
"""

import json
import logging
import os
import re
import sys
import time

# ── Load .env for local development (no-op in Docker where env is injected) ──
try:
    from dotenv import load_dotenv
    load_dotenv(override=False)   # don't override vars already in the environment
except ImportError:
    pass  # python-dotenv not installed — running in Docker, env already set

from openai import OpenAI

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

# ---------------------------------------------------------------------------
# Universal system prompt — lets the model follow the task's own instructions
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
# Fireworks client
# ---------------------------------------------------------------------------
_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.environ["FIREWORKS_API_KEY"]
        base_url = os.environ["FIREWORKS_BASE_URL"]
        _client = OpenAI(api_key=api_key, base_url=base_url)
    return _client


def get_allowed_models() -> list[str]:
    raw = os.environ.get("ALLOWED_MODELS", "")
    return [m.strip() for m in raw.split(",") if m.strip()]


def _rank_models_best_first(models: list[str]) -> list[str]:
    """Rank models: direct-answer models first, thinking models last."""
    def score(name: str) -> int:
        n = name.lower()
        # gpt-oss-120b — large, direct answers
        if "gpt-oss" in n or "120" in n:
            return 10
        # deepseek-v4-pro — large, direct answers
        if "deepseek" in n:
            return 20
        # kimi/glm — thinking models, deprioritise
        if "kimi" in n or "glm" in n:
            return 100
        return 50
    return sorted(models, key=score)


def _clean_output(content: str) -> str:
    """Strip thinking model artifacts from output."""
    # Remove <think>...</think> blocks
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

    lines = content.split("\n")
    # Detect thinking preamble
    thinking_markers = [
        r"^\s*The user wants",
        r"^\s*I need to",
        r"^\s*Let me (analyze|check|think|reason|break|consider|look|start|figure|verify|re-read)",
        r"^\s*\d+\.\s+\*\*Analyze",
        r"^\s*\d+\.\s+\*\*Retrieve",
        r"^\s*\d+\.\s+\*\*Calculate",
        r"^\s*\d+\.\s+\*\*Format",
    ]
    first_lines = "\n".join(lines[:5])
    is_thinking = any(re.search(p, first_lines, re.IGNORECASE) for p in thinking_markers)

    if not is_thinking:
        return content.strip()

    # Extract final answer block from thinking output.
    # Strategy: Find content after the last "noise" section.
    # Look for the final substantive output.

    noise_patterns = [
        re.compile(r"^\s*(The user|I need to|Let me|Wait,|Actually,|Hmm|OK|So,|Now,|This (could|should|is|means)|Or |Possible|That's )", re.IGNORECASE),
        re.compile(r"^\s*\d+\.\s+\*\*"),   # numbered bold steps
        re.compile(r"^\s*Step\s+\d+", re.IGNORECASE),
    ]

    # Walk backwards from the end to find the last block of non-noise content
    # This is the actual answer the model intended to output
    result_lines: list[str] = []
    found_content = False

    for line in reversed(lines):
        stripped = line.strip()
        if not stripped:
            if found_content:
                # Empty line while we have content — check if we've reached noise above
                result_lines.append(line)
            continue

        is_noise = any(p.match(stripped) for p in noise_patterns)
        if is_noise and found_content:
            # We've hit the reasoning section — stop here
            break

        result_lines.append(line)
        found_content = True

    if result_lines:
        result_lines.reverse()
        # Strip leading/trailing empty lines
        while result_lines and not result_lines[0].strip():
            result_lines.pop(0)
        while result_lines and not result_lines[-1].strip():
            result_lines.pop()
        if result_lines:
            return "\n".join(result_lines).strip()

    return content.strip()


def call_fireworks(prompt: str) -> str:
    """
    Call the best available Fireworks model with the universal prompt.
    Tries models in priority order, falls back on failure.
    """
    models = _rank_models_best_first(get_allowed_models())
    if not models:
        raise RuntimeError("ALLOWED_MODELS is empty or not set.")

    client = _get_client()
    last_exc: Exception | None = None

    for model in models:
        try:
            logger.info("Calling Fireworks model: %s", model)
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=MAX_TOKENS,
                temperature=0.0,
            )
            content = response.choices[0].message.content
            if content is None:
                raise ValueError(
                    f"Model '{model}' returned null content "
                    f"(finish_reason={response.choices[0].finish_reason})"
                )
            answer = _clean_output(content)
            logger.info("Model %s answered successfully.", model)
            return answer
        except Exception as exc:
            logger.warning("Model %s failed: %s — trying next.", model, exc)
            last_exc = exc

    raise RuntimeError(f"All Fireworks models failed. Last error: {last_exc}")


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
    answer = call_fireworks(prompt)
    elapsed = time.monotonic() - start
    logger.info("Task '%s' answered in %.1fs.", task_id, elapsed)

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
