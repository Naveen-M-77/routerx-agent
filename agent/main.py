"""
Entry point for the AMD Hackathon Track 1 AI Agent.

Strategy: LOCAL-FIRST for zero token cost.
  1. All tasks go to local Ollama model (qwen2.5:7b) — 0 Fireworks tokens
  2. Only falls back to Fireworks API if local model is unavailable or fails

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
    load_dotenv(override=False)
except ImportError:
    pass

import requests
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
# Paths
# ---------------------------------------------------------------------------
INPUT_PATH = os.environ.get("INPUT_PATH", "/input/tasks.json")
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "/output/results.json")

# ---------------------------------------------------------------------------
# Local model config (Ollama)
# ---------------------------------------------------------------------------
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
LOCAL_MODEL = os.environ.get("LOCAL_MODEL", "qwen2.5:7b")
LOCAL_TIMEOUT = int(os.environ.get("LOCAL_TIMEOUT_SEC", "120"))

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
def _ollama_ready() -> bool:
    """Check if Ollama server is running."""
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def _wait_for_ollama(max_wait: int = 60) -> bool:
    """Poll until Ollama server is ready."""
    deadline = time.time() + max_wait
    while time.time() < deadline:
        if _ollama_ready():
            return True
        time.sleep(2)
    return False


def call_local(prompt: str) -> str | None:
    """
    Call local Ollama model. Returns answer or None on failure.
    Zero Fireworks token cost.
    """
    if not _ollama_ready():
        logger.info("Ollama not available — skipping local model.")
        return None

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
        logger.warning("Local model returned empty content.")
        return None
    except requests.Timeout:
        logger.warning("Local model timed out after %ss.", LOCAL_TIMEOUT)
        return None
    except Exception as exc:
        logger.warning("Local model error: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Fireworks API fallback
# ---------------------------------------------------------------------------
_fw_client: OpenAI | None = None


def _get_fw_client() -> OpenAI:
    global _fw_client
    if _fw_client is None:
        api_key = os.environ.get("FIREWORKS_API_KEY", "")
        base_url = os.environ.get("FIREWORKS_BASE_URL", "")
        if not api_key or not base_url:
            raise RuntimeError("Fireworks API not configured.")
        _fw_client = OpenAI(api_key=api_key, base_url=base_url)
    return _fw_client


def _rank_models_best_first(models: list[str]) -> list[str]:
    """Rank models: direct-answer models first, thinking models last."""
    def score(name: str) -> int:
        n = name.lower()
        if "gpt-oss" in n or "120" in n:
            return 10
        if "deepseek" in n:
            return 20
        if "kimi" in n or "glm" in n:
            return 100
        return 50
    return sorted(models, key=score)


def _clean_output(content: str) -> str:
    """Strip thinking model artifacts from output."""
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

    lines = content.split("\n")
    thinking_markers = [
        r"^\s*The user wants",
        r"^\s*I need to",
        r"^\s*Let me (analyze|check|think|reason|break|consider|look|start|figure|verify)",
    ]
    first_lines = "\n".join(lines[:5])
    is_thinking = any(re.search(p, first_lines, re.IGNORECASE) for p in thinking_markers)

    if not is_thinking:
        return content.strip()

    noise_patterns = [
        re.compile(r"^\s*(The user|I need to|Let me|Wait,|Actually,|Hmm|OK|So,|Now,|This (could|should|is|means)|Or |Possible)", re.IGNORECASE),
        re.compile(r"^\s*\d+\.\s+\*\*"),
        re.compile(r"^\s*Step\s+\d+", re.IGNORECASE),
    ]

    result_lines: list[str] = []
    found_content = False

    for line in reversed(lines):
        stripped = line.strip()
        if not stripped:
            if found_content:
                result_lines.append(line)
            continue
        is_noise = any(p.match(stripped) for p in noise_patterns)
        if is_noise and found_content:
            break
        result_lines.append(line)
        found_content = True

    if result_lines:
        result_lines.reverse()
        while result_lines and not result_lines[0].strip():
            result_lines.pop(0)
        while result_lines and not result_lines[-1].strip():
            result_lines.pop()
        if result_lines:
            return "\n".join(result_lines).strip()

    return content.strip()


def call_fireworks(prompt: str) -> str:
    """Fireworks API fallback — only used if local model fails."""
    raw = os.environ.get("ALLOWED_MODELS", "")
    models = _rank_models_best_first([m.strip() for m in raw.split(",") if m.strip()])
    if not models:
        raise RuntimeError("ALLOWED_MODELS is empty or not set.")

    client = _get_fw_client()
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
                raise ValueError(f"Model '{model}' returned null content")
            return _clean_output(content)
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

    # Try local model first (zero Fireworks tokens)
    answer = call_local(prompt)
    if answer:
        elapsed = time.monotonic() - start
        logger.info("Task '%s' answered LOCALLY in %.1fs (0 tokens).", task_id, elapsed)
        return {"task_id": task_id, "answer": answer}

    # Fallback to Fireworks API
    logger.info("Task '%s': falling back to Fireworks API.", task_id)
    answer = call_fireworks(prompt)
    elapsed = time.monotonic() - start
    logger.info("Task '%s' answered via Fireworks in %.1fs.", task_id, elapsed)

    return {"task_id": task_id, "answer": answer}


def main() -> int:
    # Wait for Ollama if available
    if _wait_for_ollama(max_wait=30):
        logger.info("Ollama is ready — local-first mode enabled.")
    else:
        logger.warning("Ollama not available — will use Fireworks API only.")

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
