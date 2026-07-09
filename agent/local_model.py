"""
Local model interface via Ollama HTTP API.

Falls back gracefully (returns None) on timeout or failure so the
router can escalate to Fireworks.
"""

import logging
import os
import subprocess
import time

import requests

logger = logging.getLogger(__name__)

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
LOCAL_MODEL = os.environ.get("LOCAL_MODEL", "qwen2.5:1.5b")
LOCAL_TIMEOUT = int(os.environ.get("LOCAL_TIMEOUT_SEC", "45"))


def _wait_for_ollama(max_wait: int = 60) -> bool:
    """Poll until Ollama HTTP server is ready."""
    deadline = time.time() + max_wait
    while time.time() < deadline:
        try:
            r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=3)
            if r.status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(2)
    return False


def ensure_model_available() -> bool:
    """
    Check Ollama is running and the local model is loaded.
    Returns True if ready, False otherwise.
    """
    if not _wait_for_ollama(max_wait=60):
        logger.warning("Ollama server not reachable — local model disabled.")
        return False

    # Check if model already pulled
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        if any(LOCAL_MODEL in m for m in models):
            logger.info("Local model '%s' already available.", LOCAL_MODEL)
            return True
    except Exception as exc:
        logger.warning("Could not list Ollama models: %s", exc)

    # Try to pull
    logger.info("Pulling local model '%s'…", LOCAL_MODEL)
    try:
        result = subprocess.run(
            ["ollama", "pull", LOCAL_MODEL],
            timeout=300,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            logger.info("Model pulled successfully.")
            return True
        else:
            logger.warning("ollama pull failed: %s", result.stderr)
            return False
    except Exception as exc:
        logger.warning("Could not pull model: %s", exc)
        return False


def generate(system_prompt: str, user_prompt: str, max_tokens: int = 300) -> str | None:
    """
    Call local Ollama model. Returns the response text or None on failure.
    """
    payload = {
        "model": LOCAL_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {
            "num_predict": max_tokens,
            "temperature": 0.1,
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
        return data["message"]["content"].strip()
    except requests.Timeout:
        logger.warning("Local model timed out after %ss.", LOCAL_TIMEOUT)
        return None
    except Exception as exc:
        logger.warning("Local model error: %s", exc)
        return None
