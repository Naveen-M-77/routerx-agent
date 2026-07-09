"""
Fireworks AI client.

Uses the OpenAI-compatible SDK routed through FIREWORKS_BASE_URL.
All model IDs are read from the ALLOWED_MODELS env var at runtime.
"""

import logging
import os

from openai import OpenAI

logger = logging.getLogger(__name__)

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.environ["FIREWORKS_API_KEY"]
        base_url = os.environ["FIREWORKS_BASE_URL"]
        _client = OpenAI(api_key=api_key, base_url=base_url)
    return _client


def get_allowed_models() -> list[str]:
    """Parse and return ALLOWED_MODELS env var as a list."""
    raw = os.environ.get("ALLOWED_MODELS", "")
    return [m.strip() for m in raw.split(",") if m.strip()]


def _rank_models(models: list[str]) -> list[str]:
    """
    Sort models smallest → largest using heuristics on the model name.
    Smaller models are cheaper; we try smallest first.
    """
    def size_score(name: str) -> int:
        n = name.lower()
        import re
        # Numeric size hints: e.g. "8b", "70b", "120b"
        m = re.search(r"(\d+(?:\.\d+)?)\s*b", n)
        if m:
            return int(float(m.group(1)) * 10)
        # Version/tier hints in model slugs (p1 < p2 < pro, etc.)
        if re.search(r"p1\b", n):
            return 10
        if re.search(r"p5\b|p6\b", n):
            return 15
        if re.search(r"p2\b", n):
            return 20
        if "small" in n or "mini" in n or "tiny" in n:
            return 5
        if "medium" in n or "mid" in n:
            return 100
        if "pro" in n or "large" in n or "big" in n:
            return 200
        if "xl" in n or "xxl" in n or "120" in n:
            return 300
        return 50  # unknown → treat as medium

    return sorted(models, key=size_score)


def call_model(
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 300,
    temperature: float = 0.1,
) -> str:
    """
    Call a specific Fireworks model. Raises on failure.
    """
    client = _get_client()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    content = response.choices[0].message.content
    if content is None:
        raise ValueError(f"Model '{model}' returned null content (finish_reason={response.choices[0].finish_reason})")
    return content.strip()


def call_cheapest(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 300,
    temperature: float = 0.1,
    tier: int = 0,
) -> str:
    """
    Call a Fireworks model at the given tier (0 = smallest, -1 = largest).
    Falls back up the tier list on failure.

    tier=0  → smallest model
    tier=1  → second smallest
    tier=-1 → largest (guaranteed best quality)
    """
    models = _rank_models(get_allowed_models())
    if not models:
        raise RuntimeError("ALLOWED_MODELS is empty or not set.")

    # Build attempt order starting from tier
    if tier == -1:
        attempt_order = list(reversed(models))
    else:
        start = min(tier, len(models) - 1)
        attempt_order = models[start:] + models[:start]

    last_exc: Exception | None = None
    for model in attempt_order:
        try:
            logger.info("Calling Fireworks model: %s (tier %d)", model, tier)
            return call_model(model, system_prompt, user_prompt, max_tokens, temperature)
        except Exception as exc:
            logger.warning("Model %s failed: %s — trying next.", model, exc)
            last_exc = exc

    raise RuntimeError(f"All Fireworks models failed. Last error: {last_exc}")
