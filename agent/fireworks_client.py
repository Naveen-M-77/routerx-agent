"""
Fireworks AI client.

Uses the OpenAI-compatible SDK routed through FIREWORKS_BASE_URL.
All model IDs are read from the ALLOWED_MODELS env var at runtime.
"""

import logging
import os
import re

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
    gpt-oss-120b and deepseek-v4-pro are prioritised for direct answers.
    kimi/glm thinking models are deprioritised (tier adjusted via caller).
    """
    def size_score(name: str) -> int:
        n = name.lower()
        m = re.search(r"(\d+(?:\.\d+)?)\s*b", n)
        if m:
            return int(float(m.group(1)) * 10)
        # deepseek-v4-pro → treat as large/capable direct-answer model
        if "deepseek" in n and "pro" in n:
            return 250
        # gpt-oss-120b → very large
        if "120" in n or "gpt-oss" in n:
            return 300
        # kimi / glm thinking models — deprioritize (will be used as fallback)
        if re.search(r"kimi|glm", n):
            return 400
        if "pro" in n or "large" in n:
            return 200
        if "medium" in n or "mid" in n:
            return 100
        if "small" in n or "mini" in n or "tiny" in n:
            return 5
        return 150

    return sorted(models, key=size_score)


def _clean_thinking_output(content: str) -> str:
    """
    Strip thinking/reasoning from model output aggressively.

    Handles:
    1. <think>...</think> blocks (DeepSeek-R1 style)
    2. Numbered reasoning steps before a final answer block
    3. "The user wants me to..." preamble lines
    4. Self-correction tangents ("Wait, ...", "Let me check...")
    """
    # 1. Remove <think>...</think> tags
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

    lines = content.split("\n")

    # 2. Detect if this looks like thinking model output
    # Pattern: starts with "The user wants...", "I need to...", or numbered plan
    thinking_start_patterns = [
        re.compile(r"^\s*The user wants", re.IGNORECASE),
        re.compile(r"^\s*I need to", re.IGNORECASE),
        re.compile(r"^\s*Let me", re.IGNORECASE),
        re.compile(r"^\s*\d+\.\s+\*\*"),          # "1. **Step**"
        re.compile(r"^\s*Step\s+\d+", re.IGNORECASE),
    ]

    is_thinking_output = any(
        any(p.match(line) for p in thinking_start_patterns)
        for line in lines[:5]
    )

    if not is_thinking_output:
        return content.strip()

    # 3. For thinking model output, extract the final answer block.
    # Strategy: look for the LAST clean block of output after all the reasoning.
    # A clean block starts when we stop seeing "Wait,", "Let me", numbered steps, etc.

    noise_patterns = [
        re.compile(r"^\s*(The user|I need to|Let me|Wait,|Actually,|So,|Now,|Check|Or |Possible|Possible )", re.IGNORECASE),
        re.compile(r"^\s*\d+\.\s"),     # numbered list items
        re.compile(r"^\s*[-*]\s"),      # bullet points (reasoning)
        re.compile(r"^\s*\*\*"),        # bold headers in reasoning
        re.compile(r"^\s*Step\s+\d+", re.IGNORECASE),
    ]

    # Find all "clean" line segments (not noise)
    # We want the LAST contiguous block of clean, non-empty lines
    clean_blocks: list[list[str]] = []
    current_block: list[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if current_block:
                clean_blocks.append(current_block)
                current_block = []
            continue
        if any(p.match(stripped) for p in noise_patterns):
            if current_block:
                clean_blocks.append(current_block)
                current_block = []
        else:
            current_block.append(line)

    if current_block:
        clean_blocks.append(current_block)

    if clean_blocks:
        # Take the last clean block (the actual answer)
        last_block = "\n".join(clean_blocks[-1]).strip()
        # Extra safety: if it still starts with "I " or "The ", drop it
        if re.match(r"^(I |The |This |So |Therefore )", last_block, re.IGNORECASE):
            # Try second-to-last
            if len(clean_blocks) > 1:
                last_block = "\n".join(clean_blocks[-2]).strip()
        return last_block

    return content.strip()


def call_model(
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 512,
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
    return _clean_thinking_output(content)


def call_cheapest(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.1,
    tier: int = 0,
) -> str:
    """
    Call a Fireworks model at the given tier (0 = smallest/most direct, -1 = largest).
    Falls back to next tier on failure.
    """
    models = _rank_models(get_allowed_models())
    if not models:
        raise RuntimeError("ALLOWED_MODELS is empty or not set.")

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
