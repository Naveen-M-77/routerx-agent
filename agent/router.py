"""
Router: decides which model tier to use for each task category.

Routing strategy (cheapest first, fallback to more capable):
  - EASY categories → local Ollama model (zero Fireworks tokens)
  - MEDIUM categories → smallest Fireworks model
  - HARD categories → second smallest (or largest as fallback)

If local model is unavailable or returns None, escalates to Fireworks.
"""

import logging
import os

from agent import classifier
from agent import fireworks_client as fw
from agent import local_model as lm
from agent.prompts import SYSTEM_PROMPTS, MAX_TOKENS

logger = logging.getLogger(__name__)

# Categories handled locally by default
_LOCAL_CATEGORIES = {
    classifier.SENTIMENT,
    classifier.NER,
    classifier.SUMMARIZATION,
}

# Categories that go to smallest Fireworks model
_SMALL_FW_CATEGORIES = {
    classifier.MATH,
    classifier.LOGICAL,
    classifier.FACTUAL,
    classifier.CODE_DEBUG,
}

# Categories that go to larger Fireworks model
_LARGE_FW_CATEGORIES = {
    classifier.CODE_GEN,
}

# Whether local model is available (set once at startup)
_local_available: bool | None = None


def _is_local_enabled() -> bool:
    """Check env override and whether Ollama is up."""
    global _local_available
    if os.environ.get("DISABLE_LOCAL_MODEL", "").lower() in ("1", "true", "yes"):
        return False
    if _local_available is None:
        _local_available = lm.ensure_model_available()
    return _local_available


def route(category: str, prompt: str) -> str:
    """
    Route *prompt* (already classified as *category*) to the appropriate
    model and return the answer string.
    """
    system_prompt = SYSTEM_PROMPTS[category]
    max_tok = MAX_TOKENS[category]

    # --- Try local model for easy categories ---
    if category in _LOCAL_CATEGORIES and _is_local_enabled():
        answer = lm.generate(system_prompt, prompt, max_tokens=max_tok)
        if answer:
            logger.info("Local model answered category '%s'.", category)
            return answer
        logger.info("Local model failed for '%s' — escalating to Fireworks.", category)

    # --- Small Fireworks model ---
    if category in _SMALL_FW_CATEGORIES or category in _LOCAL_CATEGORIES:
        # LOCAL_CATEGORIES that fell through also get smallest FW model
        try:
            answer = fw.call_cheapest(
                system_prompt=system_prompt,
                user_prompt=prompt,
                max_tokens=max_tok,
                tier=0,  # smallest
            )
            logger.info("Small FW model answered category '%s'.", category)
            return answer
        except Exception as exc:
            logger.warning("Small FW model failed for '%s': %s — escalating.", category, exc)

    # --- Large Fireworks model (fallback / HARD categories) ---
    answer = fw.call_cheapest(
        system_prompt=system_prompt,
        user_prompt=prompt,
        max_tokens=max_tok,
        tier=-1,  # largest / most capable
    )
    logger.info("Large FW model answered category '%s'.", category)
    return answer
