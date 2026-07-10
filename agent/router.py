"""
Router: decides which model tier to use for each task category.

Routing strategy (accuracy-first, always Fireworks):
  ALL tasks → largest/most capable model (deepseek-v4-pro, gpt-oss-120b)
  These models give clean direct answers without thinking noise.
  Thinking models (kimi, glm) are only used as last-resort fallback.
"""

import logging

from agent import classifier
from agent import fireworks_client as fw
from agent.prompts import SYSTEM_PROMPTS, MAX_TOKENS

logger = logging.getLogger(__name__)


def route(category: str, prompt: str) -> str:
    """
    Route *prompt* (already classified as *category*) to the appropriate
    Fireworks model and return the answer string.
    Always uses the largest/most direct model for maximum accuracy.
    """
    system_prompt = SYSTEM_PROMPTS[category]
    max_tok = MAX_TOKENS[category]

    # Always start with largest (most capable, direct-answer) model
    answer = fw.call_cheapest(
        system_prompt=system_prompt,
        user_prompt=prompt,
        max_tokens=max_tok,
        tier=-1,  # largest / most direct
    )
    logger.info("FW model answered category '%s'.", category)
    return answer
