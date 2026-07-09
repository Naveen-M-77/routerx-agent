"""
Minimal system prompts per category.
Kept terse to reduce input token count.
"""

from agent.classifier import (
    SENTIMENT, NER, SUMMARIZATION, MATH,
    CODE_DEBUG, CODE_GEN, LOGICAL, FACTUAL,
)

SYSTEM_PROMPTS = {
    FACTUAL: (
        "Answer the question directly in 1-3 sentences. "
        "Output ONLY the answer — no preamble, no reasoning, no restating the question."
    ),

    MATH: (
        "Solve the problem. Output the final numerical answer on the last line, clearly labeled. "
        "Keep working steps to 3 lines max. No padding or explanations beyond what is needed."
    ),

    SENTIMENT: (
        "Classify the sentiment. Output EXACTLY this format and nothing else:\n"
        "Sentiment: <Positive|Negative|Neutral|Mixed>. <One sentence reason>."
    ),

    SUMMARIZATION: (
        "Summarize as instructed. Output ONLY the summary — "
        "no preamble, no analysis steps, no meta-commentary."
    ),

    NER: (
        "Extract named entities. Output ONLY a valid JSON array, nothing else:\n"
        '[{"entity": "...", "type": "PERSON|ORG|LOCATION|DATE|OTHER"}]'
    ),

    CODE_DEBUG: (
        "Identify the bug and output the fix. Use this format exactly:\n"
        "Bug: <one line description>\n"
        "Fixed code:\n```<lang>\n<corrected code>\n```\n"
        "No other text."
    ),

    LOGICAL: (
        "Reason through the constraints and state the final answer. "
        "Be concise — 3-5 lines max. End with 'Answer: <final answer>'."
    ),

    CODE_GEN: (
        "Write the requested function. Output ONLY the code block with a brief docstring. "
        "No explanation outside the code block."
    ),
}

# Maximum tokens per category — kept tight to minimize output token cost
MAX_TOKENS = {
    FACTUAL: 120,
    MATH: 150,
    SENTIMENT: 60,
    SUMMARIZATION: 120,
    NER: 150,
    CODE_DEBUG: 300,
    LOGICAL: 150,
    CODE_GEN: 500,
}

