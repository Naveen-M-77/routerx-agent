"""
Task classifier — zero API calls.
Returns one of 8 category strings for a given prompt.
"""

import re

# Category constants
SENTIMENT = "sentiment"
NER = "ner"
SUMMARIZATION = "summarization"
MATH = "math"
CODE_DEBUG = "code_debug"
CODE_GEN = "code_gen"
LOGICAL = "logical"
FACTUAL = "factual"


# Ordered rules: first match wins
_RULES = [
    # Code debugging — must come before code_gen to catch "has a bug / find and fix"
    (CODE_DEBUG, [
        r"\bbug\b",
        r"\bfind\s+and\s+fix\b",
        r"\bdebugg?\b",
        r"\bthis\s+function\s+should\b",
        r"\bincorrect(ly)?\b.*\bcode\b",
        r"\bwhat.*(wrong|issue|problem).*with.*code\b",
        r"\bfix\s+(the\s+)?(code|function|bug|error)\b",
        r"\berror\b.*\bcode\b",
    ]),

    # Code generation
    (CODE_GEN, [
        r"\bwrite\s+a\s+(python\s+)?function\b",
        r"\bimplement\b.*\bfunction\b",
        r"\bgenerate\s+(a\s+)?code\b",
        r"\bwrite\s+(the\s+)?code\b",
        r"\bcreate\s+a\s+function\b",
        r"\bcode\s+(that|to|which)\b",
        r"\bwrite\s+a\s+program\b",
        r"\bfunction\s+that\s+(returns|takes|accepts|computes|calculates)\b",
    ]),

    # Named entity recognition
    (NER, [
        r"\bnamed\s+entit",
        r"\bextract\s+(all\s+)?entities\b",
        r"\bidentify\s+(all\s+)?(entities|people|organizations|locations)\b",
        r"\bentity\s+(recognition|extraction|labell?ing)\b",
        r"\b(person|org(anization)?|location|date)\b.*\bextract\b",
        r"\blabel\s+(the\s+)?entities\b",
    ]),

    # Sentiment classification
    (SENTIMENT, [
        r"\bsentiment\b",
        r"\bclassify\s+(the\s+)?(sentiment|review|opinion|text)\b",
        r"\b(positive|negative|neutral)\s+(or|/)\s+(negative|positive|neutral)\b",
        r"\bhow\s+(does|do).*(feel|sound|come across)\b",
        r"\bemotion\s+(of|in|from)\b",
        r"\bopinion\s+analysis\b",
        r"\bis\s+(this|the)\s+(review|text|comment)\s+(positive|negative|neutral)\b",
    ]),

    # Summarization
    (SUMMARIZATION, [
        r"\bsummariz(e|ation)\b",
        r"\bsummaris(e|ation)\b",
        r"\bcondense\b",
        r"\bin\s+(one|1|two|2|three|3)\s+sentence",
        r"\bbrief\s+summary\b",
        r"\bshorten\b.*\btext\b",
        r"\btldr\b",
        r"\bgist\s+of\b",
        r"\bkey\s+(points|takeaways)\s+from\b",
    ]),

    # Logical / deductive reasoning
    (LOGICAL, [
        r"\bwho\s+owns\b",
        r"\bwho\s+(has|lives|works|is|sits)\b.*\b(if|given|since)\b",
        r"\bconstraint\b",
        r"\bpuzzle\b",
        r"\bdeductive\b",
        r"\bwhich\s+(person|one|friend|student)\b",
        r"\b(sam|jo|lee|alice|bob|carol|dave)\b.*\b(owns?|has|is)\b",
        r"\beach\s+(own|have|is)\s+a\s+different\b",
        r"\ball\s+(of\s+)?(the\s+)?following\s+(are\s+)?true\b",
        r"\bif\s+.+then\s+.+(who|what|which)\b",
    ]),

    # Mathematical reasoning
    (MATH, [
        r"\b(calculat|comput|solv)\w+\b",
        r"\b\d+\s*%",
        r"\bhow\s+many\b",
        r"\bhow\s+much\b",
        r"\bwhat\s+is\s+[\d\s\+\-\*\/\^]+",
        r"\b(add|subtract|multiply|divide|sum|product|quotient|remainder)\b",
        r"\b(profit|loss|interest|tax|discount|average|mean|median|total|price)\b",
        r"\b\d+\s+(items?|people|students?|dollars?|units?)\b",
        r"\bword\s+problem\b",
        r"\bif\s+\w+\s+(has|have|owns?|costs?|sells?)\s+\d+\b",
        r"\b(increase|decrease)\s+by\s+\d+",
        r"\bpercentage\b",
        r"\bprojection\b",
    ]),
]


def classify(prompt: str) -> str:
    """
    Classify *prompt* into one of 8 category strings.
    Falls back to FACTUAL if no rule matches.
    """
    text = prompt.lower()

    for category, patterns in _RULES:
        for pat in patterns:
            if re.search(pat, text):
                return category

    return FACTUAL
