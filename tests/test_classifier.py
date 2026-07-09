"""
Smoke test: runs the classifier and verifies all 8 practice tasks
are classified correctly. No API calls needed.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agent.classifier import classify, SENTIMENT, NER, SUMMARIZATION, MATH, CODE_DEBUG, CODE_GEN, LOGICAL, FACTUAL

CASES = [
    ("What is the capital of Australia, and what body of water is it near?", FACTUAL),
    ("A store has 240 items. It sells 15% on Monday and 60 more on Tuesday. How many items remain?", MATH),
    ("Classify the sentiment of this review: The battery life is great, but the screen scratches too easily.", SENTIMENT),
    ("Summarize the following in exactly one sentence: Artificial intelligence is transforming industries.", SUMMARIZATION),
    ("Extract all named entities and their types from: Maria Sanchez joined Fireworks AI in Berlin last March.", NER),
    ("This function should return the max of a list but has a bug: def get_max(nums): return nums[0]. Find and fix it.", CODE_DEBUG),
    ("Three friends, Sam, Jo, and Lee, each own a different pet: cat, dog, bird. Sam does not own the bird. Jo owns the dog. Who owns the cat?", LOGICAL),
    ("Write a Python function that returns the second-largest number in a list, handling duplicates correctly.", CODE_GEN),
]


def test_classifier():
    passed = 0
    failed = 0
    for prompt, expected in CASES:
        actual = classify(prompt)
        if actual != expected:
            failed += 1
            print(f"FAIL  expected={expected:15s} got={actual:15s}  prompt={prompt[:60]}...")
        else:
            passed += 1
            print(f"PASS  category={actual:15s}  prompt={prompt[:60]}...")

    print(f"\n{passed}/{len(CASES)} passed.")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    test_classifier()
