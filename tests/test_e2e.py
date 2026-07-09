"""
End-to-end dry-run test.
Mocks Fireworks and local model so no real API calls are made.
Verifies main.py produces valid output/results.json.
"""

import json
import os
import sys
import tempfile

# ── patch env before importing agent modules ──────────────────
os.environ.setdefault("FIREWORKS_API_KEY", "test-key")
os.environ.setdefault("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1")
os.environ.setdefault("ALLOWED_MODELS", "accounts/fireworks/models/llama-v3p1-8b-instruct")
os.environ["DISABLE_LOCAL_MODEL"] = "1"  # skip Ollama in tests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import patch
from agent.main import main


MOCK_ANSWER = "Mock answer for testing."


def test_end_to_end():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write input
        input_path = os.path.join(tmpdir, "tasks.json")
        output_path = os.path.join(tmpdir, "results.json")

        tasks = [
            {"task_id": "t1", "prompt": "What is 2 + 2?"},
            {"task_id": "t2", "prompt": "Classify the sentiment: I love this product!"},
            {"task_id": "t3", "prompt": "Write a Python function to add two numbers."},
        ]
        with open(input_path, "w") as f:
            json.dump(tasks, f)

        # Mock Fireworks call — also patch the module-level path reads in main
        with patch("agent.main.INPUT_PATH", input_path), \
             patch("agent.main.OUTPUT_PATH", output_path), \
             patch("agent.fireworks_client.call_model", return_value=MOCK_ANSWER):
            exit_code = main()

        assert exit_code == 0, f"Expected exit code 0, got {exit_code}"

        with open(output_path) as f:
            results = json.load(f)

        assert len(results) == 3, f"Expected 3 results, got {len(results)}"
        ids = {r["task_id"] for r in results}
        assert ids == {"t1", "t2", "t3"}, f"Wrong task IDs: {ids}"
        for r in results:
            assert "answer" in r, f"Missing 'answer' in result: {r}"

        print(f"End-to-end test PASSED. Results:")
        for r in results:
            print(f"  {r['task_id']}: {r['answer'][:60]}")


if __name__ == "__main__":
    test_end_to_end()
    print("\nAll tests passed!")
