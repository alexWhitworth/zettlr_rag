import os
import sys
from unittest.mock import MagicMock, patch

# Add the project root to sys.path so 'evals' can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evals.llm_judge import llm_judge


def test_llm_judge():
    test_q = "What is the primary advantage of Stein's estimator?"
    test_rag = (
        "Stein's estimator provides lower mean squared error than the MLE in high dimensions."
    )
    test_ref = (
        "Stein's estimator (1956) dominates the MLE for p >= 3 by shrinking toward the origin."
    )
    test_cites = "Stein (1956), James & Stein (1961)"

    mock_response = MagicMock()
    mock_response.text = (
        '{"factual_precision": 0.8, "content_recall": 0.5, '
        '"fluency_and_structure": 1.0, "contradiction_detected": 0}'
    )

    with patch("evals.llm_judge.GoogleGenAI") as mock_genai:
        mock_instance = MagicMock()
        mock_instance.complete.return_value = mock_response
        mock_genai.return_value = mock_instance

        results = llm_judge(test_q, test_rag, test_ref, test_cites)

        assert results["factual_precision"] == 0.8
        assert results["content_recall"] == 0.5
        assert results["fluency_and_structure"] == 1.0
        assert results["contradiction_detected"] == 0

        mock_instance.complete.assert_called_once()

        # Verify that the correct strings were injected into the prompt
        called_prompt = mock_instance.complete.call_args[0][0]
        assert test_q in called_prompt
        assert test_rag in called_prompt
        assert test_ref in called_prompt
        assert test_cites in called_prompt


def test_llm_judge_json_parsing_with_markdown():
    test_q = "What is the primary advantage of Stein's estimator?"
    test_rag = (
        "Stein's estimator provides lower mean squared error than the MLE in high dimensions."
    )
    test_ref = (
        "Stein's estimator (1956) dominates the MLE for p >= 3 by shrinking toward the origin."
    )
    test_cites = "Stein (1956), James & Stein (1961)"

    mock_response = MagicMock()
    # Simulate a response that includes markdown code blocks
    mock_response.text = """Here is your evaluation:
```json
{
  "factual_precision": 0.8,
  "content_recall": 0.5,
  "fluency_and_structure": 1.0,
  "contradiction_detected": 0
}
```
"""

    with patch("evals.llm_judge.GoogleGenAI") as mock_genai:
        mock_instance = MagicMock()
        mock_instance.complete.return_value = mock_response
        mock_genai.return_value = mock_instance

        results = llm_judge(test_q, test_rag, test_ref, test_cites)

        assert results["factual_precision"] == 0.8
        assert results["content_recall"] == 0.5
        assert results["fluency_and_structure"] == 1.0
        assert results["contradiction_detected"] == 0
