import argparse
import json
import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from evals.reliability_harness import ReliabilityHarness, _reliability_verdict, get_response_embeddings, main


@pytest.fixture
def mock_metrics():
    metrics = MagicMock()
    metrics.cost_total_usd = 0.005
    metrics.wall_time_ms = 250.0
    metrics.total_tokens = 100
    return metrics


@pytest.fixture
def test_log_path(tmp_path):
    return str(tmp_path / "test_validation_log.jsonl")


def test_reliability_verdict():
    # Consistent
    assert _reliability_verdict(4.0, 4.0, 4.0).startswith("✅")
    # Acceptable
    assert _reliability_verdict(10.0, 4.0, 4.0).startswith("⚠️")
    # Inconsistent
    assert _reliability_verdict(20.0, 4.0, 4.0).startswith("❌")


@patch("evals.reliability_harness.RAGQueryRunner")
@patch("evals.reliability_harness.get_response_embeddings")
@patch("evals.reliability_harness.compute_centroid_dispersion")
@patch("evals.reliability_harness.compute_spherical_mean_resultant_length")
@patch("evals.reliability_harness.compute_semantic_entropy")
def test_run_test(
    mock_semantic_entropy,
    mock_resultant_length,
    mock_dispersion,
    mock_get_embeddings,
    MockRunner,
    mock_metrics,
    test_log_path,
):
    # Setup mocks
    mock_runner_instance = MagicMock()
    # Provide slightly varied responses to avoid division by zero in cv
    metrics_run_1 = MagicMock(cost_total_usd=0.005, wall_time_ms=250.0, total_tokens=100)
    metrics_run_2 = MagicMock(cost_total_usd=0.006, wall_time_ms=300.0, total_tokens=110)
    
    mock_runner_instance.query.side_effect = [
        ("Response 1", metrics_run_1),
        ("Response 2", metrics_run_2),
    ]
    MockRunner.return_value = mock_runner_instance

    mock_get_embeddings.return_value = np.array([[0.1, 0.9], [0.2, 0.8]])
    mock_dispersion.return_value = 0.5
    mock_resultant_length.return_value = 0.9
    mock_semantic_entropy.return_value = 1.2

    # Create harness with semantic_entropy=True to hit that code path
    harness = ReliabilityHarness(semantic_entropy=True, log_path=test_log_path)
    
    # Run test with 2 runs
    summary = harness.run_test("test question", n_runs=2)

    # Assert outputs and structure
    assert summary["question"] == "test question"
    assert summary["n_runs"] == 2
    assert "cost" in summary
    assert "latency_ms" in summary
    assert "tokens" in summary
    assert summary["cost"]["mean"] == 0.0055
    assert "embedding_metrics" in summary
    assert summary["embedding_metrics"]["spherical_mean_resultant_length"] == 0.9
    assert summary["embedding_metrics"]["semantic_entropy"] == 1.2

    # Assert log writing
    assert os.path.exists(test_log_path)
    with open(test_log_path, "r") as f:
        lines = f.readlines()
        assert len(lines) == 1
        logged_summary = json.loads(lines[0])
        assert logged_summary["question"] == "test question"


def test_print_summary(capsys, test_log_path):
    harness = ReliabilityHarness(log_path=test_log_path)
    mock_summary = {
        "question": "test question",
        "n_runs": 2,
        "cost": {"mean": 0.005, "stdev": 0.001, "p10": 0.004, "p90": 0.006, "p95": 0.006, "cv_pct": 20.0},
        "latency_ms": {"mean": 250, "stdev": 10, "p10": 240, "p90": 260, "p95": 260, "cv_pct": 4.0},
        "tokens": {"mean": 100, "stdev": 5, "p10": 95, "p90": 105, "p95": 105, "cv_pct": 5.0},
        "embedding_metrics": {
            "spherical_mean_resultant_length": 0.9,
            "centroid_dispersion": 0.1,
            "semantic_entropy": 1.5,
        },
        "reliability_verdict": "⚠️  ACCEPTABLE",
    }
    
    harness.print(mock_summary)
    
    # Capture print output
    captured = capsys.readouterr()
    assert "RELIABILITY REPORT: test question" in captured.out
    assert "Cost (USD):" in captured.out
    assert "P10:   $0.004000" in captured.out
    assert "Semantic Entropy (H_sem): 1.5000" in captured.out
    assert "⚠️  ACCEPTABLE" in captured.out
    assert "CV" not in captured.out

@patch("llama_index.core.Settings")
@patch("zettlr_rag.rag_setup.setup_settings")
def test_get_response_embeddings(mock_setup, mock_settings):
    mock_model = MagicMock()
    mock_model.get_text_embedding.side_effect = [[0.1], [0.2]]
    mock_settings.embed_model = mock_model
    
    # We patch Settings globally as it's imported locally but used globally from llama_index.core
    with patch("llama_index.core.Settings", mock_settings):
        embeddings = get_response_embeddings(["ans1", "ans2"])
    
    mock_setup.assert_called_once()
    assert isinstance(embeddings, np.ndarray)
    assert np.array_equal(embeddings, np.array([[0.1], [0.2]]))

@patch("evals.reliability_harness.ReliabilityHarness")
@patch("evals.reliability_harness.init_telemetry")
@patch("argparse.ArgumentParser.parse_args")
def test_main(mock_parse_args, mock_init_telemetry, MockHarness):
    mock_args = MagicMock()
    mock_args.question = "Test Question?"
    mock_args.questions_file = None
    mock_args.runs = 2
    mock_args.semantic_entropy = False
    mock_parse_args.return_value = mock_args
    mock_init_telemetry.return_value = False
    
    mock_harness_instance = MagicMock()
    mock_harness_instance.run_test.return_value = {"mock": "summary"}
    MockHarness.return_value = mock_harness_instance
    
    main()
    
    MockHarness.assert_called_once_with(instrumented=False, semantic_entropy=False)
    mock_harness_instance.run_test.assert_called_once_with("Test Question?", n_runs=2)
    mock_harness_instance.print.assert_called_once_with({"mock": "summary"})
