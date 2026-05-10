# tests/test_local_storage.py
import json
import os
import sys
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from query import RAGQueryConfig, RAGQueryRunner
from zettlr_rag.metrics import QueryMetrics


def test_append_to_local_log(tmp_path):
    # Setup test file
    test_log = tmp_path / "test_query_log.jsonl"

    # Setup runner with test config
    config = RAGQueryConfig(
        log_path=str(test_log),
        graph_path=str(tmp_path / ".graph_index"),
        chroma_path=str(tmp_path / "chroma_db_academic"),
        index_persist_dir=str(tmp_path / ".index_metadata"),
    )
    # Mock engine and telemetry initialization to avoid real API/DB calls
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("query.RAGQueryRunner._initialize_engine", MagicMock())
        runner = RAGQueryRunner(config=config)
    # Create sample metrics
    metrics = QueryMetrics(
        question="What is shrinkage?",
        model_name="test-model",
        run_id="test-run",
        input_tokens=10,
        output_tokens=20,
        total_tokens=30,
        cost_total_usd=0.001,
        wall_time_ms=500.0,
        chunks_retrieved=5,
        top_similarity=0.9,
        mean_similarity=0.8,
        p10_similarity=0.7,
        p90_similarity=0.85,
    )
    answer = "Shrinkage is a statistical technique."

    # Execute method
    runner._append_to_local_log(metrics, answer)

    # Verify file content
    assert os.path.exists(test_log)
    with open(test_log) as f:
        lines = f.readlines()
        assert len(lines) == 1
        record = json.loads(lines[0])

        assert record["question"] == metrics.question
        assert record["answer"] == answer
        assert record["model_name"] == metrics.model_name
        assert record["run_id"] == metrics.run_id
        assert record["total_tokens"] == metrics.total_tokens
        assert record["cost_total_usd"] == metrics.cost_total_usd
        assert record["wall_time_ms"] == metrics.wall_time_ms
        assert record["p10_similarity"] == metrics.p10_similarity
        assert record["p90_similarity"] == metrics.p90_similarity
        assert "timestamp" in record


def test_append_to_local_log_multiple_entries(tmp_path):
    test_log = tmp_path / "test_multi_log.jsonl"
    config = RAGQueryConfig(
        log_path=str(test_log),
        graph_path=str(tmp_path / ".graph_index"),
        chroma_path=str(tmp_path / "chroma_db_academic"),
        index_persist_dir=str(tmp_path / ".index_metadata"),
    )

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("query.RAGQueryRunner._initialize_engine", MagicMock())
        runner = RAGQueryRunner(config=config)
    metrics = QueryMetrics(question="Q1")
    runner._append_to_local_log(metrics, "A1")
    runner._append_to_local_log(metrics, "A2")

    with open(test_log) as f:
        lines = f.readlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["answer"] == "A1"
        assert json.loads(lines[1])["answer"] == "A2"
