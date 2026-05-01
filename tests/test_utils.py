import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock, patch
from zettlr_rag.utils import load_query_log, load_langfuse_traces

def test_load_query_log_file_not_found(tmp_path, caplog):
    non_existent = tmp_path / "missing.jsonl"
    df = load_query_log(str(non_existent))
    assert isinstance(df, pd.DataFrame)
    assert df.empty
    assert "Query log not found" in caplog.text

def test_load_query_log_empty_file(tmp_path):
    empty_file = tmp_path / "empty.jsonl"
    empty_file.write_text("")
    # pd.read_json might raise ValueError or return empty depending on version/content
    # but our wrapper handles df.empty
    with patch("pandas.read_json") as mock_read:
        mock_read.return_value = pd.DataFrame()
        df = load_query_log(str(empty_file))
        assert df.empty

def test_load_query_log_valid(tmp_path):
    log_file = tmp_path / "valid.jsonl"
    log_file.write_text('{"timestamp": "2024-01-01T00:00:00Z", "query": "test"}\n')
    
    df = load_query_log(str(log_file))
    assert not df.empty
    assert len(df) == 1
    assert pd.api.types.is_datetime64tz_dtype(df["timestamp"])

@patch("langfuse.get_client")
@patch("dotenv.load_dotenv")
def test_load_langfuse_traces_connection_failure(mock_dotenv, mock_get_client):
    mock_get_client.side_effect = Exception("Connection Refused")
    df = load_langfuse_traces()
    assert df.empty

@patch("langfuse.get_client")
@patch("dotenv.load_dotenv")
def test_load_langfuse_traces_full_flow(mock_dotenv, mock_get_client):
    mock_lf = MagicMock()
    mock_get_client.return_value = mock_lf
    
    # Mock traces
    mock_trace = MagicMock()
    mock_trace.id = "trace_1"
    mock_trace.input = "question?"
    mock_trace.output = "answer"
    mock_trace.metadata = {"model": "gpt-4", "run_id": "run_1"}
    mock_trace.timestamp = "2024-01-01T00:00:00Z"
    mock_trace.latency = 100
    
    mock_lf.fetch_traces.return_value.data = [mock_trace]
    
    # Mock scores
    mock_score = MagicMock()
    mock_score.trace_id = "trace_1"
    mock_score.name = "accuracy"
    mock_score.value = 0.9
    
    mock_lf.fetch_scores.return_value.data = [mock_score]
    
    df = load_langfuse_traces()
    
    assert not df.empty
    assert "accuracy" in df.columns
    assert df.iloc[0]["accuracy"] == 0.9
    assert df.iloc[0]["trace_id"] == "trace_1"
    assert pd.api.types.is_datetime64tz_dtype(df["timestamp"])

@patch("langfuse.get_client")
@patch("dotenv.load_dotenv")
def test_load_langfuse_traces_no_data(mock_dotenv, mock_get_client):
    mock_lf = MagicMock()
    mock_get_client.return_value = mock_lf
    mock_lf.fetch_traces.return_value.data = []
    
    df = load_langfuse_traces()
    assert df.empty
