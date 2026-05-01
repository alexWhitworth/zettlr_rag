import numpy as np
import pytest
from unittest.mock import MagicMock

from zettlr_rag.metrics import (
    compute_centroid_dispersion,
    compute_semantic_entropy,
    compute_spherical_mean_resultant_length,
    calculate_cost,
    calculate_window_utilization,
    extract_token_usage_from_response,
    TokenUsage,
    QueryMetrics,
)

def test_spherical_mean_resultant_length():
    # Perfect alignment should be 1.0
    vecs = np.array([[1.0, 0.0], [1.0, 0.0]])
    assert compute_spherical_mean_resultant_length(vecs) == pytest.approx(1.0)

    # Opposing should be 0.0
    vecs = np.array([[1.0, 0.0], [-1.0, 0.0]])
    assert compute_spherical_mean_resultant_length(vecs) == pytest.approx(0.0)

    # Zero norm edge case
    vecs = np.array([[0.0, 0.0], [0.0, 0.0]])
    # Our implementation: norms[norms == 0] = 1.0, unit_vectors = [0,0] / 1.0 = [0,0]. Mean = [0,0]. Norm = 0.
    assert compute_spherical_mean_resultant_length(vecs) == 0.0

def test_centroid_dispersion():
    # Identical vectors should have 0 dispersion
    vecs = np.array([[1.0, 1.0], [1.0, 1.0]])
    assert compute_centroid_dispersion(vecs) == 0.0

    # Simple calculation
    vecs = np.array([[0.0, 0.0], [2.0, 0.0]])
    # Mean = [1, 0]. Distances are [1, 1]. Mean dist = 1.0
    assert compute_centroid_dispersion(vecs) == pytest.approx(1.0)

    # Single embedding edge case
    vecs = np.array([[1.0, 1.0]])
    assert compute_centroid_dispersion(vecs) == 0.0

def test_semantic_entropy():
    # Perfectly clustered should have 0 entropy
    vecs = np.array([[1.0, 0.0], [1.0, 0.1]]) # Should be same cluster
    assert compute_semantic_entropy(vecs) == 0.0

    # Distinct should have non-zero entropy
    vecs = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert compute_semantic_entropy(vecs) > 0.0

    # Single embedding edge case
    vecs = np.array([[1.0, 1.0]])
    assert compute_semantic_entropy(vecs) == 0.0

def test_calculate_cost():
    usage = TokenUsage(input_tokens=1000, output_tokens=500, cache_tokens=100)
    
    # Mock pricing table
    pricing = {
        "model-x": {"input": 0.01, "output": 0.02, "cache": 0.005}
    }
    
    # Known model
    ci, co, cc, ct = calculate_cost(usage, "model-x", pricing)
    assert ci == 10.0
    assert co == 10.0
    assert cc == 0.5
    assert ct == 20.5
    
    # Unknown model
    ci, co, cc, ct = calculate_cost(usage, "unknown", pricing)
    assert ct == 0.0

def test_calculate_window_utilization():
    windows = {"model-a": 10000}
    
    # Known
    size, pct = calculate_window_utilization(5000, "model-a", windows)
    assert size == 10000
    assert pct == 50.0
    
    # Unknown
    size, pct = calculate_window_utilization(5000, "unknown", windows)
    assert size == 0
    assert pct == 0.0

def test_extract_token_usage_path_1():
    # Path 1: prompt_token_count etc.
    response = MagicMock()
    response.metadata = {
        "prompt_token_count": 10,
        "candidates_token_count": 20,
        "cached_content_token_count": 5
    }
    usage = extract_token_usage_from_response(response)
    assert usage.input_tokens == 10
    assert usage.output_tokens == 20
    assert usage.cache_tokens == 5

def test_extract_token_usage_path_3():
    # Path 3: token_usage dict
    response = MagicMock()
    response.metadata = {
        "token_usage": {
            "prompt_tokens": 15,
            "completion_tokens": 25
        }
    }
    usage = extract_token_usage_from_response(response)
    assert usage.input_tokens == 15
    assert usage.output_tokens == 25

def test_extract_token_usage_path_4():
    # Path 4: source_nodes usage_metadata
    response = MagicMock()
    # Must have non-empty metadata to pass the initial guard clause
    response.metadata = {"some_other_key": "val"}
    node = MagicMock()
    node.metadata = {"usage_metadata": {"prompt_token_count": 100, "candidates_token_count": 200}}
    response.source_nodes = [node]
    
    usage = extract_token_usage_from_response(response)
    assert usage.input_tokens == 100
    assert usage.output_tokens == 200

def test_extract_token_usage_no_metadata():
    response = MagicMock()
    del response.metadata
    usage = extract_token_usage_from_response(response)
    assert usage.total_tokens == 0

def test_query_metrics_to_langfuse_scores():
    metrics = QueryMetrics(
        cost_total_usd=0.001234567,
        input_tokens=100,
        output_tokens=50,
        top_similarity=0.95,
        mean_similarity=0.8,
        p10_similarity=0.6,
        p90_similarity=0.9
    )
    
    scores = metrics.to_langfuse_scores()
    
    assert scores["cost_total_usd"] == 0.00123457
    assert scores["tokens_input"] == 100.0
    assert scores["top_similarity"] == 0.95
    assert scores["p10_similarity"] == 0.6
    assert scores["p90_similarity"] == 0.9
