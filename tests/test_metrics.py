import numpy as np
import pytest

from zettlr_rag.metrics import (
    compute_centroid_dispersion,
    compute_semantic_entropy,
    compute_spherical_mean_resultant_length,
)


def test_spherical_mean_resultant_length():
    # Perfect alignment should be 1.0
    vecs = np.array([[1.0, 0.0], [1.0, 0.0]])
    assert compute_spherical_mean_resultant_length(vecs) == pytest.approx(1.0)

    # Opposing should be 0.0
    vecs = np.array([[1.0, 0.0], [-1.0, 0.0]])
    assert compute_spherical_mean_resultant_length(vecs) == pytest.approx(0.0)

def test_centroid_dispersion():
    # Identical vectors should have 0 dispersion
    vecs = np.array([[1.0, 1.0], [1.0, 1.0]])
    assert compute_centroid_dispersion(vecs) == 0.0

    # Simple calculation
    vecs = np.array([[0.0, 0.0], [2.0, 0.0]])
    # Mean = [1, 0]. Distances are [1, 1]. Mean dist = 1.0
    assert compute_centroid_dispersion(vecs) == pytest.approx(1.0)

def test_semantic_entropy():
    # Perfectly clustered should have 0 entropy
    vecs = np.array([[1.0, 0.0], [1.0, 0.1]]) # Should be same cluster
    assert compute_semantic_entropy(vecs) == 0.0

    # Distinct should have non-zero entropy
    vecs = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert compute_semantic_entropy(vecs) > 0.0
