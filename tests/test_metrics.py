# tests/test_metrics.py
import pytest
from zettlr_rag.metrics import (
    TokenUsage,
    calculate_cost,
    calculate_window_utilization,
)
from zettlr_rag.consts import GEMINI_CONTEXT_WINDOWS, GEMINI_PRICING

MOCK_PRICING = {
    "test-model": {"input": 0.000_001, "output": 0.000_002, "cache": 0.0000005}
}
MOCK_WINDOWS = {"test-model": 1_000_000}


class TestTokenUsage:
    def test_total_tokens_sums_all_three(self):
        u = TokenUsage(input_tokens=100, output_tokens=50, cache_tokens=25)
        assert u.total_tokens == 175

    def test_total_tokens_zero_when_empty(self):
        assert TokenUsage().total_tokens == 0


class TestCalculateCost:
    def test_cost_breakdown_correctness(self):
        usage = TokenUsage(input_tokens=1000, output_tokens=500, cache_tokens=200)
        ci, co, cc, ct = calculate_cost(usage, "test-model", MOCK_PRICING)
        assert ci == pytest.approx(0.001, rel=1e-5)
        assert co == pytest.approx(0.001, rel=1e-5)
        assert cc == pytest.approx(0.0001, rel=1e-5)
        assert ct == pytest.approx(ci + co + cc, rel=1e-10)

    def test_unknown_model_returns_zeros(self):
        usage = TokenUsage(input_tokens=1000, output_tokens=500, cache_tokens=0)
        assert calculate_cost(usage, "nonexistent-model", MOCK_PRICING) == (0.0, 0.0, 0.0, 0.0)

    def test_zero_tokens_returns_zero_cost(self):
        usage = TokenUsage()
        _, _, _, total = calculate_cost(usage, "test-model", MOCK_PRICING)
        assert total == 0.0

    def test_all_gemini_models_present_in_pricing(self):
        """Ensures pricing table is populated for all known models."""
        for model in GEMINI_CONTEXT_WINDOWS:
            assert model in GEMINI_PRICING, f"{model} missing from GEMINI_PRICING"


class TestWindowUtilization:
    def test_utilization_calculation(self):
        window_size, pct = calculate_window_utilization(500_000, "test-model", MOCK_WINDOWS)
        assert window_size == 1_000_000
        assert pct == pytest.approx(50.0, rel=1e-5)

    def test_unknown_model_returns_zero(self):
        window_size, pct = calculate_window_utilization(1000, "nonexistent", MOCK_WINDOWS)
        assert window_size == 0
        assert pct == 0.0

    def test_full_utilization(self):
        _, pct = calculate_window_utilization(1_000_000, "test-model", MOCK_WINDOWS)
        assert pct == pytest.approx(100.0)

    def test_gemini_models_have_window_defined(self):
        for model in GEMINI_PRICING:
            assert model in GEMINI_CONTEXT_WINDOWS, f"{model} missing from GEMINI_CONTEXT_WINDOWS"
