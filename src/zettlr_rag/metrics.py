"""
Pure calculation functions for cost and utilization metrics.
No external dependencies — fully unit testable.
"""

from dataclasses import dataclass

import numpy as np
from scipy.stats import entropy
from sklearn.cluster import AgglomerativeClustering


@dataclass
class TokenUsage:
    """Token counts extracted from a Gemini API response."""
    input_tokens:  int = 0
    output_tokens: int = 0
    cache_tokens:  int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens + self.cache_tokens


@dataclass
class QueryMetrics:
    """All computed metrics for a single RAG query execution."""
    # ── Identity ──────────────────────────────────────────────────────────────
    question:           str = ""
    model_name:         str = ""
    run_id:             str | None = None

    # ── Token counts ──────────────────────────────────────────────────────────
    input_tokens:       int = 0
    output_tokens:      int = 0
    cache_tokens:       int = 0
    total_tokens:       int = 0

    # ── Cost ──────────────────────────────────────────────────────────────────
    cost_input_usd:     float = 0.0
    cost_output_usd:    float = 0.0
    cost_cache_usd:     float = 0.0
    cost_total_usd:     float = 0.0

    # ── Context window ────────────────────────────────────────────────────────
    context_window_size:    int   = 0
    window_utilization_pct: float = 0.0    # input_tokens / window_size * 100

    # ── Latency (all in milliseconds) ─────────────────────────────────────────
    wall_time_ms:       float = 0.0        # E2E from query() call to return
    llm_latency_ms:     float = 0.0        # Gemini API round-trip
    embedding_latency_ms: float = 0.0
    retrieval_latency_ms: float = 0.0
    ttft_ms:            float | None = None   # None if not streaming

    # ── Retrieval quality ─────────────────────────────────────────────────────
    chunks_retrieved:   int   = 0
    top_similarity:     float = 0.0
    mean_similarity:    float = 0.0

    # ── Scores dict for Langfuse bulk posting ─────────────────────────────────
    def to_langfuse_scores(self) -> dict[str, float]:
        """
        Returns a flat dict of numeric scores suitable for
        posting to Langfuse's score() API in a loop.
        """
        scores = {
            "cost_total_usd":          round(self.cost_total_usd, 8),
            "cost_input_usd":          round(self.cost_input_usd, 8),
            "cost_output_usd":         round(self.cost_output_usd, 8),
            "cost_cache_usd":          round(self.cost_cache_usd, 8),
            "tokens_input":            float(self.input_tokens),
            "tokens_output":           float(self.output_tokens),
            "tokens_cache":            float(self.cache_tokens),
            "tokens_total":            float(self.total_tokens),
            "window_utilization_pct":  round(self.window_utilization_pct, 4),
            "wall_time_ms":            round(self.wall_time_ms, 2),
            "llm_latency_ms":          round(self.llm_latency_ms, 2),
            "chunks_retrieved":        float(self.chunks_retrieved),
            "top_similarity":          round(self.top_similarity, 4),
            "mean_similarity":         round(self.mean_similarity, 4),
        }
        if self.ttft_ms is not None:
            scores["ttft_ms"] = round(self.ttft_ms, 2)
        return scores


def calculate_cost(
    usage: TokenUsage,
    model_name: str,
    pricing_table: dict,
) -> tuple[float, float, float, float]:
    """
    Calculate cost breakdown from token usage.

    Returns:
        (cost_input, cost_output, cost_cache, cost_total) all in USD
    """
    if model_name not in pricing_table:
        return 0.0, 0.0, 0.0, 0.0

    p = pricing_table[model_name]
    cost_input  = usage.input_tokens  * p.get("input",  0.0)
    cost_output = usage.output_tokens * p.get("output", 0.0)
    cost_cache  = usage.cache_tokens  * p.get("cache",  0.0)
    cost_total  = cost_input + cost_output + cost_cache

    return cost_input, cost_output, cost_cache, cost_total


def calculate_window_utilization(
    input_tokens: int,
    model_name: str,
    window_table: dict,
) -> tuple[int, float]:
    """
    Calculate context window utilization percentage.

    Returns:
        (window_size, utilization_pct)
        utilization_pct is 0.0 if model not in window_table
    """
    window_size = window_table.get(model_name, 0)
    if window_size == 0:
        return 0, 0.0
    utilization_pct = (input_tokens / window_size) * 100
    return window_size, round(utilization_pct, 4)


def extract_token_usage_from_response(response) -> TokenUsage:
    """
    Extract token counts from a LlamaIndex query response backed by Gemini.
    """
    usage = TokenUsage()

    if not hasattr(response, "metadata") or not response.metadata:
        return usage

    meta = response.metadata

    # ── Path 1: Gemini usage_metadata (most common in recent LlamaIndex) ──────
    usage.input_tokens  = meta.get("prompt_token_count", usage.input_tokens)
    usage.output_tokens = meta.get("candidates_token_count", usage.output_tokens)
    usage.cache_tokens  = meta.get("cached_content_token_count", usage.cache_tokens)
    if usage.total_tokens > 0:
        return usage

    # ── Path 3: LlamaIndex token_usage key ────────────────────────────────────
    token_usage = meta.get("token_usage", {})
    if token_usage:
        usage.input_tokens  = token_usage.get("prompt_tokens",     usage.input_tokens)
        usage.output_tokens = token_usage.get("completion_tokens", usage.output_tokens)
        if usage.total_tokens > 0:
            return usage

    # ── Path 4: Search in individual nodes? ──────────────────────────────────
    if hasattr(response, "source_nodes") and response.source_nodes:
        for node in response.source_nodes:
            if hasattr(node, "metadata") and node.metadata:
                node_usage = node.metadata.get("usage_metadata", {})
                if node_usage:
                    usage.input_tokens = node_usage.get(
                        "prompt_token_count", usage.input_tokens
                    )
                    usage.output_tokens = node_usage.get(
                        "candidates_token_count", usage.output_tokens
                    )
                    usage.cache_tokens = node_usage.get(
                        "cached_content_token_count", usage.cache_tokens
                    )
                    if usage.total_tokens > 0:
                        return usage

    return usage


def compute_spherical_mean_resultant_length(embeddings: np.ndarray) -> float:
    """Compute Spherical Mean Resultant Length (R) of embeddings."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    # Avoid division by zero
    norms[norms == 0] = 1.0
    unit_vectors = embeddings / norms
    return float(np.linalg.norm(np.mean(unit_vectors, axis=0)))


def compute_centroid_dispersion(embeddings: np.ndarray) -> float:
    """Compute Centroid Dispersion (CD) of embeddings."""
    if len(embeddings) <= 1:
        return 0.0
    centroid = np.mean(embeddings, axis=0)
    distances = np.linalg.norm(embeddings - centroid, axis=1)
    return float(np.mean(distances))


def compute_semantic_entropy(embeddings: np.ndarray) -> float:
    """Compute Semantic Entropy (H_sem) using agglomerative clustering."""
    if len(embeddings) <= 1:
        return 0.0

    # Distance threshold 0.1 corresponds to cosine similarity >= 0.9
    # (1 - similarity) = distance
    model = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=0.1
    )
    labels = model.fit_predict(embeddings)

    _, counts = np.unique(labels, return_counts=True)
    probs = counts / len(embeddings)
    return float(entropy(probs, base=2))
