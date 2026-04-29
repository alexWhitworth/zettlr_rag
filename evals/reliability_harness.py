# evals/reliability_harness.py
"""
Reliability harness: run the same query N times and measure consistency
of answers, cost, and latency across runs.

Usage:
    python evals/reliability_harness.py "What is the effect of sleep on memory?" --runs 5
    python evals/reliability_harness.py --questions-file evals/test_questions.txt --runs 3
"""

import argparse
import os
import statistics
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from query import RAGQueryConfig, RAGQueryRunner
from zettlr_rag.metrics import (
    compute_centroid_dispersion,
    compute_semantic_entropy,
    compute_spherical_mean_resultant_length,
)
from zettlr_rag.telemetry import init_telemetry


def get_response_embeddings(answers: list[str]) -> np.ndarray:
    """Helper to get embeddings for a list of strings using the configured embed_model."""
    from llama_index.core import Settings

    from zettlr_rag.rag_setup import setup_settings

    setup_settings()

    return np.array([Settings.embed_model.get_text_embedding(a) for a in answers])


def run_reliability_test(
    question: str,
    n_runs: int = 5,
    instrumented: bool = False,
    semantic_entropy: bool = False,
) -> dict:
    """
    Run a single question N times and compute consistency statistics.
    """
    results = []

    print(f"\n🔁 Running '{question}' x {n_runs}...", file=sys.stderr)

    for i in range(1, n_runs + 1):
        run_id = f"reliability_run_{i}_of_{n_runs}"
        config = RAGQueryConfig(instrumented=instrumented, run_id=run_id)
        runner = RAGQueryRunner(config=config)
        response, metrics = runner.query(question)

        results.append({
            "run":     i,
            "answer":  str(response),
            "metrics": metrics,
        })
        msg = f"  Run {i}/{n_runs}: ${metrics.cost_total_usd:.6f} | {metrics.wall_time_ms:.0f}ms"
        print(msg, file=sys.stderr)

    costs     = [r["metrics"].cost_total_usd  for r in results]
    latencies = [r["metrics"].wall_time_ms     for r in results]
    tokens    = [r["metrics"].total_tokens     for r in results]
    answers   = [r["answer"]                   for r in results]

    def cv(values):
        """Coefficient of variation."""
        m = statistics.mean(values)
        return (statistics.stdev(values) / m * 100) if m > 0 and len(values) > 1 else 0.0

    summary = {
        "question": question,
        "n_runs":   n_runs,
        "cost": {
            "mean":   round(statistics.mean(costs), 8),
            "stdev":  round(statistics.stdev(costs), 8) if n_runs > 1 else 0.0,
            "min":    round(min(costs), 8),
            "max":    round(max(costs), 8),
            "cv_pct": round(cv(costs), 2),
        },
        "latency_ms": {
            "mean":   round(statistics.mean(latencies), 1),
            "stdev":  round(statistics.stdev(latencies), 1) if n_runs > 1 else 0.0,
            "min":    round(min(latencies), 1),
            "max":    round(max(latencies), 1),
            "p95":    round(sorted(latencies)[int(n_runs * 0.95)] if n_runs > 0 else 0.0, 1),
            "cv_pct": round(cv(latencies), 2),
        },
        "tokens": {
            "mean":   round(statistics.mean(tokens), 1),
            "stdev":  round(statistics.stdev(tokens), 1) if n_runs > 1 else 0.0,
            "cv_pct": round(cv(tokens), 2),
        },
    }

    embeddings = get_response_embeddings(answers)
    em = {
        "spherical_mean_resultant_length": round(
            compute_spherical_mean_resultant_length(embeddings), 4
        ),
        "centroid_dispersion": round(compute_centroid_dispersion(embeddings), 4),
    }
    if semantic_entropy:
        em["semantic_entropy"] = round(compute_semantic_entropy(embeddings), 4)
    summary["embedding_metrics"] = em

    summary["reliability_verdict"] = _reliability_verdict(cv(costs), cv(latencies), cv(tokens))
    return summary


def _reliability_verdict(cost_cv: float, latency_cv: float, token_cv: float) -> str:
    """Simple heuristic verdict."""
    max_cv = max(cost_cv, latency_cv, token_cv)
    if max_cv < 5.0:
        return "✅ CONSISTENT — CV < 5% across cost, latency, tokens"
    if max_cv < 15.0:
        return f"⚠️  ACCEPTABLE — max CV {max_cv:.1f}% (review if > 15%)"
    return f"❌ INCONSISTENT — max CV {max_cv:.1f}% — investigate temperature/top_k/chunking"


def print_summary(summary: dict) -> None:
    print(f"\n{'='*60}")
    print(f"RELIABILITY REPORT: {summary['question'][:60]}")
    print(f"{'='*60}")
    print(f"Runs: {summary['n_runs']}")
    print("\nCost (USD):")
    print(f"  Mean:  ${summary['cost']['mean']:.6f}")
    print(f"  Stdev: ${summary['cost']['stdev']:.6f}  (CV: {summary['cost']['cv_pct']:.1f}%)")
    print(f"  Range: ${summary['cost']['min']:.6f} - {summary['cost']['max']:.6f}")
    print("\nLatency (ms):")
    print(f"  Mean:  {summary['latency_ms']['mean']:.0f} ms")
    stdev_lat = summary['latency_ms']['stdev']
    cv_lat = summary['latency_ms']['cv_pct']
    print(f"  Stdev: {stdev_lat:.0f} ms  (CV: {cv_lat:.1f}%)")
    print(f"  P95:   {summary['latency_ms']['p95']:.0f} ms")
    print("\nTokens:")
    print(f"  Mean:  {summary['tokens']['mean']:.0f}")
    print(f"  CV:    {summary['tokens']['cv_pct']:.1f}%")

    em = summary["embedding_metrics"]
    print("\nEmbedding Consistency:")
    line = f"  Spherical Mean Resultant Length (R): {em['spherical_mean_resultant_length']:.4f}"
    print(f"{line} (1.0 = perfect)")
    print(f"  Centroid Dispersion (CD): {em['centroid_dispersion']:.4f}")
    if "semantic_entropy" in em:
        print(f"  Semantic Entropy (H_sem): {em['semantic_entropy']:.4f}")

    print(f"\n{summary['reliability_verdict']}")
    print(f"{'='*60}\n")



def main():
    import nest_asyncio
    nest_asyncio.apply()

    parser = argparse.ArgumentParser(description="RAG Reliability Harness")
    parser.add_argument("question", type=str, nargs="?", help="Single question to test.")
    parser.add_argument("--questions-file", type=str, help="Path to .txt file.")
    parser.add_argument("--runs", type=int, default=5, help="Number of runs.")
    parser.add_argument("--semantic-entropy", action="store_true",
                        help="Enable semantic entropy.")
    args = parser.parse_args()

    instrumented = init_telemetry()

    questions = []
    if args.questions_file:
        with open(args.questions_file) as f:
            questions = [line.strip() for line in f if line.strip()]
    elif args.question:
        questions = [args.question]
    else:
        parser.error("Provide a question or --questions-file")

    for question in questions:
        summary = run_reliability_test(
            question,
            n_runs=args.runs,
            instrumented=instrumented,
            semantic_entropy=args.semantic_entropy
        )
        print_summary(summary)


if __name__ == "__main__":
    main()
