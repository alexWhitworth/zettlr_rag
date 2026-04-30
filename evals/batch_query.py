import argparse
import nest_asyncio
from query import RAGQueryRunner, RAGQueryConfig

# argparse
parser = argparse.ArgumentParser(description="Run a batch of RAG queries from a file.")
parser.add_argument("--file", type=str, default="my_queries.txt", help="Path to the file containing queries (one per line).")
args = parser.parse_args()

# Required for async operations within the loop
nest_asyncio.apply()

def run_batch(questions):
    # Initialize once
    config = RAGQueryConfig(instrumented=True, run_id="batch_75_run")
    runner = RAGQueryRunner(config=config)

    for q in questions:
        print(f"Processing: {q}")
        # RAGQueryRunner.query automatically handles local logging to query_log.jsonl
        response, metrics = runner.query(q)
        print(f"Done. Cost: ${metrics.cost_total_usd:.4f}")


if __name__ == "__main__":
    with open(args.file) as f:
        queries = [
            line.strip() for line in f 
            if line.strip() and not line.strip().startswith("#")  # Skip empty lines and comments
        ]
        run_batch(queries)