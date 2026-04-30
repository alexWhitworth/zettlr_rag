import argparse
import nest_asyncio
import os
import sys
import time
import pandas as pd

# Add the project root to sys.path to import query.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from query import RAGQueryRunner, RAGQueryConfig
from utils import load_query_log

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
        time.sleep(2)


if __name__ == "__main__":
    with open(args.file) as f:
        queries = [
            line.strip() for line in f 
            if line.strip() and not line.strip().startswith("#")  # Skip empty lines and comments
        ]

        PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))

        log_df = load_query_log(f("{PROJECT_ROOT}/query_log.jsonl"))
        completed_questions = set(log_df["question"].unique())

        # Filter out already completed questions
        questions_to_run = [q for q in queries if q not in completed_questions]
        run_batch(questions_to_run)
