# src/zettlr_rag/utils.py
"""
Utility functions for zettlr_rag.
"""

import pandas as pd
import os

def load_query_log(log_path: str = "query_log.jsonl") -> pd.DataFrame:
    """
    Load the local JSONL query log into a pandas DataFrame.

    Args:
        path: Path to the .jsonl file. Defaults to query_log.jsonl
              in the current working directory.

    Returns:
        DataFrame with one row per query, columns for all metrics.
        Returns empty DataFrame if file doesn't exist.
    """
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Log file not found: {log_path}")
        
    df = pd.read_json(log_path, lines=True)

    if df.empty:
        return df

    # Parse timestamp to datetime with UTC timezone
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    return df

"""
Utility functions for loading observability data into pandas DataFrames.
Supports both the local JSONL log and Langfuse SDK fetch.
"""

import os
import json
import logging
from pathlib import Path
from datetime import timezone

log = logging.getLogger(__name__)


def load_query_log(path: str = "query_log.jsonl") -> "pd.DataFrame":
    """
    Load the local JSONL query log into a pandas DataFrame.

    Args:
        path: Path to the .jsonl file. Defaults to query_log.jsonl
              in the current working directory.

    Returns:
        DataFrame with one row per query, columns for all metrics.
        Returns empty DataFrame if file doesn't exist.
    """
    import pandas as pd

    p = Path(path)
    if not p.exists():
        log.warning(f"Query log not found at {p.resolve()} — returning empty DataFrame")
        return pd.DataFrame()

    df = pd.read_json(p, lines=True)

    if df.empty:
        return df

    # Parse timestamp to datetime with UTC timezone
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    return df


def load_langfuse_traces() -> "pd.DataFrame":
    """
    Fetch all traces + scores from local Langfuse instance
    and return as a single flat DataFrame (one row per query).

    Requires Langfuse to be running and .env.langfuse to be configured.
    """
    import pandas as pd
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=".env.langfuse", override=False)

    try:
        from langfuse import get_client
        lf = get_client()
    except Exception as exc:
        log.error(f"Could not connect to Langfuse: {exc}")
        return pd.DataFrame()

    # ── Fetch traces ──────────────────────────────────────────────────────────
    traces = lf.fetch_traces().data
    trace_rows = []
    for t in traces:
        trace_rows.append({
            "trace_id":   t.id,
            "question":   t.input,
            "answer":     t.output,
            "model":      (t.metadata or {}).get("model"),
            "run_id":     (t.metadata or {}).get("run_id"),
            "timestamp":  t.timestamp,
            "latency_ms": t.latency,
        })
    traces_df = pd.DataFrame(trace_rows)

    if traces_df.empty:
        return traces_df

    # ── Fetch scores and pivot wide ───────────────────────────────────────────
    scores = lf.fetch_scores().data
    score_rows = [
        {"trace_id": s.trace_id, "metric": s.name, "value": s.value}
        for s in scores
    ]
    scores_df = pd.DataFrame(score_rows)

    if not scores_df.empty:
        scores_wide = scores_df.pivot_table(
            index="trace_id",
            columns="metric",
            values="value",
            aggfunc="first",
        ).reset_index()
        result = traces_df.merge(scores_wide, on="trace_id", how="left")
    else:
        result = traces_df

    if "timestamp" in result.columns:
        result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True)

    return result