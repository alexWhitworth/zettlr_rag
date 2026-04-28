# src/zettlr_rag/utils.py
"""
Utility functions for zettlr_rag.
"""

import pandas as pd
import os

def load_query_log(log_path: str = "query_log.jsonl") -> pd.DataFrame:
    """
    Loads the persistent query log JSONL file into a pandas DataFrame.
    """
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Log file not found: {log_path}")
        
    return pd.read_json(log_path, lines=True)
