# src/zettlr_rag/consts.py
"""
Central repository for all constants used in zettlr_rag.
"""

import re

# ── LLM Models ──────────────────────────────────────────────────────────────
MODEL_NAME = "gemini-3-flash-preview"
EMBEDDING_MODEL_NAME = "gemini-embedding-2-preview"

# ── Prompts ──────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "I am a Senior Staff Data Scientist, Algorithms. When I ask technical or research questions, "
    "provide high-level scientific detail and include paper citations (bibtex format). "
    "Use clean Markdown formatting with clear headers, bold text for key terms, "
    "and LaTeX for math. Prefer Python for all code examples. Assume a high level of "
    "statistical and algorithmic understanding. Provide sufficient detail to produce "
    "complete answers, but prefer brevity to unnecessarily verbose responses. "
    "Do not include conversational filler—start directly with the content."
)

# ── Pricing: USD per single token ─────────────────────────────────────────────
# Source: https://ai.google.dev/pricing (update as pricing changes)
# Keys must match MODEL_NAME exactly.
GEMINI_PRICING: dict[str, dict[str, float]] = {
    "gemini-3-flash-preview": {
        "input": 0.000_000_500,  # $0.50  per 1M input tokens
        "output": 0.000_003_000,  # $3.00  per 1M output tokens
        "cache": 0.000_000_050,  # $0.05  per 1M cached tokens
    },
    "gemini-2.0-flash": {
        "input": 0.000_000_100,
        "output": 0.000_000_400,
    },
    "gemini-2.0-flash-lite": {
        "input": 0.000_000_075,
        "output": 0.000_000_300,
    },
    "gemini-1.5-pro": {
        "input": 0.000_001_250,
        "output": 0.000_005_000,
    },
    "gemini-1.5-flash": {
        "input": 0.000_000_075,
        "output": 0.000_000_300,
    },
    "gemini-embedding-2-preview": {
        "text": 0.000_000_200,  # $0.20 per 1M tokens
        "image": 0.000_000_450,  # $0.45 per 1M tokens
        "audio": 0.000_006_500,  # $6.50 per 1M tokens
        "video": 0.000_012_000,  # $12.00 per 1M tokens
    },
    "gemini-embedding-2-preview-batch": {
        "text": 0.000_000_020,
    },
}

# ── Context window sizes in tokens ────────────────────────────────────────────
GEMINI_CONTEXT_WINDOWS: dict[str, int] = {
    # ── Generation models ────────────────────────────────────────────────────
    "gemini-3-flash-preview": 1_048_576,
    "gemini-2.0-flash": 1_048_576,
    "gemini-2.0-flash-lite": 1_048_576,
    "gemini-1.5-pro": 2_097_152,
    "gemini-1.5-flash": 1_048_576,
    # ── Embedding models ──────────────────────────────────────────────────────
    "gemini-embedding-2-preview": 8_192,
    "gemini-embedding-2-preview-batch": 8_192,
}

# --── Patterns ─────────────────────────────────────────────────────────────────
BIBTEX_PATTERN = re.compile(
    r"```bibtex|@(?:book|article|misc|inproceedings|phdthesis|techreport|unpublished)\b"
)

# ── Paths ────────────────────────────────────────────────────────────────────
CHROMA_PATH = "./chroma_db_academic"
METADATA_PATH = "./.index_metadata"
GRAPH_INDEX_PATH = "./.graph_index"
