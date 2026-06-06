# src/zettlr_rag/consts.py
"""
Central repository for all constants used in zettlr_rag.
"""

import re
from typing import Literal

# =====================================================================
# AVAILABLE MODELS
# =====================================================================

# GEMINI 3.x SERIES (Current & Recommended)
# Flagship Model (Recommended for production, coding, and agentic workflows)
GEMINI_3_5_FLASH = "gemini-3.5-flash"

# High-efficiency / Low-cost / Low-latency Models
GEMINI_3_1_FLASH_LITE = "gemini-3.1-flash-lite"
GEMINI_3_FLASH_PREVIEW = "gemini-3-flash-preview"

# Advanced Reasoning & Preview Models
GEMINI_3_1_PRO_PREVIEW = "gemini-3.1-pro-preview"
GEMINI_3_1_PRO_PREVIEW_CUSTOMTOOLS = "gemini-3.1-pro-preview-customtools"

# Multimodal Visual & Audio Generation
GEMINI_3_1_FLASH_IMAGE = "gemini-3.1-flash-image"
GEMINI_3_PRO_IMAGE = "gemini-3-pro-image"
GEMINI_3_1_FLASH_TTS_PREVIEW = "gemini-3-1-flash-tts-preview"

# GEMINI 2.5 SERIES (Legacy / Deprecating)
# Note: Google is scheduled to phase these out mid-2026.
# It is recommended to migrate to the 3.x series equivalents.
GEMINI_2_5_FLASH = "gemini-2.5-flash"
GEMINI_2_5_FLASH_LITE = "gemini-2.5-flash-lite"

# =====================================================================
# Current models used in this project (update as needed)
# =====================================================================
MODEL_NAME = "gemini-3-flash-preview"
EMBEDDING_MODEL_NAME = "gemini-embedding-2-preview"
BUILD_GRAPH_MODEL = "gemini-2.5-flash-lite"

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
    "gemini-2.5-flash-lite": {
        "input": 0.000_000_100,  # $0.10 per 1M input tokens
        "output": 0.000_000_400,  # $0.40 per 1M output tokens (incl. thinking)
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

# ── Property Graph Schema ───────────────────────────────────────────────────
GRAPH_ENTITIES = Literal[
    "Document", "Author", "Method", "Dataset", "Metric", "Concept", "Organization"
]

GRAPH_RELATIONS = Literal[
    "AUTHORED_BY",
    "USES_METHOD",
    "BENCHMARKED_ON",
    "IMPROVES_UPON",
    "DEFINES",
    "REPORTS",
    "AFFILIATED_WITH",
]
CHROMA_PATH = "./chroma_db_academic"
METADATA_PATH = "./.index_metadata"
GRAPH_INDEX_PATH = "./.graph_index"
