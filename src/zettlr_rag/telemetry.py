# src/zettlr_rag/telemetry.py
"""
Langfuse observability setup for zettlr_rag.
Import init_telemetry() once at application startup (top of query.py).
All other modules import GEMINI_PRICING and GEMINI_CONTEXT_WINDOWS from here.
"""

import logging
import os
from typing import Any

log = logging.getLogger(__name__)

def init_telemetry() -> bool:
    """
    Initialize Langfuse + OpenInference instrumentation for LlamaIndex.

    In SDK v3 the instrumentor is global — no handler object is returned
    or passed around. Returns True if instrumentation was applied,
    False if credentials are missing or packages not installed.

    Usage in query.py:
        instrumented = init_telemetry()
        # No CallbackManager needed — instrumentation is automatic
    """
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    host       = os.getenv("LANGFUSE_HOST", "http://localhost:3000")

    if not public_key or not secret_key:
        log.warning(
            "Langfuse credentials not found. "
            "Queries will run without observability. "
            "Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY in .env.langfuse"
        )
        return False

    try:
        from openinference.instrumentation.llama_index import LlamaIndexInstrumentor

        # v3: configure via environment variables — SDK reads them automatically
        os.environ.setdefault("LANGFUSE_PUBLIC_KEY", public_key)
        os.environ.setdefault("LANGFUSE_SECRET_KEY", secret_key)
        os.environ.setdefault("LANGFUSE_HOST", host)

        # One-line global instrumentation — captures ALL LlamaIndex calls
        # automatically as OTel spans forwarded to Langfuse
        LlamaIndexInstrumentor().instrument()

        log.info(f"Langfuse v3 observability enabled → {host}")
        return True

    except ImportError:
        log.warning(
            "openinference-instrumentation-llama-index not installed. "
            "Run: pip install openinference-instrumentation-llama-index"
        )
        return False

    except Exception as exc:
        log.warning(f"Langfuse initialization failed: {exc}. Proceeding without observability.")
        return False


def get_langfuse_client() -> Any:
    """
    Returns the active Langfuse v3 client for score posting.
    Returns None if langfuse is not installed or not configured.
    """
    try:
        from langfuse import get_client
        return get_client()
    except Exception:
        return None


def is_prompt_logging_enabled() -> bool:
    """Returns True if full prompt/completion logging is enabled."""
    return os.getenv("LANGFUSE_PROMPT_LOGGING_ENABLED", "true").lower() == "true"


def is_streaming_enabled() -> bool:
    """Returns True if streaming mode is enabled (required for TTFT measurement)."""
    return os.getenv("RAG_STREAMING_ENABLED", "false").lower() == "true"
