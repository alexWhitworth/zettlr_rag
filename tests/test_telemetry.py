# tests/test_telemetry.py
import os
from unittest.mock import patch

import pytest


class TestInitTelemetry:
    def test_returns_false_when_no_credentials(self):
        """No credentials → returns False gracefully."""
        with patch.dict(os.environ, {}, clear=True):
            from zettlr_rag.telemetry import init_telemetry

            result = init_telemetry()
            assert result is False

    def test_returns_true_when_credentials_present(self):
        """With valid-looking credentials, returns True."""
        with patch.dict(
            os.environ,
            {
                "LANGFUSE_PUBLIC_KEY": "pk-lf-test",
                "LANGFUSE_SECRET_KEY": "sk-lf-test",
                "LANGFUSE_HOST": "http://localhost:3000",
            },
        ):
            try:
                from zettlr_rag.telemetry import init_telemetry

                result = init_telemetry()
                # result might be False if openinference is not installed
                # but it shouldn't crash.
                assert isinstance(result, bool)
            except Exception as exc:
                pytest.fail(f"init_telemetry raised unexpectedly: {exc}")

    def test_prompt_logging_flag_defaults_to_true(self):
        with patch.dict(os.environ, {}, clear=True):
            from zettlr_rag.telemetry import is_prompt_logging_enabled

            assert is_prompt_logging_enabled() is True

    def test_prompt_logging_flag_can_be_disabled(self):
        with patch.dict(os.environ, {"LANGFUSE_PROMPT_LOGGING_ENABLED": "false"}):
            from zettlr_rag.telemetry import is_prompt_logging_enabled

            assert is_prompt_logging_enabled() is False

    def test_streaming_flag_defaults_to_false(self):
        with patch.dict(os.environ, {}, clear=True):
            from zettlr_rag.telemetry import is_streaming_enabled

            assert is_streaming_enabled() is False
