import os
import pytest
from unittest.mock import patch
from llama_index.core import Settings
from llama_index.core.llms.mock import MockLLM
from llama_index.core.embeddings.mock_embed_model import MockEmbedding
from llama_index.core.node_parser import MarkdownNodeParser


@pytest.fixture(scope="session", autouse=True)
def mock_settings():
    """Initialize LlamaIndex Settings with mocks — avoids real API calls in tests."""
    # Ensure API key exists so setup_settings() doesn't raise
    os.environ.setdefault("GEMINI_API_KEY", "fake-key-for-testing")
    os.environ.setdefault("GOOGLE_API_KEY", "fake-key-for-testing")

    # Configure Settings with mocks directly — bypass setup_settings() which
    # would try to instantiate real Gemini clients.
    Settings.llm = MockLLM()
    Settings.embed_model = MockEmbedding(embed_dim=768)
    Settings.node_parser = MarkdownNodeParser()

    yield

    # Cleanup: reset settings
    Settings.llm = None
    Settings.embed_model = None


@pytest.fixture
def temp_chroma_db(tmp_path):
    """Provides a temporary path for Chroma DB."""
    return str(tmp_path / "test_chroma_db")


@pytest.fixture
def temp_metadata_path(tmp_path):
    """Provides a temporary path for index metadata."""
    path = tmp_path / "test_metadata"
    path.mkdir(exist_ok=True)
    return str(path)
