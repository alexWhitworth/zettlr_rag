import os
import pytest
from llama_index.core import Settings
from zettlr_rag.rag_setup import setup_settings

@pytest.fixture(scope="session", autouse=True)
def mock_settings():
    """Ensure settings are initialized for tests."""
    # We might want to mock the API call or use a dummy key if needed,
    # but for now we'll just run the setup.
    # If GEMINI_API_KEY is not set, we might need to mock GoogleGenAI.
    try:
        setup_settings()
    except ValueError:
        # Fallback for environments without API keys during pure logic tests
        os.environ["GEMINI_API_KEY"] = "fake-key"
        setup_settings()

@pytest.fixture
def temp_chroma_db(tmp_path):
    """Provides a temporary path for Chroma DB."""
    return str(tmp_path / "test_chroma_db")

@pytest.fixture
def temp_metadata_path(tmp_path):
    """Provides a temporary path for index metadata."""
    return str(tmp_path / "test_metadata")
