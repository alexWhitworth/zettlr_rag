import os
import json

import pytest
from llama_index.core import Settings
from llama_index.core.embeddings.mock_embed_model import MockEmbedding
from llama_index.core.llms.mock import MockLLM
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core.base.llms.types import ChatResponse, ChatMessage


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


@pytest.fixture
def temp_graph_path(tmp_path):
    """Provides a temporary path for property graph metadata."""
    path = tmp_path / "test_graph"
    path.mkdir(exist_ok=True)
    return str(path)


@pytest.fixture
def temp_workspace(tmp_path):
    """Create a temporary workspace with reproducible state."""
    # Create library structure
    lib_dir = tmp_path / "library"
    lib_dir.mkdir()

    # Create subfolder and fixture paper with unique content
    folder_a = lib_dir / "FolderA"
    folder_a.mkdir()

    paper_path = folder_a / "paper1.md"
    paper_path.write_text(
        "---\ntitle: Fixture Paper\n---\n"
        "Unique text for fixture paper. This paper presents a comprehensive analysis "
        "of the subject matter using rigorous empirical methodology."
    )

    graph_dir = tmp_path / "graph"
    graph_dir.mkdir()

    return {
        "root": str(tmp_path),
        "lib": str(lib_dir),
        "chroma": str(tmp_path / "chroma"),
        "metadata": str(tmp_path / "metadata"),
        "graph": str(tmp_path / "graph"),
    }


class TripleExtractingMockLLM(MockLLM):
    """Returns minimal valid SchemaLLMPathExtractor output for testing."""

    def chat(self, messages, **kwargs):
        # Return a valid triple that SchemaLLMPathExtractor can parse
        triple = {
            "entities": [
                {"name": "test_entity_a", "type": "CONCEPT"},
                {"name": "test_entity_b", "type": "CONCEPT"},
            ],
            "relations": [
                {
                    "source": "test_entity_a",
                    "target": "test_entity_b",
                    "type": "RELATED_TO",
                }
            ],
        }
        return ChatResponse(
            message=ChatMessage(role="assistant", content=json.dumps(triple))
        )

    async def achat(self, messages, **kwargs):
        return self.chat(messages, **kwargs)

    def structured_predict(self, output_cls, prompt, **kwargs):
        # SchemaLLMPathExtractor uses structured_predict
        entities = [
            output_cls.__fields__["entities"].outer_type_(
                name="test_entity_a", label="CONCEPT"
            ),
            output_cls.__fields__["entities"].outer_type_(
                name="test_entity_b", label="CONCEPT"
            ),
        ]
        return output_cls(entities=entities, relations=[])

    async def astructured_predict(self, output_cls, prompt, **kwargs):
        return self.structured_predict(output_cls, prompt, **kwargs)