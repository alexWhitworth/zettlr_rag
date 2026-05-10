import json
import os
from unittest.mock import MagicMock, patch

import pytest
from llama_index.core.embeddings.mock_embed_model import MockEmbedding
from llama_index.core.llms.mock import MockLLM

from zettlr_rag.consts import SYSTEM_PROMPT
from zettlr_rag.rag_setup import (
    TokenCapturingGemini,
    _usage_store,
    get_last_token_usage,
    load_academic_markdown,
    main_async,
    sanitize_metadata,
    setup_settings,
)


@pytest.mark.asyncio
async def test_load_academic_markdown(tmp_path):
    # Create a mock library
    lib_dir = tmp_path / "mock_lib"
    lib_dir.mkdir()
    paper_text = (
        "---\ntitle: Mock\n---\nThis is some sufficiently long content for testing purposes."
    )
    (lib_dir / "paper1.md").write_text(paper_text)

    documents = load_academic_markdown(str(lib_dir))
    assert len(documents) > 0
    for doc in documents:
        assert doc.id_ == doc.metadata["file_path"]
        assert doc.metadata["file_path"].endswith(".md")  # verify .md filter


def test_system_prompt_is_defined():
    """SYSTEM_PROMPT should exist as a module constant, not in Settings."""
    assert SYSTEM_PROMPT  # non-empty
    assert isinstance(SYSTEM_PROMPT, str)


@pytest.mark.asyncio
async def test_main_async_survey(temp_chroma_db, temp_metadata_path, tmp_path):
    # Create mock library
    lib_dir = tmp_path / "mock_lib"
    lib_dir.mkdir()
    paper_text = (
        "---\ntitle: Mock\n---\nThis is some sufficiently long content for testing purposes."
    )
    (lib_dir / "paper1.md").write_text(paper_text)

    with (
        patch("zettlr_rag.rag_setup.GoogleGenAI") as mock_llm_class,
        patch("zettlr_rag.rag_setup.GoogleGenAIEmbedding") as mock_embed_class,
    ):
        mock_llm_class.return_value = MockLLM()
        mock_embed_class.return_value = MockEmbedding(embed_dim=768)

        await main_async(
            base_path=str(lib_dir),
            chroma_path=temp_chroma_db,
            metadata_path=temp_metadata_path,
            run_verification=False,
        )

        assert os.path.exists(temp_chroma_db)
        assert os.path.exists(temp_metadata_path)
        assert len(os.listdir(temp_metadata_path)) > 0


def test_sanitize_metadata():
    raw = {
        "str": "val",
        "int": 1,
        "float": 1.5,
        "none": None,
        "list": ["a", "b"],
        "dict": {"k": "v"},
        "other": object(),
    }
    sanitized = sanitize_metadata(raw)
    assert sanitized["str"] == "val"
    assert sanitized["int"] == 1
    assert sanitized["float"] == 1.5
    assert sanitized["none"] is None
    assert sanitized["list"] == "a, b"
    assert sanitized["dict"] == json.dumps({"k": "v"})
    assert isinstance(sanitized["other"], str)


def test_token_capturing_gemini_store():
    # Reset store
    if hasattr(_usage_store, "last_usage"):
        del _usage_store.last_usage

    with patch("zettlr_rag.rag_setup.GoogleGenAI.__init__", return_value=None):
        llm = TokenCapturingGemini(model="models/gemini-1.5-flash", api_key="dummy")

    # Mock response with raw.usage_metadata
    resp = MagicMock()
    resp.raw = MagicMock()
    resp.raw.usage_metadata = MagicMock()
    resp.raw.usage_metadata.prompt_token_count = 10
    resp.raw.usage_metadata.candidates_token_count = 20
    resp.raw.usage_metadata.cached_content_token_count = 5

    llm._store(resp)
    usage = get_last_token_usage()
    assert usage.input_tokens == 10
    assert usage.output_tokens == 20
    assert usage.cache_tokens == 5

    # Mock response with additional_kwargs
    resp2 = MagicMock()
    resp2.raw = None
    resp2.additional_kwargs = {"prompt_tokens": 100, "completion_tokens": 200}

    llm._store(resp2)
    usage = get_last_token_usage()
    assert usage.input_tokens == 110  # accumulated
    assert usage.output_tokens == 220


@patch("zettlr_rag.rag_setup.load_dotenv")
@patch("os.getenv")
def test_setup_settings_error(mock_getenv, mock_dotenv):
    mock_getenv.return_value = None
    with pytest.raises(ValueError, match="API Key not found"):
        setup_settings()


@patch("zettlr_rag.rag_setup.load_dotenv")
@patch("os.getenv")
@patch("os.environ", {})
def test_setup_settings_success(mock_getenv, mock_dotenv):
    mock_getenv.side_effect = lambda k, default=None: "dummy_key" if "API_KEY" in k else None

    with (
        patch("zettlr_rag.rag_setup.TokenCapturingGemini") as mock_gemini,
        patch("zettlr_rag.rag_setup.GoogleGenAIEmbedding") as mock_embed,
    ):
        mock_gemini.return_value = MockLLM()
        mock_embed.return_value = MockEmbedding(embed_dim=768)
        setup_settings()
        assert mock_gemini.called
