import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from llama_index.core.embeddings.mock_embed_model import MockEmbedding
from llama_index.core.llms.mock import MockLLM

from zettlr_rag.consts import SYSTEM_PROMPT
from zettlr_rag.rag_setup import (
    AcademicRAGSync,
    TokenCapturingGemini,
    _usage_store,
    get_last_token_usage,
    load_academic_markdown,
    main_async,
    reset_token_usage,
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
async def test_main_async_survey(temp_chroma_db, temp_metadata_path, temp_graph_path, tmp_path):
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
            graph_path=temp_graph_path,
            run_verification=False,
        )

        assert os.path.exists(temp_chroma_db)
        assert os.path.exists(temp_metadata_path)
        assert os.path.exists(temp_graph_path)
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


@patch("zettlr_rag.rag_setup.load_dotenv")
@patch("os.getenv")
def test_setup_settings_copies_gemini_key_to_google(mock_getenv, mock_dotenv):
    """When GOOGLE_API_KEY is absent, setup_settings copies GEMINI_API_KEY into os.environ."""
    mock_getenv.side_effect = lambda k, default=None: (
        "gemini-key" if k == "GEMINI_API_KEY" else None
    )
    with (
        patch("os.environ", {}) as mock_env,
        patch("zettlr_rag.rag_setup.TokenCapturingGemini") as mock_gemini,
        patch("zettlr_rag.rag_setup.GoogleGenAIEmbedding") as mock_embed,
    ):
        mock_gemini.return_value = MockLLM()
        mock_embed.return_value = MockEmbedding(embed_dim=768)
        setup_settings()
        assert mock_env.get("GOOGLE_API_KEY") == "gemini-key"


def test_reset_token_usage():
    """reset_token_usage clears accumulated counts to zero."""
    _usage_store.last_usage = MagicMock(input_tokens=99, output_tokens=88)
    reset_token_usage()
    usage = get_last_token_usage()
    assert usage.input_tokens == 0
    assert usage.output_tokens == 0


def test_token_capturing_gemini_complete():
    """complete() calls super and stores token usage."""
    with patch("zettlr_rag.rag_setup.GoogleGenAI.__init__", return_value=None):
        llm = TokenCapturingGemini(model="models/gemini-1.5-flash", api_key="dummy")

    mock_resp = MagicMock()
    mock_resp.raw = None
    mock_resp.additional_kwargs = {"prompt_tokens": 5, "completion_tokens": 10}

    with patch("zettlr_rag.rag_setup.GoogleGenAI.complete", return_value=mock_resp):
        reset_token_usage()
        result = llm.complete("hello")
        assert result is mock_resp
        assert get_last_token_usage().input_tokens == 5
        assert get_last_token_usage().output_tokens == 10


def test_token_capturing_gemini_chat():
    """chat() calls super and stores token usage."""
    with patch("zettlr_rag.rag_setup.GoogleGenAI.__init__", return_value=None):
        llm = TokenCapturingGemini(model="models/gemini-1.5-flash", api_key="dummy")

    mock_resp = MagicMock()
    mock_resp.raw = None
    mock_resp.additional_kwargs = {"prompt_tokens": 7, "completion_tokens": 14}

    with patch("zettlr_rag.rag_setup.GoogleGenAI.chat", return_value=mock_resp):
        reset_token_usage()
        result = llm.chat([])
        assert result is mock_resp
        assert get_last_token_usage().input_tokens == 7
        assert get_last_token_usage().output_tokens == 14


@pytest.mark.asyncio
async def test_token_capturing_gemini_acomplete():
    """acomplete() calls super and stores token usage."""
    with patch("zettlr_rag.rag_setup.GoogleGenAI.__init__", return_value=None):
        llm = TokenCapturingGemini(model="models/gemini-1.5-flash", api_key="dummy")

    mock_resp = MagicMock()
    mock_resp.raw = None
    mock_resp.additional_kwargs = {"prompt_tokens": 3, "completion_tokens": 6}

    with patch("zettlr_rag.rag_setup.GoogleGenAI.acomplete", new=AsyncMock(return_value=mock_resp)):
        reset_token_usage()
        result = await llm.acomplete("hello")
        assert result is mock_resp
        assert get_last_token_usage().input_tokens == 3


@pytest.mark.asyncio
async def test_token_capturing_gemini_achat():
    """achat() calls super and stores token usage."""
    with patch("zettlr_rag.rag_setup.GoogleGenAI.__init__", return_value=None):
        llm = TokenCapturingGemini(model="models/gemini-1.5-flash", api_key="dummy")

    mock_resp = MagicMock()
    mock_resp.raw = None
    mock_resp.additional_kwargs = {"prompt_tokens": 4, "completion_tokens": 8}

    with patch("zettlr_rag.rag_setup.GoogleGenAI.achat", new=AsyncMock(return_value=mock_resp)):
        reset_token_usage()
        result = await llm.achat([])
        assert result is mock_resp
        assert get_last_token_usage().output_tokens == 8


def test_load_academic_markdown_missing_dir():
    """load_academic_markdown raises FileNotFoundError for nonexistent path."""
    with pytest.raises(FileNotFoundError, match="Directory not found"):
        load_academic_markdown("/nonexistent/path/that/does/not/exist")


@pytest.mark.asyncio
async def test_plan_sync_raises_before_initialize(temp_workspace):
    """plan_sync raises RuntimeError when index is not yet initialized."""
    mgr = AcademicRAGSync(
        base_path=temp_workspace["lib"],
        chroma_path=temp_workspace["chroma"],
        metadata_path=temp_workspace["metadata"],
        graph_path=temp_workspace["graph"],
    )
    with pytest.raises(RuntimeError, match="Index not initialized"):
        mgr.plan_sync([])


@pytest.mark.asyncio
async def test_execute_moves_raises_before_initialize(temp_workspace):
    """execute_moves raises RuntimeError when index is not initialized."""
    mgr = AcademicRAGSync(
        base_path=temp_workspace["lib"],
        chroma_path=temp_workspace["chroma"],
        metadata_path=temp_workspace["metadata"],
        graph_path=temp_workspace["graph"],
    )
    with pytest.raises(RuntimeError, match="Index or collection not initialized"):
        mgr.execute_moves([])


@pytest.mark.asyncio
async def test_execute_deletions_raises_before_initialize(temp_workspace):
    """execute_deletions raises RuntimeError when index is not initialized."""
    mgr = AcademicRAGSync(
        base_path=temp_workspace["lib"],
        chroma_path=temp_workspace["chroma"],
        metadata_path=temp_workspace["metadata"],
        graph_path=temp_workspace["graph"],
    )
    with pytest.raises(RuntimeError, match="Index not initialized"):
        mgr.execute_deletions(["some-id"])


@pytest.mark.asyncio
async def test_index_documents_raises_before_initialize(temp_workspace):
    """index_documents raises RuntimeError when index is not initialized."""
    mgr = AcademicRAGSync(
        base_path=temp_workspace["lib"],
        chroma_path=temp_workspace["chroma"],
        metadata_path=temp_workspace["metadata"],
        graph_path=temp_workspace["graph"],
    )
    from llama_index.core.schema import Document
    with pytest.raises(RuntimeError, match="Index or vector store not initialized"):
        await mgr.index_documents([Document(text="test")])


@pytest.mark.asyncio
async def test_initialize_graph_loads_existing(temp_workspace):
    """_initialize_graph loads from disk when graph_path is populated."""
    import pathlib
    graph_path = temp_workspace["graph"]
    # Simulate a populated graph directory
    pathlib.Path(os.path.join(graph_path, "index_store.json")).write_text("{}")

    mgr = AcademicRAGSync(
        base_path=temp_workspace["lib"],
        chroma_path=temp_workspace["chroma"],
        metadata_path=temp_workspace["metadata"],
        graph_path=graph_path,
    )

    mock_pg = MagicMock()
    with (
        patch("zettlr_rag.rag_setup.StorageContext") as mock_sc,
        patch("zettlr_rag.rag_setup.load_index_from_storage", return_value=mock_pg),
    ):
        await mgr._initialize_graph()
        assert mgr.pg_index is mock_pg


@pytest.mark.asyncio
async def test_index_documents_graph_update(temp_workspace):
    """index_documents triggers incremental graph update when pg_index is set."""
    lib_path = temp_workspace["lib"]
    chroma_path = temp_workspace["chroma"]
    metadata_path = temp_workspace["metadata"]
    graph_path = temp_workspace["graph"]

    with (
        patch("zettlr_rag.rag_setup.GoogleGenAI") as mock_llm_class,
        patch("zettlr_rag.rag_setup.GoogleGenAIEmbedding") as mock_embed_class,
    ):
        mock_llm_class.return_value = MockLLM()
        mock_embed_class.return_value = MockEmbedding(embed_dim=768)

        import os
        p = os.path.join(lib_path, "paper.md")
        with open(p, "w") as f:
            f.write("---\ntitle: Graph Test\n---\nContent for graph update test.")

        mgr = AcademicRAGSync(
            base_path=lib_path,
            chroma_path=chroma_path,
            metadata_path=metadata_path,
            graph_path=graph_path,
        )
        await mgr.initialize()
        # Inject a mock pg_index so the graph-update branch executes
        mock_pg = MagicMock()
        mgr.pg_index = mock_pg

        docs = load_academic_markdown(lib_path)
        with patch("zettlr_rag.rag_setup.nest_asyncio"):
            with patch("zettlr_rag.rag_setup.SchemaLLMPathExtractor"):
                await mgr.index_documents(docs)

        assert mock_pg.insert_nodes.called


@pytest.mark.asyncio
async def test_index_documents_batch_embed_fallback(temp_workspace):
    """index_documents falls back to per-node embedding when batch embed raises."""
    lib_path = temp_workspace["lib"]
    individual_calls = []

    class FailingBatchEmbed(MockEmbedding):
        """MockEmbedding that fails on batch but succeeds individually."""

        def get_text_embedding_batch(self, texts, **kwargs):  # type: ignore[override]
            raise RuntimeError("batch failed")

        def get_text_embedding(self, text):  # type: ignore[override]
            individual_calls.append(text)
            return [0.1] * 768

    with (
        patch("zettlr_rag.rag_setup.GoogleGenAI") as mock_llm_class,
        patch("zettlr_rag.rag_setup.GoogleGenAIEmbedding") as mock_embed_class,
    ):
        mock_llm_class.return_value = MockLLM()
        mock_embed_class.return_value = FailingBatchEmbed(embed_dim=768)

        p = os.path.join(lib_path, "paper.md")
        with open(p, "w") as f:
            f.write("---\ntitle: Fallback Test\n---\nContent for fallback embedding test.")

        mgr = AcademicRAGSync(
            base_path=lib_path,
            chroma_path=temp_workspace["chroma"],
            metadata_path=temp_workspace["metadata"],
            graph_path=temp_workspace["graph"],
        )
        await mgr.initialize()

        docs = load_academic_markdown(lib_path)
        await mgr.index_documents(docs)

    assert len(individual_calls) > 0
