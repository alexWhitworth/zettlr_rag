import os
from unittest.mock import patch

import pytest
from llama_index.core.embeddings.mock_embed_model import MockEmbedding
from llama_index.core.llms.mock import MockLLM

from zettlr_rag.rag_setup import SYSTEM_PROMPT, load_academic_markdown, main_async


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
