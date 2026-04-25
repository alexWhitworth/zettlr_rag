import os
from unittest.mock import patch

import pytest
from llama_index.core.embeddings.mock_embed_model import MockEmbedding
from llama_index.core.llms.mock import MockLLM

from zettlr_rag.rag_setup import AcademicRAGSync


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
    paper_path.write_text("---\ntitle: Fixture Paper\n---\nUnique text for fixture paper.")

    return {
        "root": str(tmp_path),
        "lib": str(lib_dir),
        "chroma": str(tmp_path / "chroma"),
        "metadata": str(tmp_path / "metadata"),
    }


@pytest.mark.asyncio
async def test_full_rag_lifecycle(temp_workspace, capsys):
    """
    Test the full lifecycle of the RAG index using isolated temp state.
    """
    lib_path = temp_workspace["lib"]
    chroma_path = temp_workspace["chroma"]
    metadata_path = temp_workspace["metadata"]

    with (
        patch("zettlr_rag.rag_setup.GoogleGenAI") as mock_llm_class,
        patch("zettlr_rag.rag_setup.GoogleGenAIEmbedding") as mock_embed_class,
    ):
        mock_llm_class.return_value = MockLLM()
        mock_embed_class.return_value = MockEmbedding(embed_dim=768)

        sync_manager = AcademicRAGSync(
            base_path=lib_path, chroma_path=chroma_path, metadata_path=metadata_path
        )

        # --- PHASE 1: Initial Ingestion ---
        # 1. Create two new papers at the root
        p1_path = os.path.join(lib_path, "paper1.md")
        with open(p1_path, "w") as f:
            f.write("---\ntitle: Paper 1\nyear: 2021\n---\nUnique text for paper 1.")

        p2_path = os.path.join(lib_path, "paper2.md")
        with open(p2_path, "w") as f:
            f.write("---\ntitle: Paper 2\nyear: 2022\n---\nUnique text for paper 2.")

        await sync_manager.run_sync(run_verification=False)
        captured = capsys.readouterr()
        # 2 we just created + 1 fixture = 3
        assert "3 new files to embed" in captured.out

        # --- PHASE 2: Incremental Updates ---
        p3_path = os.path.join(lib_path, "paper3.md")
        with open(p3_path, "w") as f:
            f.write("---\ntitle: Paper 3\n---\nUnique text for paper 3.")

        with open(p1_path, "a") as f:
            f.write("\nAdded more content to paper 1.")

        await sync_manager.run_sync(run_verification=False)
        captured = capsys.readouterr()
        assert "1 new files to embed" in captured.out
        assert "1 modified files to re-embed" in captured.out

        # --- PHASE 3: Smart Move ---
        # Rename FolderA -> RenamedFolder (fixture paper moves)
        # Move paper2 -> RenamedFolder (paper2 moves)
        old_folder = os.path.join(lib_path, "FolderA")
        new_folder = os.path.join(lib_path, "RenamedFolder")
        os.rename(old_folder, new_folder)

        # move paper2 from root into the new folder
        new_p2_path = os.path.join(new_folder, "paper2.md")
        os.rename(p2_path, new_p2_path)

        await sync_manager.run_sync(run_verification=False)
        captured = capsys.readouterr()
        assert "moved files to update metadata" in captured.out
        assert "Moving: paper2.md" in captured.out

        # --- PHASE 4: Pruning ---
        os.remove(p3_path)
        await sync_manager.run_sync(run_verification=False)
        captured = capsys.readouterr()
        assert "1 stale files to prune" in captured.out
