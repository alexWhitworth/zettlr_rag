import os
from unittest.mock import patch

import pytest
from llama_index.core.embeddings.mock_embed_model import MockEmbedding
from llama_index.core.llms.mock import MockLLM

from zettlr_rag.build_graph import build_graph
from zettlr_rag.rag_setup import AcademicRAGSync


@pytest.mark.asyncio
async def test_build_graph_standalone(temp_workspace):
    """Test the standalone build_graph script logic."""
    lib_path = temp_workspace["lib"]
    chroma_path = temp_workspace["chroma"]
    metadata_path = temp_workspace["metadata"]
    graph_path = temp_workspace["graph"]

    # 1. First, populate the vector store and docstore using regular sync
    # (Mocking LLM/Embed for speed)
    with (
        patch("zettlr_rag.rag_setup.GoogleGenAI") as mock_llm_class,
        patch("zettlr_rag.rag_setup.GoogleGenAIEmbedding") as mock_embed_class,
    ):
        mock_llm_class.return_value = MockLLM()
        mock_embed_class.return_value = MockEmbedding(embed_dim=768)

        # Create a paper
        p1_path = os.path.join(lib_path, "paper1.md")
        with open(p1_path, "w") as f:
            f.write("---\ntitle: Paper 1\n---\nUnique text for paper 1.")

        sync_manager = AcademicRAGSync(
            base_path=lib_path,
            chroma_path=chroma_path,
            metadata_path=metadata_path,
            graph_path=graph_path,
        )
        await sync_manager.initialize()
        # Mock _initialize_graph for the initial sync to avoid conflict if any
        with patch.object(AcademicRAGSync, "_initialize_graph", return_value=None):
            await sync_manager.run_sync(run_verification=False)

        # 2. Now wipe the graph index (simulating cold start)
        if os.path.exists(graph_path):
            import shutil

            shutil.rmtree(graph_path)

        # 3. Run the standalone build_graph
        # Note: build_graph is synchronous and calls asyncio.run() internally via PropertyGraphIndex
        # We run it in an executor to avoid "asyncio.run() cannot be called from a running event
        # loop"
        import asyncio

        loop = asyncio.get_event_loop()

        with (
            patch("zettlr_rag.build_graph.setup_settings"),
            patch("zettlr_rag.build_graph.SchemaLLMPathExtractor", return_value=None),
            patch("zettlr_rag.build_graph.PropertyGraphIndex") as mock_pg_index,
        ):
            await loop.run_in_executor(None, build_graph, chroma_path, metadata_path, graph_path)

            # Verify it tried to create the index
            assert mock_pg_index.called
            # Verify nodes were passed to it
            _, kwargs = mock_pg_index.call_args
            assert "nodes" in kwargs
            assert len(kwargs["nodes"]) > 0
