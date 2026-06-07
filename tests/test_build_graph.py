import os
from unittest.mock import MagicMock, call, patch

import pytest
from llama_index.core.embeddings.mock_embed_model import MockEmbedding
from llama_index.core.llms.mock import MockLLM

from zettlr_rag.build_graph import CHECKPOINT_FILE, build_graph
from zettlr_rag.rag_setup import AcademicRAGSync


def _run_build_graph(chroma_path, metadata_path, graph_path):
    """Run build_graph with all external dependencies mocked."""
    with (
        patch("zettlr_rag.build_graph.setup_settings"),
        patch("zettlr_rag.build_graph.SchemaLLMPathExtractor", return_value=MagicMock()),
        patch("zettlr_rag.build_graph.PropertyGraphIndex") as mock_pg_cls,
        patch("zettlr_rag.build_graph.load_index_from_storage") as mock_load,
    ):
        mock_pg_instance = MagicMock()
        mock_pg_cls.return_value = mock_pg_instance
        mock_load.return_value = mock_pg_instance
        build_graph(chroma_path, metadata_path, graph_path, batch_size=2)
        return mock_pg_cls, mock_pg_instance


@pytest.mark.asyncio
async def test_build_graph_fresh_start(temp_workspace):
    """Test full build from scratch: initializes empty index, processes all nodes in batches."""
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

        for i in range(3):
            p = os.path.join(lib_path, f"paper{i}.md")
            with open(p, "w") as f:
                f.write(f"---\ntitle: Paper {i}\n---\nUnique text for paper {i}.")

        sync_manager = AcademicRAGSync(
            base_path=lib_path,
            chroma_path=chroma_path,
            metadata_path=metadata_path,
            graph_path=graph_path,
        )
        await sync_manager.initialize()
        with patch.object(AcademicRAGSync, "_initialize_graph", return_value=None):
            await sync_manager.run_sync(run_verification=False)

    import asyncio
    loop = asyncio.get_event_loop()

    with (
        patch("zettlr_rag.build_graph.setup_settings"),
        patch("zettlr_rag.build_graph.SchemaLLMPathExtractor", return_value=MagicMock()),
        patch("zettlr_rag.build_graph.PropertyGraphIndex") as mock_pg_cls,
    ):
        mock_pg_instance = MagicMock()
        mock_pg_cls.return_value = mock_pg_instance

        await loop.run_in_executor(
            None, build_graph, chroma_path, metadata_path, graph_path
        )

        assert mock_pg_cls.called
        _, kwargs = mock_pg_cls.call_args
        assert kwargs["nodes"] == []  # initialized empty
        assert mock_pg_instance.insert_nodes.called
        assert mock_pg_instance.storage_context.persist.called


@pytest.mark.asyncio
async def test_build_graph_checkpoint_resume(temp_workspace):
    """Test that re-running skips already-processed nodes and only inserts remaining ones."""
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

        for i in range(4):
            p = os.path.join(lib_path, f"paper{i}.md")
            with open(p, "w") as f:
                f.write(f"---\ntitle: Paper {i}\n---\nUnique text for paper {i}.")

        sync_manager = AcademicRAGSync(
            base_path=lib_path,
            chroma_path=chroma_path,
            metadata_path=metadata_path,
            graph_path=graph_path,
        )
        await sync_manager.initialize()
        with patch.object(AcademicRAGSync, "_initialize_graph", return_value=None):
            await sync_manager.run_sync(run_verification=False)

    # Collect real node IDs from docstore
    import chromadb
    from llama_index.core import StorageContext, load_index_from_storage
    from llama_index.vector_stores.chroma import ChromaVectorStore

    db = chromadb.PersistentClient(path=chroma_path)
    col = db.get_or_create_collection("research_papers")
    vs = ChromaVectorStore(chroma_collection=col)
    sc = StorageContext.from_defaults(vector_store=vs, persist_dir=metadata_path)

    with patch("zettlr_rag.build_graph.setup_settings"):
        from llama_index.core import VectorStoreIndex
        from typing import cast
        idx = cast(VectorStoreIndex, load_index_from_storage(sc))
        all_nodes = [
            n for n in idx.docstore.docs.values()
            if hasattr(n, "embedding") and n.embedding is not None
        ]

    assert len(all_nodes) >= 2, "Need at least 2 nodes to test partial checkpoint"

    # Simulate a prior partial run — first node already done
    import json
    checkpoint_path = os.path.join(graph_path, CHECKPOINT_FILE)
    pre_processed = {all_nodes[0].node_id}
    with open(checkpoint_path, "w") as f:
        json.dump(list(pre_processed), f)

    # Create a fake index_store.json so resume path is taken
    import pathlib
    pathlib.Path(os.path.join(graph_path, "index_store.json")).write_text("{}")

    import asyncio
    loop = asyncio.get_event_loop()

    # Load real source index so the mock can return it for the first load_index_from_storage call
    from llama_index.core import VectorStoreIndex
    from typing import cast as tcast
    real_source_index = tcast(VectorStoreIndex, load_index_from_storage(sc))

    with (
        patch("zettlr_rag.build_graph.setup_settings"),
        patch("zettlr_rag.build_graph.SchemaLLMPathExtractor", return_value=MagicMock()),
        patch("zettlr_rag.build_graph.PropertyGraphIndex") as mock_pg_cls,
        patch("zettlr_rag.build_graph.load_index_from_storage") as mock_load,
        patch("zettlr_rag.build_graph.StorageContext"),
    ):
        mock_pg_instance = MagicMock()
        mock_pg_cls.return_value = mock_pg_instance
        # First call: source docstore; second call: resume graph index
        mock_load.side_effect = [real_source_index, mock_pg_instance]

        await loop.run_in_executor(
            None, build_graph, chroma_path, metadata_path, graph_path, 10
        )

        # PropertyGraphIndex constructor should NOT be called (resumed from disk)
        mock_pg_cls.assert_not_called()

        # Only remaining nodes should have been inserted
        inserted_ids: set[str] = set()
        for c in mock_pg_instance.insert_nodes.call_args_list:
            for node in c.args[0]:
                inserted_ids.add(node.node_id)

        assert all_nodes[0].node_id not in inserted_ids, "Pre-processed node should be skipped"
        for n in all_nodes[1:]:
            assert n.node_id in inserted_ids, f"Node {n.node_id} should have been processed"


@pytest.mark.asyncio
async def test_build_graph_nothing_to_do(temp_workspace):
    """Test that re-running when all nodes are checkpointed exits cleanly."""
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

        p = os.path.join(lib_path, "paper0.md")
        with open(p, "w") as f:
            f.write("---\ntitle: Paper 0\n---\nUnique text.")

        sync_manager = AcademicRAGSync(
            base_path=lib_path,
            chroma_path=chroma_path,
            metadata_path=metadata_path,
            graph_path=graph_path,
        )
        await sync_manager.initialize()
        with patch.object(AcademicRAGSync, "_initialize_graph", return_value=None):
            await sync_manager.run_sync(run_verification=False)

    # Checkpoint all nodes upfront
    import chromadb
    from llama_index.core import StorageContext, load_index_from_storage
    from llama_index.vector_stores.chroma import ChromaVectorStore
    import json

    db = chromadb.PersistentClient(path=chroma_path)
    col = db.get_or_create_collection("research_papers")
    vs = ChromaVectorStore(chroma_collection=col)
    sc = StorageContext.from_defaults(vector_store=vs, persist_dir=metadata_path)
    with patch("zettlr_rag.build_graph.setup_settings"):
        from llama_index.core import VectorStoreIndex
        from typing import cast
        idx = cast(VectorStoreIndex, load_index_from_storage(sc))
        all_ids = [
            n.node_id for n in idx.docstore.docs.values()
            if hasattr(n, "embedding") and n.embedding is not None
        ]

    checkpoint_path = os.path.join(graph_path, CHECKPOINT_FILE)
    with open(checkpoint_path, "w") as f:
        json.dump(all_ids, f)

    import asyncio
    loop = asyncio.get_event_loop()

    with (
        patch("zettlr_rag.build_graph.setup_settings"),
        patch("zettlr_rag.build_graph.PropertyGraphIndex") as mock_pg_cls,
    ):
        await loop.run_in_executor(
            None, build_graph, chroma_path, metadata_path, graph_path
        )
        mock_pg_cls.assert_not_called()


@pytest.mark.asyncio
async def test_triplets_extracted_into_graph_store(temp_workspace):
    """Regression test: verify triplets from the KG extractor land in the graph store.

    Previously, strict=True silently dropped all LLM output, leaving
    property_graph_store.graph.relations empty even after a full build.
    This test uses a real PropertyGraphIndex and a mock LLM that returns one
    valid KGSchema triplet, then asserts the relation is present in the store.
    """
    import asyncio
    import os

    import chromadb
    from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
    from llama_index.core.indices.property_graph import SchemaLLMPathExtractor
    from llama_index.vector_stores.chroma import ChromaVectorStore
    from typing import cast

    from zettlr_rag.consts import GRAPH_ENTITIES, GRAPH_RELATIONS

    lib_path = temp_workspace["lib"]
    chroma_path = temp_workspace["chroma"]
    metadata_path = temp_workspace["metadata"]
    graph_path = temp_workspace["graph"]

    # --- Step 1: index a document so the docstore has an embedded node ---
    with (
        patch("zettlr_rag.rag_setup.GoogleGenAI") as mock_llm_class,
        patch("zettlr_rag.rag_setup.GoogleGenAIEmbedding") as mock_embed_class,
    ):
        mock_llm_class.return_value = MockLLM()
        mock_embed_class.return_value = MockEmbedding(embed_dim=768)

        sync_manager = AcademicRAGSync(
            base_path=lib_path,
            chroma_path=chroma_path,
            metadata_path=metadata_path,
            graph_path=graph_path,
        )
        await sync_manager.initialize()
        with patch.object(AcademicRAGSync, "_initialize_graph", return_value=None):
            await sync_manager.run_sync(run_verification=False)

    # --- Step 2: build a real graph using a mock LLM that returns one triplet ---
    # Build an extractor so we can grab its kg_schema_cls to construct a valid response.
    from llama_index.core.llms.mock import MockLLM as _MockLLM

    extractor_probe = SchemaLLMPathExtractor(
        llm=_MockLLM(),
        possible_entities=GRAPH_ENTITIES,
        possible_relations=GRAPH_RELATIONS,
        strict=False,
    )
    KGSchema = extractor_probe.kg_schema_cls

    valid_schema = KGSchema(
        triplets=[
            {
                "subject": {"name": "FixturePaper", "type": "Document"},
                "relation": {"type": "AUTHORED_BY"},
                "object": {"name": "Smith", "type": "Author"},
            }
        ]
    )

    class TripletMockLLM(_MockLLM):
        """MockLLM that returns a valid KGSchema with one triplet."""

        async def astructured_predict(self, output_cls, prompt, **kwargs):
            return valid_schema

        def structured_predict(self, output_cls, prompt, **kwargs):
            return valid_schema

    loop = asyncio.get_event_loop()
    with (
        patch("zettlr_rag.build_graph.setup_settings"),
        patch(
            "zettlr_rag.build_graph._make_kg_extractor",
            return_value=SchemaLLMPathExtractor(
                llm=TripletMockLLM(),
                possible_entities=GRAPH_ENTITIES,
                possible_relations=GRAPH_RELATIONS,
                strict=False,
            ),
        ),
    ):
        await loop.run_in_executor(
            None, build_graph, chroma_path, metadata_path, graph_path
        )

    # --- Step 3: reload the persisted graph and assert relations > 0 ---
    pg_storage = StorageContext.from_defaults(persist_dir=graph_path)
    from llama_index.core import PropertyGraphIndex
    pg_index = cast(PropertyGraphIndex, load_index_from_storage(pg_storage))

    relations = pg_index.property_graph_store.graph.relations
    assert len(relations) > 0, (
        f"Expected at least 1 relation in graph store, got 0. "
        f"This likely means strict=True is silently dropping all extractor output."
    )
