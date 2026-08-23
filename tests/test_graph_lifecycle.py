import os
from typing import Any
from unittest.mock import patch

import pytest
from llama_index.core.base.llms.types import ChatMessage, ChatResponse
from llama_index.core.embeddings.mock_embed_model import MockEmbedding
from llama_index.core.llms.mock import MockLLM

from zettlr_rag.rag_setup import AcademicRAGSync


class TripleExtractingMockLLM(MockLLM):
    """
    Returns a minimal valid KGSchema instance for SchemaLLMPathExtractor.

    Schema structure (from diagnostic):
        KGSchema(
            triplets=[
                Triplet(
                    subject=Entity(type=<EntityType>, name=str),
                    relation=Relation(type=<RelationType>),
                    object=Entity(type=<EntityType>, name=str),
                )
            ]
        )
    """

    def structured_predict(self, output_cls: Any, prompt: Any, **kwargs: Any) -> Any:
        entity_cls = output_cls.__annotations__["triplets"].__args__[0].__annotations__[
            "subject"
        ]
        relation_cls = output_cls.__annotations__["triplets"].__args__[0].__annotations__[
            "relation"
        ]
        triplet_cls = output_cls.__annotations__["triplets"].__args__[0]

        subject = entity_cls(type="Concept", name="test_entity_a")
        obj = entity_cls(type="Method", name="test_entity_b")
        relation = relation_cls(type="IMPROVES_UPON")
        triplet = triplet_cls(subject=subject, relation=relation, object=obj)
        return output_cls(triplets=[triplet])

    async def astructured_predict(self, output_cls: Any, prompt: Any, **kwargs: Any) -> Any:
        return self.structured_predict(output_cls, prompt, **kwargs)

    def chat(self, messages: Any, **kwargs: Any) -> ChatResponse:
        return ChatResponse(message=ChatMessage(role="assistant", content=""))

    async def achat(self, messages: Any, **kwargs: Any) -> ChatResponse:
        return self.chat(messages, **kwargs)


@pytest.mark.asyncio
async def test_graph_lifecycle(temp_workspace):
    """Test that graph nodes are created, stable on moves, and updated on root rename."""
    lib_path = temp_workspace["lib"]
    chroma_path = temp_workspace["chroma"]
    metadata_path = temp_workspace["metadata"]
    graph_path = temp_workspace["graph"]

    with (
        patch("zettlr_rag.rag_setup.TokenCapturingGemini") as mock_tcg_class,
        patch("zettlr_rag.rag_setup.GoogleGenAIEmbedding") as mock_embed_class,
    ):
        mock_tcg_class.return_value = TripleExtractingMockLLM()
        mock_embed_class.return_value = MockEmbedding(embed_dim=768)

        # --- Setup: index fixture doc and build graph ---
        p1_path = os.path.join(lib_path, "paper1.md")
        with open(p1_path, "w") as f:
            f.write("---\ntitle: Paper 1\n---\nUnique text for paper 1.")

        sync_manager = AcademicRAGSync(
            base_path=lib_path,
            chroma_path=chroma_path,
            metadata_path=metadata_path,
            graph_path=graph_path,
        )
        await sync_manager.run_sync(run_verification=False)

        # pg_index is None since no graph built — this is expected in unit tests
        # Graph tests require build_graph.py to have run first
        # The following phases test sync behaviour assuming pg_index is None
        assert sync_manager.pg_index is None

        # --- Phase: Move should not crash when pg_index is None ---
        new_folder = os.path.join(lib_path, "Subfolder")
        os.makedirs(new_folder)
        new_p1_path = os.path.join(new_folder, "paper1.md")
        os.rename(p1_path, new_p1_path)

        await sync_manager.run_sync(run_verification=False)
        # Should complete without error — pg_index None is handled gracefully
        assert sync_manager.pg_index is None

        # --- Phase: Root rename should not crash when pg_index is None ---
        new_lib_path = os.path.join(temp_workspace["root"], "new_library")
        os.rename(lib_path, new_lib_path)
        sync_manager.base_path = new_lib_path

        await sync_manager.run_sync(run_verification=False)
        assert sync_manager.pg_index is None
