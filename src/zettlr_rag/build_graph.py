import logging
import os
from typing import cast

import chromadb
from llama_index.core import (
    PropertyGraphIndex,
    Settings,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.google_genai import GoogleGenAI

from zettlr_rag.consts import (
    CHROMA_PATH,
    GRAPH_ENTITIES,
    GRAPH_INDEX_PATH,
    GRAPH_RELATIONS,
    METADATA_PATH,
    BUILD_GRAPH_MODEL,
)
from zettlr_rag.rag_setup import setup_settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_graph(
    chroma_path: str = CHROMA_PATH,
    metadata_path: str = METADATA_PATH,
    graph_path: str = GRAPH_INDEX_PATH,
) -> None:
    """Build the property graph index from existing docstore nodes.

    This is a one-time sync operation that must be run separately from
    rag_setup.py due to PropertyGraphIndex.__init__ calling asyncio.run()
    internally, which is incompatible with an existing async context.

    Subsequent incremental updates are handled automatically by rag_setup.py.
    """
    setup_settings()

    # Load existing vector index and docstore
    db = chromadb.PersistentClient(path=chroma_path)
    chroma_collection = db.get_or_create_collection("research_papers")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
        persist_dir=metadata_path,
    )
    index = cast(VectorStoreIndex, load_index_from_storage(storage_context))

    # Get existing content nodes only (not parent Document nodes)
    existing_nodes = [
        n
        for n in index.docstore.docs.values()
        if hasattr(n, "embedding") and n.embedding is not None
    ]
    logger.info(f"Building graph from {len(existing_nodes)} nodes...")

    kg_extractor = SchemaLLMPathExtractor(
        llm=GoogleGenAI(
            model=f"models/{BUILD_GRAPH_MODEL}",
            api_key=cast(str, os.getenv("GEMINI_API_KEY")),
        ),
        possible_entities=GRAPH_ENTITIES,
        possible_relations=GRAPH_RELATIONS,
        strict=True,
        num_workers=16,
        max_triplets_per_chunk=10,
    )

    os.makedirs(graph_path, exist_ok=True)
    pg_index = PropertyGraphIndex(
        nodes=existing_nodes,
        kg_extractors=[kg_extractor],
        show_progress=True,
    )
    pg_index.storage_context.persist(persist_dir=graph_path)
    logger.info(f"✅ Graph index persisted to {graph_path}")


if __name__ == "__main__":
    import sys

    build_graph(
        chroma_path=sys.argv[1] if len(sys.argv) > 1 else CHROMA_PATH,
        metadata_path=sys.argv[2] if len(sys.argv) > 2 else METADATA_PATH,
        graph_path=sys.argv[3] if len(sys.argv) > 3 else GRAPH_INDEX_PATH,
    )
