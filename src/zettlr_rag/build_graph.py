import json
import logging
import os
from typing import cast

import chromadb
from llama_index.core import (
    PropertyGraphIndex,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor
from llama_index.core.schema import BaseNode
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.vector_stores.chroma import ChromaVectorStore

from zettlr_rag.consts import (
    BUILD_GRAPH_MODEL,
    CHROMA_PATH,
    GRAPH_ENTITIES,
    GRAPH_INDEX_PATH,
    GRAPH_RELATIONS,
    METADATA_PATH,
)
from zettlr_rag.rag_setup import setup_settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BATCH_SIZE = 50
CHECKPOINT_FILE = "build_graph_checkpoint.json"


def _load_checkpoint(graph_path: str) -> set[str]:
    """Return set of already-processed node IDs from checkpoint file.

    Arguments:
        graph_path: Directory where graph index and checkpoint are stored.

    Returns:
        Set of node ID strings that have already been processed.
    """
    path = os.path.join(graph_path, CHECKPOINT_FILE)
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return set(json.load(f))


def _save_checkpoint(graph_path: str, processed_ids: set[str]) -> None:
    """Persist processed node IDs to checkpoint file.

    Arguments:
        graph_path: Directory where graph index and checkpoint are stored.
        processed_ids: Set of node ID strings to persist.

    Returns:
        None
    """
    path = os.path.join(graph_path, CHECKPOINT_FILE)
    with open(path, "w") as f:
        json.dump(list(processed_ids), f)


def _make_kg_extractor(api_key: str) -> SchemaLLMPathExtractor:
    """Construct the KG extractor with project schema.

    Arguments:
        api_key: Gemini API key string.

    Returns:
        Configured SchemaLLMPathExtractor instance.
    """
    return SchemaLLMPathExtractor(
        llm=GoogleGenAI(
            model=f"models/{BUILD_GRAPH_MODEL}",
            api_key=api_key,
        ),
        possible_entities=GRAPH_ENTITIES,
        possible_relations=GRAPH_RELATIONS,
        strict=False,  # strict=True silently drops all output when LLM labels don't match exactly
        num_workers=4,
        max_triplets_per_chunk=10,
    )


def build_graph(
    chroma_path: str = CHROMA_PATH,
    metadata_path: str = METADATA_PATH,
    graph_path: str = GRAPH_INDEX_PATH,
    batch_size: int = BATCH_SIZE,
) -> None:
    """Build or resume a property graph index from existing docstore nodes.

    Processes nodes in batches of `batch_size`, persisting the graph and a
    checkpoint file after each batch. Re-running after a failure resumes from
    the last completed batch rather than starting over.

    This must be run separately from rag_setup.py because PropertyGraphIndex
    calls asyncio.run() internally, which is incompatible with an existing
    async context.

    Arguments:
        chroma_path: Path to the ChromaDB persistent store.
        metadata_path: Path to the LlamaIndex docstore metadata directory.
        graph_path: Directory where the property graph index is persisted.
        batch_size: Number of nodes to process per batch before checkpointing.

    Returns:
        None
    """
    setup_settings()
    api_key = cast(str, os.getenv("GEMINI_API_KEY"))
    os.makedirs(graph_path, exist_ok=True)

    # Load docstore and collect nodes to process
    db = chromadb.PersistentClient(path=chroma_path)
    chroma_collection = db.get_or_create_collection("research_papers")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
        persist_dir=metadata_path,
    )
    source_index = cast(VectorStoreIndex, load_index_from_storage(storage_context))

    all_nodes: list[BaseNode] = [
        n
        for n in source_index.docstore.docs.values()
        if hasattr(n, "embedding")
        and n.embedding is not None
        and not (getattr(n, "text", "") or "").lstrip().startswith("#")
    ]

    # Resume: skip already-processed nodes
    processed_ids = _load_checkpoint(graph_path)
    remaining = [n for n in all_nodes if n.node_id not in processed_ids]

    logger.info(
        f"Total nodes: {len(all_nodes)} | Already processed: {len(processed_ids)} | "
        f"Remaining: {len(remaining)}"
    )

    if not remaining:
        logger.info("✅ All nodes already processed — nothing to do.")
        return

    # Load existing graph if resuming, otherwise create empty one
    graph_index_store = os.path.join(graph_path, "index_store.json")
    if os.path.exists(graph_index_store):
        logger.info("Resuming from existing graph index...")
        pg_storage = StorageContext.from_defaults(persist_dir=graph_path)
        pg_index = cast(PropertyGraphIndex, load_index_from_storage(pg_storage))
        # Attach extractor for incremental inserts
        pg_index._kg_extractors = [_make_kg_extractor(api_key)]
    else:
        logger.info("Creating new graph index...")
        pg_index = PropertyGraphIndex(
            nodes=[],
            kg_extractors=[_make_kg_extractor(api_key)],
            show_progress=False,
        )
        pg_index.storage_context.persist(persist_dir=graph_path)

    # Process in batches
    total_batches = (len(remaining) + batch_size - 1) // batch_size
    for i in range(0, len(remaining), batch_size):
        batch = remaining[i : i + batch_size]
        batch_num = i // batch_size + 1
        logger.info(f"Batch {batch_num}/{total_batches} — {len(batch)} nodes...")

        pg_index.insert_nodes(batch)
        pg_index.storage_context.persist(persist_dir=graph_path)

        processed_ids.update(n.node_id for n in batch)
        _save_checkpoint(graph_path, processed_ids)
        relation_count = len(pg_index.property_graph_store.graph.relations)
        logger.info(
            f"  ✅ Batch {batch_num} done. Processed: {len(processed_ids)} | "
            f"Graph relations so far: {relation_count}"
        )

    logger.info(f"✅ Graph index complete. Persisted to {graph_path}")


if __name__ == "__main__":
    import sys

    build_graph(
        chroma_path=sys.argv[1] if len(sys.argv) > 1 else CHROMA_PATH,
        metadata_path=sys.argv[2] if len(sys.argv) > 2 else METADATA_PATH,
        graph_path=sys.argv[3] if len(sys.argv) > 3 else GRAPH_INDEX_PATH,
    )
