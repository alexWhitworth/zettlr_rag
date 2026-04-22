import asyncio
import os
import time
import logging
from typing import List

import chromadb
from dotenv import load_dotenv
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.vector_stores.chroma import ChromaVectorStore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "I am a Senior Staff Data Scientist, Algorithms. When I ask technical or research questions, "
    "provide high-level scientific detail and include paper citations (bibtex format). "
    "Use clean Markdown formatting with clear headers, bold text for key terms, and LaTeX for math. "
    "Prefer Python for all code examples. Assume a high level of statistical and algorithmic understanding. "
    "Provide sufficient detail to produce complete answers, but prefer brevity to unnecessarily verbose responses. "
    "Do not include conversational filler—start directly with the content."
)


def setup_settings() -> None:
    """Initialize global LlamaIndex settings."""
    load_dotenv()
    if not os.getenv("GEMINI_API_KEY"):
        raise ValueError("API Key not found. Check your .env file.")

    if not os.getenv("GOOGLE_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

    Settings.llm = GoogleGenAI(
        model="models/gemini-3-flash-preview", api_key=os.getenv("GEMINI_API_KEY")
    )

    # Use the stock GoogleGenAIEmbedding — its built-in retry handles 429s
    embed_model = GoogleGenAIEmbedding(
        model_name="models/gemini-embedding-2-preview",
        api_key=os.getenv("GEMINI_API_KEY"),
        embed_batch_size=10,
        retry_min_seconds=10,
        retries=10,
    )

    # Monkey-patch a sleep before each batch call
    original_get_batch = embed_model._get_text_embeddings
    original_aget_batch = embed_model._aget_text_embeddings

    def rate_limited_batch(texts):
        if texts:
            logger.info(f"Embedding batch of {len(texts)}")
            time.sleep(1.0)
        return original_get_batch(texts)

    async def rate_limited_abatch(texts):
        if texts:
            logger.info(f"Embedding batch of {len(texts)} (async)")
            await asyncio.sleep(1.0)
        return await original_aget_batch(texts)

    embed_model._get_text_embeddings = rate_limited_batch
    embed_model._aget_text_embeddings = rate_limited_abatch

    Settings.embed_model = embed_model
    Settings.node_parser = MarkdownNodeParser()


def load_academic_markdown(directory: str) -> list:
    """Load MD Files while preserving YAML Metadata."""
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")
    reader = SimpleDirectoryReader(
        input_dir=directory,
        recursive=True,
        required_exts=[".md"],
        exclude_hidden=True,
    )
    documents = reader.load_data()
    # Stable IDs based on file path (for refresh_ref_docs to detect unchanged docs)
    for doc in documents:
        doc.id_ = doc.metadata["file_path"]  # ← change from doc.doc_id to doc.id_

    return documents



async def main_async(
    base_path: str = "/Users/awhitworth/Library/CloudStorage/ProtonDrive-whitworth.alex@protonmail.com-folder/Zettlr-Papers",
    chroma_path: str = "./chroma_db_academic",
    metadata_path: str = "./.index_metadata",
    checkpoint_batch_size: int = 50,
    run_verification: bool = True,
) -> None:
    setup_settings()

    print(f"📂 Scanning library: {base_path}")
    documents = load_academic_markdown(base_path)
    print(f"Found {len(documents)} documents.")

    # 1. Initialize Vector Store (Chroma)
    db = chromadb.PersistentClient(path=chroma_path)
    chroma_collection = db.get_or_create_collection("research_papers")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # 2. Load or Initialize Storage Context (with DocStore for Hash tracking)
    if os.path.exists(metadata_path) and os.listdir(metadata_path):
        print("Loading existing index metadata...")
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store, persist_dir=metadata_path
        )
        index = load_index_from_storage(storage_context)
    else:
        print("Initializing new index metadata...")
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex([], storage_context=storage_context)

    # 3. Smart Sync: Refresh Index in batches with checkpointing
    print(
        f"🔄 Smart Sync: Processing {len(documents)} docs in batches of "
        f"{checkpoint_batch_size}..."
    )
    total_updated = 0
    total_batches = (len(documents) + checkpoint_batch_size - 1) // checkpoint_batch_size

    try:
        for i in range(0, len(documents), checkpoint_batch_size):
            batch = documents[i : i + checkpoint_batch_size]
            batch_num = i // checkpoint_batch_size + 1
            print(
                f"\n📦 Batch {batch_num}/{total_batches} "
                f"({len(batch)} docs, {i + len(batch)}/{len(documents)} total)..."
            )

            refreshed = index.refresh_ref_docs(batch, show_progress=True)
            batch_updated = sum(refreshed)
            total_updated += batch_updated

            # Checkpoint: persist after every batch
            index.storage_context.persist(persist_dir=metadata_path)
            print(
                f"💾 Checkpoint saved. Batch updated: {batch_updated}, "
                f"total updated: {total_updated}"
            )
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted — persisting final state before exit...")
        index.storage_context.persist(persist_dir=metadata_path)
        print(f"💾 State saved. Updated {total_updated} docs before interrupt.")
        raise
    except Exception as e:
        print(f"\n❌ Error during sync: {e}")
        print("Attempting to persist state before exit...")
        index.storage_context.persist(persist_dir=metadata_path)
        print(f"💾 State saved. Updated {total_updated} docs before error.")
        raise

    print(f"\n✅ Sync complete. Updated/Added {total_updated} documents total.")
    print(f"💾 Final metadata persisted to {metadata_path}")

    # 4. Final Verification Query
    if run_verification:
        print("\n📝 Running verification query...")
        query_engine = index.as_query_engine(
            similarity_top_k=20,
            system_prompt=SYSTEM_PROMPT,
        )
        response = query_engine.query(
            "Summarize the shrinkage can be used to improve experiment estimates and their precision."
        )
        print(f"\n# Query Response\n{response}")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
