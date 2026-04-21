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

class RateLimitedEmbedding(GoogleGenAIEmbedding):
    sleep_seconds: float = 1.0  # must declare as class field for pydantic

    def __init__(self, sleep_seconds: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.sleep_seconds = sleep_seconds

    def _get_text_embedding(self, text: str) -> List[float]:
        time.sleep(self.sleep_seconds)
        return super()._get_text_embedding(text)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        logger.info(f"Embedding {len(texts)} texts (sync) via manual loop")
        embeddings = []
        for text in texts:
            embeddings.append(self._get_text_embedding(text))
        return embeddings

    async def _aget_text_embedding(self, text: str) -> List[float]:
        await asyncio.sleep(self.sleep_seconds)
        return await super()._aget_text_embedding(text)

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        logger.info(f"Embedding {len(texts)} texts (async) via gather")
        # To strictly respect sleep_seconds per request, we should probably do them sequentially
        # or stagger them. For simplicity and correctness, let's do sequential first.
        embeddings = []
        for text in texts:
            embeddings.append(await self._aget_text_embedding(text))
        return embeddings


def setup_settings() -> None:
    """Initialize global LlamaIndex settings."""
    load_dotenv()
    if not os.getenv("GEMINI_API_KEY"):
        raise ValueError("API Key not found. Check your .env file.")
    
    # Ensure GOOGLE_API_KEY is also set as some underlying SDKs expect it
    if not os.getenv("GOOGLE_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

    Settings.llm = GoogleGenAI(
        model="models/gemini-3-flash-preview", api_key=os.getenv("GEMINI_API_KEY")
    )

    Settings.embed_model = RateLimitedEmbedding(
        model_name="models/gemini-embedding-2-preview",
        api_key=os.getenv("GEMINI_API_KEY"),
        embed_batch_size=10,
        retry_min_seconds=10,
        retries=10,
        sleep_seconds=1.0,  # Add a delay between embedding calls to respect rate limits
    )

    Settings.system_prompt = (
        "I am a Senior Staff Data Scientist, Algorithms. When I ask technical or research questions, "
        "provide high-level scientific detail and include paper citations (bibtex format). "
        "Use clean Markdown formatting with clear headers, bold text for key terms, and LaTeX for math. "
        "Prefer Python for all code examples. Assume a high level of statistical and algorithmic understanding. "
        "Provide sufficient detail to produce complete answers, but prefer brevity to unnecessarily verbose responses. "
        "Do not include conversational filler—start directly with the content."
    )

    Settings.node_parser = MarkdownNodeParser()


def load_academic_markdown(directory: str) -> list:
    """Load MD Files while preserving YAML Metadata."""
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")
    reader = SimpleDirectoryReader(input_dir=directory, recursive=True)
    documents = reader.load_data()
    return documents


async def main_async(
    base_path: str = "/Users/awhitworth/Library/CloudStorage/ProtonDrive-whitworth.alex@protonmail.com-folder/Zettlr-Papers",
    chroma_path: str = "./chroma_db_academic",
    metadata_path: str = "./.index_metadata",
    run_verification: bool = True
) -> None:
    setup_settings()

    print(f"📂 Scanning library: {base_path}")
    documents = load_academic_markdown(base_path)

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

    # 3. Smart Sync: Refresh Index
    print("🔄 Smart Sync: Detecting changes (additions, edits, deletions)...")
    refreshed_docs = index.refresh_ref_docs(documents, show_progress=True)

    # Log results of the sync
    new_count = sum(refreshed_docs)
    print(f"✅ Sync complete. Updated/Added {new_count} documents.")

    # 4. Persist Metadata (crucial for detecting changes next time)
    index.storage_context.persist(persist_dir=metadata_path)
    print(f"💾 Metadata persisted to {metadata_path}")

    # 5. Final Verification Query
    if run_verification:
        print("\n📝 Running verification query...")
        query_engine = index.as_query_engine(similarity_top_k=20)
        response = query_engine.query(
            "Summarize the shrinkage can be used to improve experiment estimates and their precision."
        )
        print(f"\n# Query Response\n{response}")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
