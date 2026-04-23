import asyncio
import logging
import os
import re

import chromadb
import frontmatter
from dotenv import load_dotenv
from llama_index.core import (
    Document,
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

    Settings.embed_model = embed_model
    Settings.node_parser = MarkdownNodeParser()


def process_documents_metadata(documents: list[Document], directory: str) -> list[Document]:
    """Extract YAML frontmatter and add fallback metadata to documents."""
    for doc in documents:
        # 1. Parse frontmatter
        post = frontmatter.loads(doc.text)

        # 2. Update doc content and metadata
        doc.set_content(post.content)
        doc.metadata.update(post.metadata)

        # 3. Stable IDs based on file path
        doc.id_ = doc.metadata["file_path"]

        # 4. Fallback for category and year if not in YAML
        if "category" not in doc.metadata or not doc.metadata["category"]:
            rel_path = os.path.relpath(doc.metadata["file_path"], directory)
            path_parts = rel_path.split(os.sep)
            if len(path_parts) > 1:
                doc.metadata["category"] = path_parts[0]
            else:
                doc.metadata["category"] = "Uncategorized"
        if "year" not in doc.metadata or not doc.metadata["year"]:
            # Try to find a 4-digit year in the path or filename
            match = re.search(r"(19|20)\d{2}", doc.metadata["file_name"])
            if match:
                doc.metadata["year"] = int(match.group(0))
            else:
                doc.metadata["year"] = "N/A"

    return documents


def load_academic_markdown(directory: str) -> list[Document]:
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
    return process_documents_metadata(documents, directory)



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

    # 2. Load or Initialize Storage Context
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

    # 3. Smart Sync: Manual embedding + direct vector store insertion
    # (Bypasses refresh_ref_docs which has a KeyError bug)
    print(f"🔄 Manual Sync: Processing {len(documents)} docs...")

    # Get existing doc IDs and hashes from docstore
    existing_hashes = {}
    for doc_id in list(index.docstore.docs.keys()):
        existing_hashes[doc_id] = index.docstore.get_document_hash(doc_id)

    # Filter to only new or changed docs
    docs_to_process = []
    docs_to_delete = []
    for doc in documents:
        current_hash = doc.hash
        existing_hash = existing_hashes.get(doc.id_)
        if existing_hash is None:
            docs_to_process.append(doc)  # new
        elif existing_hash != current_hash:
            docs_to_delete.append(doc.id_)  # changed — delete then re-add
            docs_to_process.append(doc)
        # else: unchanged, skip

    print(f"📊 Sync plan: {len(docs_to_process)} to embed, {len(docs_to_delete)} to delete first, {len(documents) - len(docs_to_process)} unchanged")

    # Delete changed docs first
    for doc_id in docs_to_delete:
        try:
            index.delete_ref_doc(doc_id, delete_from_docstore=True)
        except Exception as e:
            logger.warning(f"Could not delete {doc_id}: {e}")

    # Parse docs into nodes
    parser = Settings.node_parser
    total_processed = 0
    total_batches = (len(docs_to_process) + checkpoint_batch_size - 1) // checkpoint_batch_size

    try:
        for i in range(0, len(docs_to_process), checkpoint_batch_size):
            batch_docs = docs_to_process[i : i + checkpoint_batch_size]
            batch_num = i // checkpoint_batch_size + 1

            print(f"\n📦 Batch {batch_num}/{total_batches} ({len(batch_docs)} docs)...")

            # Parse batch into nodes
            nodes = parser.get_nodes_from_documents(batch_docs)
            print(f"   Parsed into {len(nodes)} nodes")

            # Filter out tiny nodes (< 20 chars)
            nodes = [n for n in nodes if len(n.get_content().strip()) >= 20]
            print(f"   {len(nodes)} nodes after filtering tiny ones")

            if not nodes:
                continue

            # Embed in sub-batches of 10 [not working]
            # switch to: Embed one at a time (more reliable than batch, avoids silent drops)
            print(f"   Embedding {len(nodes)} nodes individually...")
            embedded_count = 0
            failed_nodes = []
            for n_idx, node in enumerate(nodes):
                text = node.get_content(metadata_mode="embed")
                try:
                    node.embedding = Settings.embed_model.get_text_embedding(text)
                    embedded_count += 1
                except Exception as e:
                    logger.warning(f"   Failed to embed node {n_idx} ({node.metadata.get('file_name', 'unknown')}): {e}")
                    failed_nodes.append(node)

                if (n_idx + 1) % 25 == 0:
                    print(f"   Embedded {n_idx + 1}/{len(nodes)}")

            print(f"   Embedded {embedded_count}/{len(nodes)} nodes ({len(failed_nodes)} failed)")

            # Filter out failed nodes
            nodes = [n for n in nodes if n.embedding is not None]

            # Verify all nodes have embeddings before adding
            missing = [n for n in nodes if n.embedding is None]
            if missing:
                logger.warning(f"   {len(missing)} nodes missing embeddings — filtering out")
                nodes = [n for n in nodes if n.embedding is not None]

            if not nodes:
                print(f"   No valid nodes to add in batch {batch_num}")
                continue

            # Add to vector store
            vector_store.add(nodes)

            # Add to docstore (for hash tracking)
            for doc in batch_docs:
                index.docstore.set_document_hash(doc.id_, doc.hash)
                index.docstore.add_documents([doc], allow_update=True)

            total_processed += len(batch_docs)

            # Persist checkpoint
            index.storage_context.persist(persist_dir=metadata_path)
            print(f"💾 Checkpoint: {total_processed}/{len(docs_to_process)} docs processed")

    except KeyboardInterrupt:
        print("\n⚠️  Interrupted — persisting final state before exit...")
        index.storage_context.persist(persist_dir=metadata_path)
        print(f"💾 State saved. Processed {total_processed} docs before interrupt.")
        raise
    except Exception as e:
        print(f"\n❌ Error during sync: {e}")
        import traceback
        traceback.print_exc()
        index.storage_context.persist(persist_dir=metadata_path)
        print(f"💾 State saved. Processed {total_processed} docs before error.")
        raise

    print(f"\n✅ Sync complete. Processed {total_processed} documents total.")
    print(f"💾 Final metadata persisted to {metadata_path}")

    # 4. Verification Query
    if run_verification:
        print("\n📝 Running verification query...")
        query_engine = index.as_query_engine(
            similarity_top_k=20,
            system_prompt=SYSTEM_PROMPT,
        )
        # Use async query to avoid nested asyncio.run() conflict
        response = await query_engine.aquery(
            "Summarize the shrinkage can be used to improve experiment estimates and their precision."
        )
        print(f"\n# Query Response\n{response}")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
