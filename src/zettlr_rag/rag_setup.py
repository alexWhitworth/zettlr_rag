import asyncio
import logging
import os
import re
import threading
from collections.abc import Sequence
from typing import Any, cast

import chromadb
import frontmatter  # type: ignore
from chromadb.api.models.Collection import Collection
from dotenv import load_dotenv
from llama_index.core import (
    Document,
    PropertyGraphIndex,
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.base.llms.types import ChatMessage, ChatResponse, CompletionResponse
from llama_index.core.graph_stores import SimplePropertyGraphStore
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core.schema import (
    MetadataMode,
    NodeRelationship,
    RelatedNodeInfo,
)
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.vector_stores.chroma import ChromaVectorStore

from zettlr_rag.consts import (
    BIBTEX_PATTERN,
    CHROMA_PATH,
    EMBEDDING_MODEL_NAME,
    GRAPH_INDEX_PATH,
    METADATA_PATH,
    MODEL_NAME,
    SYSTEM_PROMPT,
)
from zettlr_rag.metrics import TokenUsage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


_usage_store = threading.local()


def reset_token_usage() -> None:
    """Resets the accumulated token usage for the current thread."""
    _usage_store.last_usage = TokenUsage()


def get_last_token_usage() -> TokenUsage:
    """Returns token usage captured from the most recent Gemini calls."""
    return cast(TokenUsage, getattr(_usage_store, "last_usage", TokenUsage()))


class TokenCapturingGemini(GoogleGenAI):
    """
    Wraps GoogleGenAI to capture and accumulate usage_metadata.

    This custom wrapper is required due to a bug in llama_index
    (https://github.com/run-llama/llama_index/issues/19293) where token usage metadata is
    discarded during the query engine pipeline.

    This class intercepts the response and accumulates tokens across multiple calls (e.g.,
    LLMRerank + Synthesis) into thread-local storage before llama_index drops the data.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        response = cast(CompletionResponse, super().complete(prompt, formatted=formatted, **kwargs))
        self._store(response)
        return response

    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        response = cast(ChatResponse, super().chat(messages, **kwargs))
        self._store(response)
        return response

    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        response = cast(
            CompletionResponse, await super().acomplete(prompt, formatted=formatted, **kwargs)
        )
        self._store(response)
        return response

    async def achat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        response = cast(ChatResponse, await super().achat(messages, **kwargs))
        self._store(response)
        return response

    def _store(self, response: Any) -> None:
        try:
            # Initialize if not exists
            if not hasattr(_usage_store, "last_usage"):
                _usage_store.last_usage = TokenUsage()

            current_usage = _usage_store.last_usage

            # Try raw attribute first
            raw = getattr(response, "raw", None)
            if raw:
                meta = getattr(raw, "usage_metadata", None)
                if meta:
                    current_usage.input_tokens += getattr(meta, "prompt_token_count", 0) or 0
                    current_usage.output_tokens += getattr(meta, "candidates_token_count", 0) or 0
                    current_usage.cache_tokens += (
                        getattr(meta, "cached_content_token_count", 0) or 0
                    )
                    return

            # Try additional_kwargs
            extra = getattr(response, "additional_kwargs", {}) or {}

            # Map based on observed keys: 'prompt_tokens', 'completion_tokens', 'total_tokens'
            input_tokens = extra.get("prompt_tokens", 0)
            output_tokens = extra.get("completion_tokens", 0)

            if input_tokens or output_tokens:
                current_usage.input_tokens += input_tokens or 0
                current_usage.output_tokens += output_tokens or 0

        except Exception:
            pass  # Falls through — not fatal


def sanitize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """
    Flatten metadata for ChromaDB compatibility.
    ChromaDB only accepts str, int, float, or None as metadata values.
    """
    sanitized: dict[str, Any] = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float)) or value is None:
            sanitized[key] = value
        elif isinstance(value, list):
            sanitized[key] = ", ".join(str(v) for v in value)
        elif isinstance(value, dict):
            import json

            sanitized[key] = json.dumps(value)
        else:
            sanitized[key] = str(value)
    return sanitized


def setup_settings() -> None:
    """Initialize global LlamaIndex settings."""
    load_dotenv()
    if not os.getenv("GEMINI_API_KEY"):
        raise ValueError("API Key not found. Check your .env file.")

    if not os.getenv("GOOGLE_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = cast(str, os.getenv("GEMINI_API_KEY"))

    # Use wrapper instead of GoogleGenAI directly
    Settings.llm = TokenCapturingGemini(
        model=f"models/{MODEL_NAME}", api_key=cast(str, os.getenv("GEMINI_API_KEY"))
    )

    # Use the stock GoogleGenAIEmbedding — its built-in retry handles 429s
    embed_model = GoogleGenAIEmbedding(
        model_name=f"models/{EMBEDDING_MODEL_NAME}",
        api_key=cast(str, os.getenv("GEMINI_API_KEY")),
        embed_batch_size=10,
        retry_min_seconds=10,
        retries=15,
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
        doc.id_ = cast(str, doc.metadata["file_path"])

        # 4. Fallback for category and year if not in YAML
        if "category" not in doc.metadata or not doc.metadata["category"]:
            rel_path = os.path.relpath(cast(str, doc.metadata["file_path"]), directory)
            path_parts = rel_path.split(os.sep)
            doc.metadata["category"] = path_parts[0] if len(path_parts) > 1 else "Uncategorized"
        if "year" not in doc.metadata or not doc.metadata["year"]:
            # Try to find a 4-digit year in the path or filename
            match = re.search(r"(19|20)\d{2}", cast(str, doc.metadata["file_name"]))
            doc.metadata["year"] = int(match.group(0)) if match else "N/A"

        # 5. Sanitize all metadata for ChromaDB compatibility
        doc.metadata = sanitize_metadata(doc.metadata)

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


class AcademicRAGSync:
    """
    Manages the synchronization of local academic markdown files with a Vector Index.

    This class provides "Smart Sync" logic to efficiently maintain a ChromaDB
    vector store alongside a LlamaIndex document store. By analyzing file hashes
    and existing index metadata, it identifies new, modified, moved, or deleted
    files on disk. This prevents redundant embedding API calls and ensures that
    the vector store accurately reflects the current state of the markdown library.

    The synchronization pipeline handles embedding extraction, batching, metadata
    sanitization, and persistent storage of the updated index state.

    Methods:
        initialize: Sets up the vector store, storage context, and index.
        plan_sync: Compares disk files against the index to categorize needed changes.
        execute_moves: Updates metadata and relationships for files that were relocated.
        execute_deletions: Removes stale documents from both the docstore and vector store.
        index_documents: Batches, embeds, and indexes new or modified documents.
        run_sync: Orchestrates the complete plan, execute, and verify synchronization cycle.
        verify: Runs a sample test query against the updated index.
    """

    def __init__(
        self,
        base_path: str,
        chroma_path: str = CHROMA_PATH,
        metadata_path: str = METADATA_PATH,
        graph_path: str = GRAPH_INDEX_PATH,
        checkpoint_batch_size: int = 50,
    ):
        self.base_path = base_path
        self.chroma_path = chroma_path
        self.metadata_path = metadata_path
        self.graph_path = graph_path
        self.checkpoint_batch_size = checkpoint_batch_size

        self.index: VectorStoreIndex | None = None
        self.vector_store: ChromaVectorStore | None = None
        self.chroma_collection: Collection | None = None
        self.pg_index: PropertyGraphIndex | None = None

    def initialize(self) -> None:
        """Initialize settings, vector store, and index."""
        setup_settings()

        # 1. Initialize Vector Store (Chroma)
        db = chromadb.PersistentClient(path=self.chroma_path)
        self.chroma_collection = db.get_or_create_collection("research_papers")
        self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)

        # 2. Load or Initialize Storage Context
        if os.path.exists(self.metadata_path) and os.listdir(self.metadata_path):
            logger.info("Loading existing index metadata...")
            storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store, persist_dir=self.metadata_path
            )
            loaded_index = load_index_from_storage(storage_context)
            self.index = cast(VectorStoreIndex, loaded_index)
        else:
            logger.info("Initializing new index metadata...")
            storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
            self.index = VectorStoreIndex([], storage_context=storage_context)

        self._initialize_graph()

    def _initialize_graph(self) -> None:
        """Initialize the Property Graph store and index."""
        if os.path.exists(self.graph_path) and os.listdir(self.graph_path):
            logger.info("Loading existing property graph index...")
            storage_context = StorageContext.from_defaults(persist_dir=self.graph_path)
            self.pg_index = cast(PropertyGraphIndex, load_index_from_storage(storage_context))
        else:
            from typing import Literal

            logger.info("Initializing new property graph index...")
            kg_extractor = SchemaLLMPathExtractor(
                llm=Settings.llm,
                possible_entities=Literal[
                    "Document", "Author", "Method", "Dataset", "Metric", "Concept", "Organization"
                ],
                possible_relations=Literal[
                    "AUTHORED_BY",
                    "USES_METHOD",
                    "BENCHMARKED_ON",
                    "IMPROVES_UPON",
                    "DEFINES",
                    "REPORTS",
                    "AFFILIATED_WITH",
                ],
                strict=True,
            )

            # Link it to the existing docstore so nodes map correctly
            if self.index and self.index.storage_context:
                storage_context = StorageContext.from_defaults(
                    docstore=self.index.storage_context.docstore
                )
            else:
                storage_context = StorageContext.from_defaults()

            self.pg_index = PropertyGraphIndex(
                nodes=[],
                kg_extractors=[kg_extractor],
                storage_context=storage_context,
            )
            os.makedirs(self.graph_path, exist_ok=True)
            self.pg_index.storage_context.persist(persist_dir=self.graph_path)

    def plan_sync(self, documents: list[Document]) -> dict[str, Any]:
        """Analyze disk documents vs index to create a sync plan."""
        if self.index is None:
            raise RuntimeError("Index not initialized. Call initialize() first.")

        # Get existing doc IDs and hashes from docstore
        ref_doc_info = self.index.docstore.get_all_ref_doc_info() or {}
        existing_doc_ids = list(ref_doc_info.keys())
        existing_hashes = {
            doc_id: self.index.docstore.get_document_hash(doc_id) for doc_id in existing_doc_ids
        }

        # Determine if we need to backfill the graph
        is_graph_empty = False
        if self.pg_index and self.pg_index.property_graph_store:
            graph_nodes = getattr(self.pg_index.property_graph_store, "graph", None)
            if graph_nodes and not graph_nodes.nodes:
                is_graph_empty = True

        # If the docstore has entries but the graph is completely empty,
        # force re-indexing of all existing valid documents to backfill the graph.
        force_reindex = is_graph_empty and len(existing_hashes) > 0

        disk_doc_ids = {doc.id_ for doc in documents}
        stale_doc_ids = [doc_id for doc_id in existing_doc_ids if doc_id not in disk_doc_ids]
        new_docs = [doc for doc in documents if doc.id_ not in existing_hashes]

        if force_reindex:
            logger.info("Graph is empty but docstore is populated. Forcing full graph backfill.")
            # Treat all disk documents that aren't stale as "changed" to force re-extraction
            changed_docs = [doc for doc in documents if doc.id_ in existing_hashes]
        else:
            changed_docs = [
                doc
                for doc in documents
                if doc.id_ in existing_hashes and existing_hashes[doc.id_] != doc.hash
            ]

        # Move Detection
        docs_to_move: list[tuple[str, Document]] = []
        truly_new_docs = []

        if stale_doc_ids and new_docs and not force_reindex:
            logger.info(f"🔍 Checking {len(stale_doc_ids)} stale entries for potential moves...")
            stale_text_map = {}
            for s_id in stale_doc_ids:
                try:
                    stale_doc = self.index.docstore.get_document(s_id)
                    if stale_doc and stale_doc.get_content():
                        stale_text_map[stale_doc.get_content()] = s_id
                except Exception:
                    continue

            for n_doc in new_docs:
                n_text = n_doc.get_content()
                if n_text in stale_text_map:
                    s_id = stale_text_map[n_text]
                    docs_to_move.append((s_id, n_doc))
                    del stale_text_map[n_text]
                else:
                    truly_new_docs.append(n_doc)
        else:
            truly_new_docs = new_docs

        moved_stale_ids = {s_id for s_id, _ in docs_to_move}
        truly_stale_ids = [s_id for s_id in stale_doc_ids if s_id not in moved_stale_ids]

        return {
            "new": truly_new_docs,
            "changed": changed_docs,
            "moved": docs_to_move,
            "stale": truly_stale_ids,
            "unchanged_count": len(documents)
            - len(truly_new_docs)
            - len(changed_docs)
            - len(docs_to_move),
        }

    def execute_moves(self, docs_to_move: list[tuple[str, Document]]) -> list[Document]:
        """Perform metadata-only updates for moved files."""
        if self.index is None or self.chroma_collection is None:
            raise RuntimeError("Index or collection not initialized.")

        failed_moves = []
        for s_id, n_doc in docs_to_move:
            print(f"🚚 Moving: {os.path.basename(s_id)} -> {n_doc.metadata['file_path']}")
            try:
                ref_doc_info = self.index.docstore.get_ref_doc_info(s_id)
                if not ref_doc_info:
                    logger.warning(f"No ref_doc_info for {s_id}, skipping move.")
                    failed_moves.append(n_doc)
                    continue

                node_ids = ref_doc_info.node_ids

                # Update nodes in docstore and Chroma
                nodes = self.index.docstore.get_nodes(node_ids)

                for node in nodes:
                    # Update relationships and metadata
                    node.relationships[NodeRelationship.PARENT] = RelatedNodeInfo(node_id=n_doc.id_)
                    node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id=n_doc.id_)
                    node.metadata = {**node.metadata, **n_doc.metadata}

                    # Update metadata in Chroma
                    res = self.chroma_collection.get(ids=[node.node_id])
                    if res and res["metadatas"]:
                        meta = dict(res["metadatas"][0])
                        meta.update(n_doc.metadata)
                        self.chroma_collection.update(ids=[node.node_id], metadatas=[meta])

                    # Update metadata in Property Graph
                    if self.pg_index and self.pg_index.property_graph_store:
                        # Graph nodes share the same node_id as docstore nodes
                        graph_store = cast(
                            SimplePropertyGraphStore, self.pg_index.property_graph_store
                        )
                        if hasattr(graph_store, "get"):
                            # LlamaIndex SimplePropertyGraphStore often exposes a get method or
                            # internal dict
                            try:
                                # We try to find nodes that correspond to this chunk
                                # Often the text node is stored directly
                                if node.node_id in graph_store.graph.nodes:
                                    g_node = graph_store.graph.nodes[node.node_id]
                                    g_node.properties.update(n_doc.metadata)
                            except Exception as e:
                                logger.warning(
                                    f"Could not update graph node metadata for {node.node_id}: {e}"
                                )

                # Cleanup and Transfer in Docstore
                try:
                    self.index.docstore.delete_ref_doc(s_id)
                except Exception as e:
                    logger.warning(f"Could not delete ref_doc_info {s_id} from docstore: {e}")
                try:
                    self.index.docstore.delete_document(s_id)
                except Exception as e:
                    logger.warning(f"Could not delete document {s_id} from docstore: {e}")

                # Finalize new doc in docstore
                self.index.docstore.add_documents(nodes, allow_update=True)
                self.index.docstore.set_document_hash(n_doc.id_, n_doc.hash)
                self.index.docstore.add_documents([n_doc], allow_update=True)

            except Exception as e:
                logger.error(f"Failed to move {s_id}: {e}")
                failed_moves.append(n_doc)

        self.index.storage_context.persist(persist_dir=self.metadata_path)
        if self.pg_index:
            self.pg_index.storage_context.persist(persist_dir=self.graph_path)

        return failed_moves

    def execute_deletions(self, doc_ids: list[str], is_stale: bool = True) -> None:
        """Delete documents from index and docstore."""
        if self.index is None:
            raise RuntimeError("Index not initialized.")

        for doc_id in doc_ids:
            try:
                self.index.delete_ref_doc(doc_id, delete_from_docstore=True)

                if self.pg_index:
                    try:
                        self.pg_index.delete_ref_doc(doc_id)
                    except Exception as pg_e:
                        logger.warning(f"Could not delete {doc_id} from property graph: {pg_e}")

                if is_stale:
                    logger.info(f"🗑️ Pruned stale document: {doc_id}")
            except Exception as e:
                logger.warning(f"Could not delete {doc_id}: {e}")

        if self.pg_index:
            self.pg_index.storage_context.persist(persist_dir=self.graph_path)

    async def index_documents(self, documents: list[Document]) -> int:
        """Batch process and index documents."""
        if self.index is None or self.vector_store is None:
            raise RuntimeError("Index or vector store not initialized.")
        if not documents:
            return 0

        parser = Settings.node_parser
        total_processed = 0
        total_docs = len(documents)
        batch_size = self.checkpoint_batch_size
        total_batches = (total_docs + batch_size - 1) // batch_size

        for i in range(0, total_docs, batch_size):
            batch_docs = documents[i : i + batch_size]
            batch_num = i // self.checkpoint_batch_size + 1

            print(f"\n📦 Batch {batch_num}/{total_batches} ({len(batch_docs)} docs)...")

            nodes = parser.get_nodes_from_documents(batch_docs)
            nodes = [
                n
                for n in nodes
                # Filter short nodes and raw BibTeX blocks
                if len(n.get_content().strip()) >= 20 and not BIBTEX_PATTERN.search(n.get_content())
            ]

            if not nodes:
                continue

            print(f"   Embedding {len(nodes)} nodes individually...")
            embedded_count = 0
            for node in nodes:
                text = node.get_content(metadata_mode=MetadataMode.EMBED)
                try:
                    node.embedding = Settings.embed_model.get_text_embedding(text)
                    embedded_count += 1
                except Exception as e:
                    logger.warning(f"   Failed to embed node: {e}")

            nodes = [n for n in nodes if n.embedding is not None]
            if not nodes:
                continue

            # 1. Write vectors to ChromaDB
            self.vector_store.add(nodes)

            # 2. Register source docs first — docstore uses SOURCE relationship to populate
            # ref_doc_info
            for doc in batch_docs:
                self.index.docstore.add_documents([doc], allow_update=True)
                self.index.docstore.set_document_hash(doc.id_, doc.hash)

            # 3. Add nodes — docstore walks SOURCE relationships and wires up ref_doc_info
            self.index.docstore.add_documents(nodes, allow_update=True)

            total_processed += len(batch_docs)
            self.index.storage_context.persist(persist_dir=self.metadata_path)
            logger.info(f"💾 Checkpoint: {total_processed}/{total_docs} docs processed")

        return total_processed

    async def run_sync(self, run_verification: bool = True) -> None:
        """Orchestrate the full synchronization process."""
        self.initialize()

        print(f"📂 Scanning library: {self.base_path}")
        documents = load_academic_markdown(self.base_path)
        print(f"Found {len(documents)} documents.")

        plan = self.plan_sync(documents)
        print(
            f"📊 Sync plan:\n"
            f"   - {len(plan['new'])} new files to embed\n"
            f"   - {len(plan['changed'])} modified files to re-embed\n"
            f"   - {len(plan['moved'])} moved files to update metadata\n"
            f"   - {len(plan['stale'])} stale files to prune\n"
            f"   - {plan['unchanged_count']} unchanged"
        )

        # 1. Moves
        failed_moves = self.execute_moves(plan["moved"])

        # 2. Deletions
        self.execute_deletions([doc.id_ for doc in plan["changed"]] + plan["stale"])

        # 3. New/Changed Indexing
        to_index = plan["new"] + plan["changed"] + failed_moves
        await self.index_documents(to_index)

        print("\n✅ Sync complete.")
        print(f"💾 Final metadata persisted to {self.metadata_path}")

        if run_verification:
            await self.verify()

    async def verify(self) -> None:
        """Run a verification query."""
        if self.index is None:
            raise RuntimeError("Index not initialized.")

        print("\n📝 Running verification query...")
        query_engine = self.index.as_query_engine(
            similarity_top_k=20,
            system_prompt=SYSTEM_PROMPT,
        )
        query_text = "Summarize how shrinkage can be used to improve experiment estimates and their precision."
        response = await query_engine.aquery(query_text)
        print(f"\n# Query Response\n{response}")


async def main_async(
    base_path: str = (
        "/Users/awhitworth/Library/CloudStorage/"
        "ProtonDrive-whitworth.alex@protonmail.com-folder/Zettlr-Papers"
    ),
    chroma_path: str = "./chroma_db_academic",
    metadata_path: str = "./.index_metadata",
    checkpoint_batch_size: int = 50,
    run_verification: bool = True,
) -> None:
    sync_manager = AcademicRAGSync(
        base_path=base_path,
        chroma_path=chroma_path,
        metadata_path=metadata_path,
        checkpoint_batch_size=checkpoint_batch_size,
    )
    await sync_manager.run_sync(run_verification=run_verification)


def main() -> None:
    # Default to the full library root
    default_path = (
        "/Users/awhitworth/Library/CloudStorage/"
        "ProtonDrive-whitworth.alex@protonmail.com-folder/Zettlr/Papers"
    )
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else default_path

    if not os.path.exists(path):
        print(f"❌ Error: Path does not exist: {path}")
        sys.exit(1)

    # verification off: process is stable
    asyncio.run(main_async(base_path=path, run_verification=True))


if __name__ == "__main__":
    main()
