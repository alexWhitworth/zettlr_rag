import os
import sys
import time
from typing import Any

import chromadb
from llama_index.core import Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.indices.base import BaseIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from zettlr_rag.rag_setup import process_documents_metadata, setup_settings


class NewPaperHandler(FileSystemEventHandler):
    def __init__(
        self,
        index: BaseIndex[Any],
        metadata_path: str,
        base_path: str,
        node_parser: Any | None = None,
    ) -> None:
        self.index = index
        self.metadata_path = metadata_path
        self.base_path = base_path
        self.node_parser = node_parser or Settings.node_parser

    def on_created(self, event: FileSystemEvent) -> None:
        src_path = event.src_path
        if isinstance(src_path, bytes):
            src_path = src_path.decode()
        if not event.is_directory and src_path.endswith(".md"):
            print(f"🚀 New research detected: {src_path}")
            self.process_file(src_path)

    def on_modified(self, event: FileSystemEvent) -> None:
        src_path = event.src_path
        if isinstance(src_path, bytes):
            src_path = src_path.decode()
        if not event.is_directory and src_path.endswith(".md"):
            print(f"📝 Research modified: {src_path}")
            self.process_file(src_path)

    def process_file(self, file_path: str) -> None:
        # 1. Load the specific file
        reader = SimpleDirectoryReader(input_files=[file_path])
        documents = reader.load_data()

        # 2. Extract YAML and fallback metadata
        documents = process_documents_metadata(documents, self.base_path)

        # 3. Refresh the index for these specific documents
        refreshed_docs = self.index.refresh_ref_docs(documents)

        if any(refreshed_docs):
            print(f"✅ Successfully updated index for {os.path.basename(file_path)}")
            # 4. Persist metadata
            self.index.storage_context.persist(persist_dir=self.metadata_path)
            print(f"💾 Metadata persisted to {self.metadata_path}")
        else:
            print(f"info: No changes detected in {os.path.basename(file_path)}")


def start_monitor(
    path_to_watch: str,
    chroma_path: str = "./chroma_db_academic",
    metadata_path: str = "./.index_metadata",
) -> None:
    # This will now initialize GoogleGenAI with gemini-3-flash-preview
    setup_settings()

    # Initialize persistent storage
    db = chromadb.PersistentClient(path=chroma_path)
    chroma_collection = db.get_or_create_collection("research_papers")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # Load the index or initialize new one
    from llama_index.core import load_index_from_storage

    if os.path.exists(metadata_path) and os.listdir(metadata_path):
        print(f"Loading existing index metadata from {metadata_path}...")
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store, persist_dir=metadata_path
        )
        index = load_index_from_storage(storage_context)
    else:
        print("Initializing new index metadata...")
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex([], storage_context=storage_context)

    event_handler = NewPaperHandler(index, metadata_path, path_to_watch)
    observer = Observer()
    observer.schedule(event_handler, path_to_watch, recursive=True)

    print(f"👀 Monitoring {path_to_watch} for new research...")
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


def main() -> None:
    # Default to the full library root
    default_path = (
        "/Users/awhitworth/Library/CloudStorage/"
        "ProtonDrive-whitworth.alex@protonmail.com-folder/Zettlr-Papers"
    )
    path = default_path
    if len(sys.argv) > 1:
        path = sys.argv[1]

    if not os.path.exists(path):
        print(f"❌ Error: Path does not exist: {path}")
        sys.exit(1)

    start_monitor(path)


if __name__ == "__main__":
    main()
