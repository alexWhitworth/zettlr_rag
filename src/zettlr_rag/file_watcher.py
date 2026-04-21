import os
import sys
import time

import chromadb
from llama_index.core import Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from zettlr_rag.rag_setup import setup_settings
from llama_index.vector_stores.chroma import ChromaVectorStore


class NewPaperHandler(FileSystemEventHandler):
    def __init__(self, index, metadata_path, node_parser=None):
        self.index = index
        self.metadata_path = metadata_path
        self.node_parser = node_parser or Settings.node_parser

    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith(".md"):
            print(f"🚀 New research detected: {event.src_path}")
            self.process_file(event.src_path)

    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith(".md"):
            print(f"📝 Research modified: {event.src_path}")
            self.process_file(event.src_path)

    def process_file(self, file_path):
        # 1. Load the specific file
        reader = SimpleDirectoryReader(input_files=[file_path])
        documents = reader.load_data()

        # 2. Set stable doc ID based on file path (consistent with rag_setup.py)
        for doc in documents:
            doc.doc_id = doc.metadata["file_path"]

        # 3. Refresh the index for these specific documents
        refreshed_docs = self.index.refresh_ref_docs(documents)
        
        if any(refreshed_docs):
            print(f"✅ Successfully updated index for {os.path.basename(file_path)}")
            # 4. Persist metadata
            self.index.storage_context.persist(persist_dir=self.metadata_path)
            print(f"💾 Metadata persisted to {self.metadata_path}")
        else:
            print(f"ℹ️ No changes detected in {os.path.basename(file_path)}")


def start_monitor(
    path_to_watch: str, 
    chroma_path: str = "./chroma_db_academic",
    metadata_path: str = "./.index_metadata"
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

    event_handler = NewPaperHandler(index, metadata_path)
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
    path = "/Users/awhitworth/Library/CloudStorage/ProtonDrive-whitworth.alex@protonmail.com-folder/Zettlr-Papers"
    if len(sys.argv) > 1:
        path = sys.argv[1]

    if not os.path.exists(path):
        print(f"❌ Error: Path does not exist: {path}")
        sys.exit(1)

    start_monitor(path)


if __name__ == "__main__":
    main()
