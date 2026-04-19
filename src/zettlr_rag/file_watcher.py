import time
import os
import sys
import chromadb
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from llama_index.core import StorageContext, VectorStoreIndex, Settings, SimpleDirectoryReader
from zettlr_rag.rag_setup import setup_settings

class NewPaperHandler(FileSystemEventHandler):
    def __init__(self, index):
        self.index = index

    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith(".md"):
            print(f"🚀 New research detected: {event.src_path}")
            self.process_new_file(event.src_path)

    def process_new_file(self, file_path):
        # 1. Load the specific new file
        documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
        
        # 2. Parse using MarkdownElementNodeParser from Settings
        nodes = Settings.node_parser.get_nodes_from_documents(documents)
        base_nodes, objects = Settings.node_parser.get_nodes_and_objects(nodes)
        
        # 3. Insert into the existing persistent index
        self.index.insert_nodes(base_nodes + objects)
        print(f"✅ Successfully indexed {os.path.basename(file_path)}")

def start_monitor(path_to_watch: str) -> None:
    # This will now initialize GoogleGenAI with gemini-3-flash-preview
    setup_settings()
    
    # Initialize persistent storage
    db = chromadb.PersistentClient(path="./chroma_db_academic")
    chroma_collection = db.get_or_create_collection("research_papers")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    # Load the index
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex([], storage_context=storage_context) 
    
    event_handler = NewPaperHandler(index)
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
