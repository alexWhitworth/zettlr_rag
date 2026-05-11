"""
This script checks the number of chunks in the ChromaDB collection, how many are tracked in the
index's ref_doc_info, and identifies any orphaned chunks that exist in ChromaDB but are not
referenced in the index. It also counts how many chunks contain BibTeX entries, which are not
meaningful for retrieval.
"""

import os
from typing import cast

import chromadb
from dotenv import load_dotenv
from llama_index.core import (
    PropertyGraphIndex,
    Settings,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from zettlr_rag.consts import (
    BIBTEX_PATTERN,
    CHROMA_PATH,
    GRAPH_INDEX_PATH,
    METADATA_PATH,
    MODEL_NAME,
)
from zettlr_rag.rag_setup import TokenCapturingGemini

load_dotenv()
if not os.getenv("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = cast(str, os.getenv("GEMINI_API_KEY"))

Settings.embed_model = GoogleGenAIEmbedding(
    model_name="models/gemini-embedding-2-preview",
    api_key=cast(str, os.getenv("GEMINI_API_KEY")),
)

Settings.llm = TokenCapturingGemini(
    model=f"models/{MODEL_NAME}", api_key=cast(str, os.getenv("GEMINI_API_KEY"))
)

db = chromadb.PersistentClient(path=CHROMA_PATH)
col = db.get_collection("research_papers")
vector_store = ChromaVectorStore(chroma_collection=col)
storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=METADATA_PATH)
index = cast(VectorStoreIndex, load_index_from_storage(storage_context))

# Load graph index
try:
    pg_storage_context = StorageContext.from_defaults(persist_dir=GRAPH_INDEX_PATH)
    pg_index = cast(PropertyGraphIndex, load_index_from_storage(pg_storage_context))
    graph_loaded = True
except Exception as e:
    print(f"Warning: Could not load PropertyGraphIndex: {e}")
    pg_index = None
    graph_loaded = False

# Orphan check
ref_doc_info = index.docstore.get_all_ref_doc_info() or {}
tracked_node_ids = set()
for info in ref_doc_info.values():
    tracked_node_ids.update(info.node_ids)

chroma_ids = set(col.get()["ids"])
orphaned = chroma_ids - tracked_node_ids

# Bibtex chunk count
results = col.get(include=["documents"])
bibtex_chunks = [
    id_
    for id_, doc in zip(results["ids"], results["documents"], strict=True)
    if doc and BIBTEX_PATTERN.search(doc)
]

print(f"Chunks in ChromaDB:              {len(chroma_ids)}")
print(f"Chunks tracked in ref_doc_info:  {len(tracked_node_ids)}")
print(f"Orphaned (untracked) chunks:     {len(orphaned)}")
print(f"Docs tracked in ref_doc_info:    {len(ref_doc_info)}")
print(f"Bibtex chunks:                   {len(bibtex_chunks)}")
print(f"Meaningful chunks:               {len(chroma_ids) - len(bibtex_chunks)}")

if graph_loaded and pg_index and pg_index.property_graph_store:
    g = pg_index.property_graph_store.graph
    print(f"\nGraph Nodes:                     {len(g.nodes)}")
    print(f"Graph Relations (Triplets):      {len(g.relations)}")
else:
    print("\nGraph Index not populated or unavailable.")

# Appendix:
# du -sh ./chroma_db_academic # file size
