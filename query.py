import os
import sys
import chromadb
import nest_asyncio
import argparse
import json
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.vector_stores import (
    MetadataFilters, 
    MetadataFilter, 
    FilterOperator, 
    FilterCondition
)
from llama_index.vector_stores.chroma import ChromaVectorStore
from zettlr_rag.rag_setup import setup_settings

def get_query_engine(filters=None):
    """Connects to the persistent index and returns a query engine."""
    setup_settings()
    db = chromadb.PersistentClient(path="./chroma_db_academic")
    chroma_collection = db.get_or_create_collection("research_papers")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    index = VectorStoreIndex.from_vector_store(vector_store)
    
    return index.as_query_engine(
        similarity_top_k=5,
        filters=filters,
        system_prompt=Settings.system_prompt
    )

def parse_complex_filters(filter_data):
    """
    Recursively parses a dict into LlamaIndex MetadataFilters.
    Supports 'and'/'or' keys with lists of conditions.
    """
    if not isinstance(filter_data, dict):
        return filter_data

    condition_str = "and"
    if "or" in filter_data:
        condition_str = "or"
        items = filter_data["or"]
    else:
        items = filter_data.get("and", [])

    filters = []
    for item in items:
        if isinstance(item, dict) and ("and" in item or "or" in item):
            # Nested filters
            filters.append(parse_complex_filters(item))
        else:
            # Single filter: {"key": "year", "value": 2024, "operator": "=="}
            filters.append(MetadataFilter(
                key=item["key"],
                value=item["value"],
                operator=item.get("operator", "==")
            ))
            
    return MetadataFilters(
        filters=filters,
        condition=FilterCondition.OR if condition_str == "or" else FilterCondition.AND
    )

def main():
    nest_asyncio.apply()

    parser = argparse.ArgumentParser(description="Query the Zettlr MD-RAG Library")
    parser.add_argument("question", type=str, help="The question to ask.")
    parser.add_argument("--year", type=int, help="Filter papers by year.")
    parser.add_argument("--category", type=str, help="Filter by folder category.")
    parser.add_argument("--tag", type=str, help="Filter by specific tag.")
    parser.add_argument("--filter-json", type=str, help="Complex Boolean logic (JSON string or path to .json file).")
    
    args = parser.parse_args()

    filters = None
    if args.filter_json:
        # Check if it's a file path or raw JSON
        if os.path.exists(args.filter_json):
            with open(args.filter_json, 'r') as f:
                filter_data = json.load(f)
        else:
            filter_data = json.loads(args.filter_json)
        
        print(f"Applying complex filters", file=sys.stderr)
        filters = parse_complex_filters(filter_data)
    else:
        # Simple AND logic from flags
        filter_list = []
        if args.year:
            filter_list.append(MetadataFilter(key="year", value=args.year))
        if args.category:
            filter_list.append(MetadataFilter(key="category", value=args.category))
        if args.tag:
            filter_list.append(MetadataFilter(key="tags", value=args.tag))

        if filter_list:
            print(f"Applying simple filters: {args.__dict__}", file=sys.stderr)
            filters = MetadataFilters(filters=filter_list, condition=FilterCondition.AND)

    engine = get_query_engine(filters=filters)
    response = engine.query(args.question)
    
    print(f"# Query: {args.question}\n")
    print(response)
    
    print("\n---", file=sys.stderr)
    print("📚 Sources used:", file=sys.stderr)
    for node in response.source_nodes:
        m = node.metadata
        print(f"- {m.get('file_name', 'Unknown')} [{m.get('category', 'N/A')}, {m.get('year', 'N/A')}] (Score: {node.get_score():.2f})", file=sys.stderr)

if __name__ == "__main__":
    main()
