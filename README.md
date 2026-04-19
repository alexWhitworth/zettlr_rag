# Zettlr RAG (MD-RAG)

A specialized Retrieval-Augmented Generation (RAG) system for academic paper libraries. This system implements **MD-RAG** (Metadata RAG), preserving and utilizing YAML frontmatter from Zettlr markdown files for high-precision scientific retrieval.

## Tech Stack
- **LLM**: Gemini 3 Flash Preview (`gemini-3-flash-preview`)
- **Embeddings**: Gemini Embedding 2 Preview (`gemini-embedding-2-preview`)
- **Vector Store**: ChromaDB (Persistent)
- **Persona**: Senior Staff Data Scientist, Algorithms

## Installation

```bash
uv pip install -e .
```

## Usage

### 1. Smart Sync (Setup & Maintenance)
The `zettlr-rag-setup` command uses **Smart Sync** logic. It tracks file hashes in `./.index_metadata` to perform incremental updates.

- **Initial Run**: Processes your entire library.
- **Subsequent Runs**: Detects and indexes **only** new or modified files.
- **Deletions**: Automatically removes vectors for files you've deleted from your library.

```bash
zettlr-rag-setup
```

### 2. Live Monitoring (Optional)
If you prefer real-time updates while you write, run the watcher. It will automatically trigger indexing for any file you save.

```bash
zettlr-rag-watch
```

### 3. Querying

#### CLI Usage
The `query.py` script is the primary entry point for terminal-based research.

```bash
# Standard Query
uv run query.py "What are the core components of GP models?"

# Chaining to Markdown (Redirect stdout to file)
uv run query.py "Summarize Gaussian Process requirements" >> research_notes.md

# Simple Filters (AND logic)
uv run query.py "Algorithm breakdown" --category economics --year 2024

# Advanced Boolean Logic (JSON String)
uv run query.py "Complex search" --filter-json '{"or": [{"and": [{"key": "category", "value": "economics"}, {"key": "year", "value": 2024}]}, {"key": "category", "value": "statistics"}]}'
```

#### Python Usage
```python
import nest_asyncio
from query import get_query_engine

# Required for async parsing in notebooks
nest_asyncio.apply()

engine = get_query_engine()
response = engine.query("Your technical question here")
print(response)
```

## Advanced Querying (MD-RAG Filters)

The system automatically maps your YAML header to searchable metadata.

| YAML Key | Filter Type | Example Use Case |
| :--- | :--- | :--- |
| `year` | `ExactMatch`, `Range` | Filter by specific year or period. |
| `category` | `ExactMatch` | Narrow search to a top-level folder. |
| `tags` | `InFilter` | Filter by methodology or sub-topic. |
| `authors` | `InFilter` | Find research by a specific scientist. |

## Scientific Persona & Formatting
The system uses a **Senior Staff Data Scientist** persona:
- **BibTeX citations** included for all referenced papers.
- **LaTeX** for all mathematical expressions.
- **Python** preferred for code examples.
- **Pure Markdown** output for direct file redirection/chaining.

## Implementation Details
- **Structural Parsing**: Uses `MarkdownNodeParser` to preserve headers and logical sections.
- **Smart Sync**: Uses a persistent Document Store to track file hashes, preventing double-indexing.
- **Rate Limit Optimized**: Implements exponential backoff and batch-size control (1 node/request).
- **Persistent Storage**: Database stored in `./chroma_db_academic`.
