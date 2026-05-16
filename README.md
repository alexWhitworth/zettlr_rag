# Zettlr RAG (MD-RAG) + LLM Knowledge Base

A specialized Retrieval-Augmented Generation (RAG) system for personal note libraries. This system 
implements **MD-RAG** (Metadata RAG), preserving and utilizing YAML frontmatter from Zettlr markdown 
files for high-precision scientific retrieval.

It can equivalently be thought of as an "LLM Knowledge Base" ([Karpathy](https://x.com/karpathy/status/2039805659525644595)),
though technically it is the "fancy RAG" that Karpathys states he doesn't need for his ~100 
file LLM knowledge base. 

The implementation was fully evaluated with ~700 academic papers vs [Consensus AI](https://consensus.app/) 
(SotA Academic RAG).
- **tl;dr:** While I can't compete with Consensus AI's 200M+ paper breadth, my system achieved 
*45% parity* on deep synthesis queries. 
    - _See `evals/eval_results.md` for the full details._

## Installation

```bash
uv pip install -e .
```

## Architectural Design

### Initialization: Ingest and Compile

1. Academic papers were first summarized via a `RESEARCH_AGENT.md` SKILL using Claude Opus 4.6, which 
utilizes clear markdown formatting aligned with MD-RAG. New papers are added utilizing the same 
SKILL. **This is the most expensive step in setup.** The markdown summaries, and not the raw PDFs, 
are then submitted to the embedding model and used in the RAG.
2. Books and blog-post notes, which come from reading non-fiction books and articles over time, are 
cleaned up. Specifically, some books have meticulous "book-report" style notes while others are just
raw dumps of [Kindle highlighting](https://read.amazon.com/notebook). I designed and use a 
`MARKDOWN_CLEANUP.md` SKILL to compile these to a standard format.

### Retrieval Strategy

The system utilizes a multi-stage hybrid retrieval pipeline to ensure high precision and diversity
in the context provided to the LLM:

1.  **Hybrid Retrieval**: Combines semantic search (**Vector**), keyword-based search (**BM25**), 
and relational triplet search (**Property Graph**).
2.  **Property Graph Extraction**: During ingestion, a `SchemaLLMPathExtractor` processes the nodes 
to build an academic knowledge graph based on defined entity relationships (e.g. `Paper` `USES_METHOD` 
`Method`). This graph runs parallel to the vector store to ground relationships.
3.  **Fusion**: Uses **Reciprocal Rank Fusion (RRF)** with `reciprocal_rerank` mode to merge and 
normalize results from the Vector, BM25, and Graph retrievers.
4.  **Refinement Pipeline**:
    *   **MMR (Maximum Marginal Relevance)**: Reranks for diversity to avoid redundant information
    in the context window.
    *   **LLM Reranking**: Uses the LLM to perform a final precision-based reranking of the top nodes.
    *   **Long Context Reordering**: Reorders nodes to combat the "lost in the middle" effect, 
    placing most relevant information at the start and end of the prompt.

### Tech Stack
- **LLM**: Gemini 3 Flash Preview (`gemini-3-flash-preview`)
- **Embeddings**: Gemini Embedding 2 Preview (`gemini-embedding-2-preview`)
- **Vector Store**: ChromaDB (Persistent)
- **Persona**: Senior Staff Data Scientist, Algorithms

### Implementation Details
- **Structural Parsing**: U ses `MarkdownNodeParser` to preserve headers and logical sections.
- **Smart Sync**: Uses a persistent Document Store to track file hashes, preventing 
double-indexing. It handles file moves by applying metadata updates in-place to both the Vector 
and Graph stores, bypassing expensive LLM re-extractions.
- **Rate Limit Optimized**: Implements exponential backoff and batch-size control (1 node/request).
- **Persistent Storage**: Database stored in `./chroma_db_academic` and `./.graph_index`.

### Scientific Persona & Formatting

The system uses a **Senior Staff Data Scientist** persona as defined in the `SYSTEM_PROMPT` constant:
- **BibTeX citations** included for all referenced papers.
- **LaTeX** for all mathematical expressions.
- **Python** preferred for code examples.
- **Pure Markdown** output for direct file redirection/chaining.

The persona is configurable. See `src/zettlr_rag/consts.py` for the `SYSTEM_PROMPT`.

## Evaluation

**Results:** _See `evals/eval_results.md` for details._

### Measurement Framework

We implement observability and evaluation to cover key dimensions of RAGAS (Shahul, et al (2024))
and CLEAR (Sushant, 2025). (1) Cost; (2) Latency; (3) Answer Quality; and (4) Answer Reliability.

_Note: In production settings, we recommend implementing AMDM (Shukla, 2025). While I did write a_
_full system design for AMDM, full implementation was overkill for this project._

### Implementation

- **Observability**: Integrates **Langfuse v3+** and **OpenInference** for full-stack RAG observability.
    - Captures query traces, spans, and metadata automatically.
    - Tracks token usage, latency, cost and retrieval metrics per query.
    - Persistent local audit log in `query_log.jsonl`.
- **Answer Reliability:** Do repeated query calls yield equivalent answers?
    - Implements `ReliabilityHarness` class for evaluations with persistant log in `evals/data/validation_log.jsonl`
    - Observability metrics: Cost, latency, token usage
    - Semantic consistency (utilizing answer embeddings): 
        1. Spherical Mean Resultant Length: $R = \frac{1}{N} \sum_i ||\hat v_i||$ where $\hat v_i$ is a unit vector
            - $R \in [0,1]$. Higher is better
        2. Centroid Dispersion: $CD = \frac{1}{N} \sum_i ||v_i - \mu||_2$ where $\mu = \frac{1}{n} \sum_i v_i$
            - $CD \in [0, \infty)$. Lower is better
        3. Semantic Entropy: $H_{\text{sem}} = -\sum_i p_i \times log2(p_i))$ where $p_i$ is the 
        proportion of embeddings in cluster i resulting from agglomerative clustering.
            - **Gold Standard** measurement, but more expensive.
- **Answer Quality:** Implemented an LLM-as-a-Judge pipeline following RAGAS evaluation principles.

## Usage

### 1. Smart Sync (Setup & Maintenance)
Uses **Smart Sync** logic. It tracks file hashes in `./.index_metadata` to perform incremental
updates.

- **Initial Run**: Processes your entire library.
- **Subsequent Runs**: Detects and indexes **only** new or modified files.
- **Deletions**: Automatically removes vectors for files you've deleted from your library.

```bash
# preferably done within tmux:
# 01. Vector initialization (one-time)
uv run src/zettlr_rag/rag_setup.py

# 02. Property Graph setup (one-time)
uv run src/zettlr_rag/build_graph.py

# 03. Maintenance (smart-sync): updates both vector embeddings and Property Graph
uv run src/zettlr_rag/rag_setup.py
```

### 2. Querying

#### CLI Usage
The `query.py` script is the primary entry point for terminal-based research. All queries are automatically logged to `query_log.jsonl` for persistent auditing.

```bash
# Standard Query
uv run query.py "What are the core components of GP models?"

# Chaining to Markdown (Redirect stdout to file)
uv run query.py "Summarize Gaussian Process requirements" >> research_notes.md

# Simple Filters (AND logic)
uv run query.py "Algorithm breakdown" --category economics --year 2024

# Advanced Boolean Logic (JSON String or File)
# You can pass a raw JSON string or a path to a *.json file.
uv run query.py "Complex search" --filter-json '{
    "or": [
        {"and": [{"key": "category", "value": "economics"}, 
        {"key": "year", "value": 2024}]}, 
        {"key": "category", "value": "statistics"}
    ]
}'

# Using a filter file
uv run query.py "Research from 2020-2022" --filter-json filters.json
```

#### Accessing Logs
You can load your query history into a pandas DataFrame using the provided utility:

```python
from zettlr_rag.utils import load_query_log
df = load_query_log("query_log.jsonl")
print(df.head())
```

#### JSON Filter Syntax (ExactMatch, Range, InFilter)
The `--filter-json` option supports standard LlamaIndex operators to implement different filter types.

| Filter Type | Operator | Example | Description |
| :--- | :--- | :--- | :--- |
| **ExactMatch** | `==` | `{"key": "year", "value": 2024}` | Matches the value exactly. |
| **Range** | `>`, `<`, `>=`, `<=` | `{"key": "year", "value": 2020, "operator": ">="}` | Matches values within a numerical range. |
| **InFilter** | `in` | `{"key": "tags", "value": "GP", "operator": "in"}` | Matches if the value is within a list (e.g., tags). |

**Example `filters.json`:**
```json
{
    "and": [
        {"key": "year", "value": 2020, "operator": ">="},
        {"key": "year", "value": 2022, "operator": "<="},
        {"key": "tags", "value": "economics", "operator": "in"}
    ]
}
```

#### Python Usage
```python
import nest_asyncio
from query import get_query_engine, parse_complex_filters

# Required for async parsing in notebooks
nest_asyncio.apply()

# Define filters as a dictionary
filter_data = {
    "and": [
        {"key": "year", "value": 2022, "operator": ">="},
        {"key": "tags", "value": "statistics", "operator": "in"}
    ]
}

# Parse and apply to engine
filters = parse_complex_filters(filter_data)
engine = get_query_engine(filters=filters)

response = engine.query("Your technical question here")
print(response)
```

## Advanced Querying (MD-RAG Filters)

The system automatically maps your YAML header to searchable metadata.

| YAML Key | Filter Type | JSON Operator | Example Use Case |
| :--- | :--- | :--- | :--- |
| `year` | `ExactMatch`, `Range` | `==`, `>`, `<`, etc. | Filter by specific year or period. |
| `category` | `ExactMatch` | `==` | Narrow search to a top-level folder. |
| `tags` | `InFilter` | `in` | Filter by methodology or sub-topic. |
| `authors` | `InFilter` | `in` | Find research by a specific scientist. |

## Testing

The project uses `pytest` for testing. The tests are designed to be isolated and use temporary 
directories for ChromaDB and metadata to avoid impacting your production data.

Test coverage is currently ~85%.

### Running Tests
To run all tests:

```bash
uv run pytest
```
