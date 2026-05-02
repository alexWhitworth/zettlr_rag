# Zettlr RAG (MD-RAG)

A specialized Retrieval-Augmented Generation (RAG) system for academic paper libraries. This system implements **MD-RAG** (Metadata RAG), preserving and utilizing YAML frontmatter from Zettlr markdown files for high-precision scientific retrieval.

## Installation

```bash
uv pip install -e .
```

## Architectural Design

### Initialization

Historical papers were first summarized via a `RESEARCH_AGENT.md` SKILL using Claude Opus 4.6, which 
utilizes clear markdown formatting aligned with MD-RAG. New papers are added utilizing the same 
SKILL. This is the most expensive step in setup.

The markdown summaries, and not the raw PDFs, are then submitted to the embeddings model and used 
in the RAG.

### Advanced Retrieval Strategy

The system utilizes a multi-stage hybrid retrieval pipeline to ensure high precision and diversity in the context provided to the LLM:

1.  **Hybrid Retrieval**: Combines semantic search (**Vector**) with keyword-based search (**BM25**).
2.  **Fusion**: Uses **Reciprocal Rank Fusion (RRF)** with `reciprocal_rerank` mode to merge and normalize results from different retrieval methods.
3.  **Refinement Pipeline**:
    *   **MMR (Maximum Marginal Relevance)**: Reranks for diversity to avoid redundant information in the context window.
    *   **LLM Reranking**: Uses the LLM to perform a final precision-based reranking of the top nodes.
    *   **Long Context Reordering**: Reorders nodes to combat the "lost in the middle" effect, placing most relevant information at the start and end of the prompt.

### Tech Stack
- **LLM**: Gemini 3 Flash Preview (`gemini-3-flash-preview`)
- **Embeddings**: Gemini Embedding 2 Preview (`gemini-embedding-2-preview`)
- **Vector Store**: ChromaDB (Persistent)
- **Persona**: Senior Staff Data Scientist, Algorithms

### Implementation Details
- **Structural Parsing**: Uses `MarkdownNodeParser` to preserve headers and logical sections.
- **Smart Sync**: Uses a persistent Document Store to track file hashes, preventing double-indexing.
- **Rate Limit Optimized**: Implements exponential backoff and batch-size control (1 node/request).
- **Persistent Storage**: Database stored in `./chroma_db_academic`.

### Evaluation
- **Observability**: Integrates **Langfuse v3+** and **OpenInference** for full-stack RAG observability.
    - Captures query traces, spans, and metadata automatically.
    - Tracks token usage, latency, and cost per query using custom `TokenCapturingGemini` wrapper.
        - Needed due to [llama_index bug](https://github.com/run-llama/llama_index/issues/19293)
    - Persistent local audit log in `query_log.jsonl`.
- **Reliability:** Do repeated query calls yield equivalent responses?
    - Implements `ReliabilityHarness` for evaluations
    - Standard metrics: Cost, latency, token usage
    - Semantic consistency evaluated based on answer embeddings: 
        1. Spherical Mean Resultant Length: $R = \frac{1}{N} \sum_i ||\hat v_i||$ where $\hat v_i$ is a unit vector
            - $R \in [0,1]$. Higher is better
        2. Centroid Dispersion: $CD = \frac{1}{N} \sum_i ||v_i - \mu||_2$ where $\mu = \frac{1}{n} \sum_i v_i$
            - $CD \in [0, \infty). Lower is better
        3. **Gold Standard** Semantic Entropy: $H_\text{sem} = -\sum_i p_i \times log2(p_i))$ where $p_i$ is the proportion of embeddings in cluster i resulting from agglomerative clustering.
- **Answer Quality:** Implemented an LLM-as-a-Judge pipeline over 75 'goldset' questions covering the span of my acadmeic library
    - RAG answers are compared to [Consensus AI](https://consensus.app/) and associated citations
    - **Results:** _Pending_

### Scientific Persona & Formatting

The system uses a **Senior Staff Data Scientist** persona as defined in the `SYSTEM_PROMPT` constant:
- **BibTeX citations** included for all referenced papers.
- **LaTeX** for all mathematical expressions.
- **Python** preferred for code examples.
- **Pure Markdown** output for direct file redirection/chaining.

The persona is configurable. See `src/zettlr_rag/consts.py` for the `SYSTEM_PROMPT`.

## Usage

### 1. Smart Sync (Setup & Maintenance)
The `zettlr-rag-setup` command uses **Smart Sync** logic. It tracks file hashes in `./.index_metadata` 
to perform incremental updates.

- **Initial Run**: Processes your entire library.
- **Subsequent Runs**: Detects and indexes **only** new or modified files.
- **Deletions**: Automatically removes vectors for files you've deleted from your library.

```bash
zettlr-rag-setup
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

### Running Tests
To run all tests:

```bash
uv run pytest
```

### Test Coverage
The tests cover:
- **RAG Setup**: Initializing settings, loading documents, and the full indexing pipeline (using mocks to avoid API calls).
- **Isolated Environments**: Verification that tests do not modify `chroma_db_academic` or `.index_metadata`.
