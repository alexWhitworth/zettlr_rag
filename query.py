# query.py
import argparse
import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime

import chromadb
import nest_asyncio
import numpy as np
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.postprocessor import (
    LLMRerank,
    LongContextReorder,
    SimilarityPostprocessor,
)
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import QueryFusionRetriever, VectorIndexRetriever
from llama_index.core.vector_stores import (
    FilterCondition,
    MetadataFilter,
    MetadataFilters,
)
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.chroma import ChromaVectorStore

from zettlr_rag.consts import (
    GEMINI_CONTEXT_WINDOWS,
    GEMINI_PRICING,
    MODEL_NAME,
    SYSTEM_PROMPT,
)
from zettlr_rag.metrics import (
    QueryMetrics,
    calculate_cost,
    calculate_window_utilization,
)
from zettlr_rag.postprocessors import (
    MMRPostprocessor,
)
from zettlr_rag.rag_setup import get_last_token_usage, reset_token_usage, setup_settings
from zettlr_rag.telemetry import (
    get_langfuse_client,
    init_telemetry,
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING)


@dataclass(frozen=True)
class RAGQueryConfig:
    """Configuration for RAG query execution."""
    similarity_top_k: int = 25
    system_prompt: str = SYSTEM_PROMPT
    instrumented: bool = False
    run_id: str | None = None
    chroma_path: str = "./chroma_db_academic"
    index_persist_dir: str = "./.index_metadata"
    collection_name: str = "research_papers"
    log_path: str = "query_log.jsonl"


class RAGQueryRunner:
    """
    Encapsulates the RAG query lifecycle: execution, metrics, and telemetry.

    This class coordinates the multi-stage hybrid retrieval pipeline, including
    vector and BM25 search, Reciprocal Rank Fusion (RRF), and post-processing steps 
    like MMR diversity filtering and LLM-based reranking. It is designed to provide 
    high-precision answers from academic markdown libraries while strictly tracking 
    performance and financial costs.

    The runner integrates thread-local token capturing to ensure that the token usage 
    from all pipeline stages—including expensive reranking steps—is accurately 
    summed and logged. It also supports metadata-based filtering and Langfuse 
    instrumentation for deep observability.

    Methods:
        query: Orchestrates the full query lifecycle from retrieval to metric logging.
    """

    def __init__(self, config: RAGQueryConfig, filters: MetadataFilters | None = None):
        self.config = config
        self.filters = filters
        self.engine = self._initialize_engine()
        self.lf_client = get_langfuse_client() if config.instrumented else None

    def _initialize_engine(self):
        """Connects to the persistent index and returns a hybrid retriever query engine."""
        setup_settings()
        db = chromadb.PersistentClient(path=self.config.chroma_path)
        chroma_collection = db.get_or_create_collection(self.config.collection_name)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

        # Load storage context for docstore (required by BM25)
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store,
            persist_dir=self.config.index_persist_dir,
        )
        index = load_index_from_storage(storage_context)

        # 1. Base Retrievers
        vector_retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=self.config.similarity_top_k,
            filters=self.filters,
        )

        # BM25 requires exactly one of index, nodes, or docstore.
        # Passing the index is the cleanest way to initialize it.
        bm25_retriever = BM25Retriever.from_defaults(
            index=index,
            similarity_top_k=self.config.similarity_top_k,
        )

        # 2. Fusion
        fusion_retriever = QueryFusionRetriever(
            [vector_retriever, bm25_retriever],
            similarity_top_k=self.config.similarity_top_k,
            num_queries=1,  # No query generation
            mode="reciprocal_rerank",
            use_async=False,
        )

        # 3. Post-processing Pipeline
        node_postprocessors = [
            MMRPostprocessor(mmr_threshold=0.55, top_n=15),
            LLMRerank(top_n=8),
            LongContextReorder(),
        ]

        return RetrieverQueryEngine.from_args(
            retriever=fusion_retriever,
            node_postprocessors=node_postprocessors,
            system_prompt=self.config.system_prompt,
        )

    def query(self, question: str) -> tuple[object, QueryMetrics]:
        """Orchestrates the full query lifecycle."""
        reset_token_usage()
        wall_start = time.monotonic()

        if self.config.instrumented:
            response = self._monitored_query(question)
        else:
            response = self.engine.query(question)

        wall_time = time.monotonic() - wall_start

        # ── Extract and compute metrics ───────────────────────────────────────
        usage = get_last_token_usage()
        input_tokens = usage.input_tokens
        output_tokens = usage.output_tokens
        cache_tokens = usage.cache_tokens
        total_tokens = usage.total_tokens

        cost_input, cost_output, cost_cache, cost_total = calculate_cost(
            usage=usage,
            model_name=MODEL_NAME,
            pricing_table=GEMINI_PRICING,
        )
        window_size, window_util_pct = calculate_window_utilization(
            input_tokens=input_tokens,
            model_name=MODEL_NAME,
            window_table=GEMINI_CONTEXT_WINDOWS,
        )

        source_nodes = response.source_nodes or []
        similarity_scores = [n.get_score() for n in source_nodes if n.get_score() is not None]
        unique_docs = len(set(n.node.metadata.get("file_name") for n in source_nodes if "file_name" in n.node.metadata))

        metrics = QueryMetrics(
            question=question,
            model_name=MODEL_NAME,
            run_id=self.config.run_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_tokens=cache_tokens,
            total_tokens=total_tokens,
            cost_input_usd=cost_input,
            cost_output_usd=cost_output,
            cost_cache_usd=cost_cache,
            cost_total_usd=cost_total,
            context_window_size=window_size,
            window_utilization_pct=window_util_pct,
            wall_time_ms=round(wall_time * 1000, 2),
            chunks_retrieved=len(source_nodes),
            docs_retrieved=unique_docs,
            top_similarity=max(similarity_scores) if similarity_scores else 0.0,
            mean_similarity=(
                sum(similarity_scores) / len(similarity_scores)
                if similarity_scores else 0.0
            ),
            p10_similarity=float(np.percentile(similarity_scores, 10)) if similarity_scores else 0.0,
            p90_similarity=float(np.percentile(similarity_scores, 90)) if similarity_scores else 0.0,
        )

        # ── Persistent Local Storage ─────────────────────────────────────────
        self._append_to_local_log(metrics, str(response))

        # ── Post telemetry ───────────────────────────────────────────────────
        if self.config.instrumented and self.lf_client:
            try:
                self._post_scores(metrics)
            except Exception as exc:
                log.warning(f"Failed to post scores to Langfuse: {exc}")

        return response, metrics

    def _monitored_query(self, question: str):
        """Inner query call wrapped with Langfuse observation."""
        from langfuse import observe

        @observe(name="rag-query")
        def _run_aquery():
            # Attempt async query for potentially better metadata preservation
            try:
                return asyncio.run(self.engine.aquery(question))
            except Exception:
                return self.engine.query(question)

        return _run_aquery()

    def _post_scores(self, metrics: QueryMetrics):
        """Post computed metrics as scores to the current Langfuse trace."""
        from langfuse import observe

        @observe(name="post-metrics")
        def _do_post():
            trace_id = self.lf_client.get_current_trace_id()
            if trace_id:
                for name, value in metrics.to_langfuse_scores().items():
                    self.lf_client.create_score(
                        trace_id=trace_id,
                        name=name,
                        value=value,
                        data_type="NUMERIC",
                    )
        _do_post()

    def _append_to_local_log(self, metrics: QueryMetrics, answer: str) -> None:
        """
        Appends one result record to a local JSONL file.
        Creates the file if it doesn't exist.
        Each line is a complete, self-contained JSON record.
        """
        record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "question": metrics.question,
            "answer": answer,
            "model_name": metrics.model_name,
            "run_id": metrics.run_id,
            "input_tokens": metrics.input_tokens,
            "output_tokens": metrics.output_tokens,
            "cache_tokens": metrics.cache_tokens,
            "total_tokens": metrics.total_tokens,
            "cost_input_usd": metrics.cost_input_usd,
            "cost_output_usd": metrics.cost_output_usd,
            "cost_cache_usd": metrics.cost_cache_usd,
            "cost_total_usd": metrics.cost_total_usd,
            "window_utilization_pct": metrics.window_utilization_pct,
            "wall_time_ms": metrics.wall_time_ms,
            "chunks_retrieved": metrics.chunks_retrieved,
            "docs_retrieved": metrics.docs_retrieved,
            "top_similarity": metrics.top_similarity,
            "mean_similarity": metrics.mean_similarity,
            "p10_similarity": metrics.p10_similarity,
            "p90_similarity": metrics.p90_similarity,
        }

        with open(self.config.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")


def parse_complex_filters(filter_data):
    """Recursively parses a dict into LlamaIndex MetadataFilters."""
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
            filters.append(parse_complex_filters(item))
        else:
            filters.append(
                MetadataFilter(
                    key=item["key"],
                    value=item["value"],
                    operator=item.get("operator", "=="),
                )
            )

    return MetadataFilters(
        filters=filters,
        condition=FilterCondition.OR if condition_str == "or" else FilterCondition.AND,
    )


def print_metrics_stderr(metrics: QueryMetrics) -> None:
    """Print a human-readable metrics summary to stderr."""
    print("\n---", file=sys.stderr)
    print("📊 Query Metrics:", file=sys.stderr)
    print(f"  Wall time:             {metrics.wall_time_ms:.0f} ms", file=sys.stderr)
    print(f"  Tokens (in/out/cache): {metrics.input_tokens} / {metrics.output_tokens} / {metrics.cache_tokens}", file=sys.stderr)
    print(f"  Total tokens:          {metrics.total_tokens}", file=sys.stderr)
    print(f"  Cost:                  ${metrics.cost_total_usd:.6f} USD", file=sys.stderr)
    print(f"  Window utilization:    {metrics.window_utilization_pct:.2f}%", file=sys.stderr)
    print(f"  Chunks retrieved:      {metrics.chunks_retrieved}", file=sys.stderr)
    print(f"  Documents retrieved:   {metrics.docs_retrieved}", file=sys.stderr)
    print(f"  Similarity (top/mean): {metrics.top_similarity:.3f} / {metrics.mean_similarity:.3f}", file=sys.stderr)
    print(f"  Similarity (p10/p90):  {metrics.p10_similarity:.3f} / {metrics.p90_similarity:.3f}", file=sys.stderr)


def main():
    nest_asyncio.apply()

    # see README.md for usage examples, including complex filter construction
    parser = argparse.ArgumentParser(description="Query the Zettlr MD-RAG Library")
    parser.add_argument("question",      type=str, help="The question to ask.")
    parser.add_argument("--year",        type=int, help="Filter papers by year.")
    parser.add_argument("--category",    type=str, help="Filter by folder category.")
    parser.add_argument("--tag",         type=str, help="Filter by specific tag.")
    parser.add_argument("--filter-json", type=str, help="Complex Boolean logic (JSON string or path to .json file).")
    parser.add_argument("--run-id",      type=str, help="Optional run ID for reliability testing.", default=None)
    parser.add_argument("--no-metrics",  action="store_true", help="Suppress metrics output to stderr.")
    parser.add_argument("--show-sources", action="store_true", help="Display retrieved source nodes.")

    args = parser.parse_args()

    # ── Initialize Telemetry ─────────────────────────────────────────────────
    instrumented = init_telemetry()

    # ── Filter construction ───────────────────────────────────────────────────
    filters = None
    if args.filter_json:
        if os.path.exists(args.filter_json):
            with open(args.filter_json) as f:
                filter_data = json.load(f)
        else:
            filter_data = json.loads(args.filter_json)
        filters = parse_complex_filters(filter_data)
    else:
        filter_list = []
        if args.year:
            filter_list.append(MetadataFilter(key="year", value=args.year))
        if args.category:
            filter_list.append(MetadataFilter(key="category", value=args.category))
        if args.tag:
            filter_list.append(MetadataFilter(key="tags", value=args.tag))
        if filter_list:
            filters = MetadataFilters(filters=filter_list, condition=FilterCondition.AND)

    # ── Execute query ──────────────────────────────────────────────────────────
    config = RAGQueryConfig(instrumented=instrumented, run_id=args.run_id)
    runner = RAGQueryRunner(config=config, filters=filters)
    response, metrics = runner.query(args.question)

    # ── Output ────────────────────────────────────────────────────────────────
    print(f"# Query: {args.question}\n")
    print(response)

    if not args.no_metrics:
        print_metrics_stderr(metrics)

    if args.show_sources:
        print("\n📚 Sources used:", file=sys.stderr)
        for node in response.source_nodes:
            m = node.metadata
            print(
                f"- {m.get('file_name', 'Unknown')} [{m.get('category', 'N/A')}, "
                f"{m.get('year', 'N/A')}] (Score: {node.get_score():.2f})",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()
