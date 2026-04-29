"""
Custom node postprocessors for the Zettlr RAG pipeline.
"""


import numpy as np
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle


class BibtexExclusionPostprocessor(BaseNodePostprocessor):
    """
    Postprocessor to exclude nodes containing BibTeX entries.
    Identified by the header '### 7. BibTeX'.
    """

    @classmethod
    def class_name(cls) -> str:
        return "BibtexExclusionPostprocessor"

    def _postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: QueryBundle | None = None,
    ) -> list[NodeWithScore]:
        return [
            node for node in nodes
            if "### 7. BibTeX" not in node.node.get_content()
        ]


class MMRPostprocessor(BaseNodePostprocessor):
    """
    Maximum Marginal Relevance (MMR) postprocessor.
    Balances relevance and diversity.

    λ: 0 = pure diversity, 1 = pure relevance.
    """
    top_n: int = 12
    mmr_threshold: float = 0.6

    @classmethod
    def class_name(cls) -> str:
        return "MMRPostprocessor"

    def _postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: QueryBundle | None = None,
    ) -> list[NodeWithScore]:
        if not nodes:
            return []

        # Filter out nodes without embeddings
        nodes_with_embs = [n for n in nodes if n.node.embedding is not None]
        if not nodes_with_embs:
            return nodes[:self.top_n]

        embeddings = np.array([n.node.embedding for n in nodes_with_embs])

        # Normalise embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.where(norms == 0, 1, norms)

        selected_indices: list[int] = []
        remaining_indices = list(range(len(embeddings)))

        while len(selected_indices) < self.top_n and remaining_indices:
            if not selected_indices:
                # First pick: most relevant (highest score from retriever)
                scores = [nodes_with_embs[i].score or 0.0 for i in remaining_indices]
                best_idx_in_remaining = int(np.argmax(scores))
                best = remaining_indices[best_idx_in_remaining]
            else:
                # MMR: balance relevance vs similarity to already selected
                selected_embs = embeddings[selected_indices]
                best, best_score = -1, -np.inf

                for i in remaining_indices:
                    relevance = nodes_with_embs[i].score or 0.0
                    # Redundancy is the max similarity to any already selected node
                    redundancy = float(np.max(embeddings[i] @ selected_embs.T))
                    mmr_score = (self.mmr_threshold * relevance) - ((1 - self.mmr_threshold) * redundancy)

                    if mmr_score > best_score:
                        best, best_score = i, mmr_score

            if best == -1:
                break

            selected_indices.append(best)
            remaining_indices.remove(best)

        return [nodes_with_embs[i] for i in selected_indices]
