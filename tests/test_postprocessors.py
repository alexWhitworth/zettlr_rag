from llama_index.core.schema import NodeWithScore, TextNode

from zettlr_rag.postprocessors import MMRPostprocessor


def test_mmr_postprocessor_diversity() -> None:
    # Create nodes with specific embeddings to test diversity selection
    # Node 0 and Node 1 are very similar
    # Node 2 is different
    node0 = NodeWithScore(node=TextNode(text="Node 0", embedding=[1.0, 0.0]), score=0.9)
    node1 = NodeWithScore(node=TextNode(text="Node 1", embedding=[0.9, 0.1]), score=0.85)
    node2 = NodeWithScore(node=TextNode(text="Node 2", embedding=[0.0, 1.0]), score=0.7)

    nodes = [node0, node1, node2]

    # MMR with top_n=2 and mmr_threshold=0.5 (equal weight)
    # 1st pick: Node 0 (highest score)
    # 2nd pick: Should be Node 2 because Node 1 is too redundant with Node 0
    postprocessor = MMRPostprocessor(top_n=2, mmr_threshold=0.5)
    selected_nodes = postprocessor.postprocess_nodes(nodes)

    assert len(selected_nodes) == 2
    assert selected_nodes[0].get_content() == "Node 0"
    assert selected_nodes[1].get_content() == "Node 2"


def test_mmr_postprocessor_relevance_dominant() -> None:
    # If mmr_threshold is high, it should favor relevance (scores)
    node0 = NodeWithScore(node=TextNode(text="Node 0", embedding=[1.0, 0.0]), score=0.9)
    node1 = NodeWithScore(node=TextNode(text="Node 1", embedding=[0.9, 0.1]), score=0.85)
    node2 = NodeWithScore(node=TextNode(text="Node 2", embedding=[0.0, 1.0]), score=0.7)

    nodes = [node0, node1, node2]

    # MMR with top_n=2 and mmr_threshold=0.99 (pure relevance)
    postprocessor = MMRPostprocessor(top_n=2, mmr_threshold=0.99)
    selected_nodes = postprocessor.postprocess_nodes(nodes)

    assert len(selected_nodes) == 2
    assert selected_nodes[0].get_content() == "Node 0"
    assert selected_nodes[1].get_content() == "Node 1"
