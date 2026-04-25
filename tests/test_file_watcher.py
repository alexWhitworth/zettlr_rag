from unittest.mock import MagicMock

from zettlr_rag.file_watcher import NewPaperHandler


def test_process_new_file(tmp_path):
    # Setup mock library structure
    lib_dir = tmp_path / "library"
    lib_dir.mkdir()
    test_file = lib_dir / "test_paper.md"
    test_file.write_text("---\ntitle: Test\n---\nContent")

    mock_index = MagicMock()
    mock_node_parser = MagicMock()
    mock_node_parser.get_nodes_from_documents.return_value = [MagicMock()]

    # Corrected instantiation with base_path
    handler = NewPaperHandler(
        index=mock_index,
        metadata_path=str(tmp_path / "metadata"),
        base_path=str(lib_dir),
        node_parser=mock_node_parser,
    )

    handler.process_file(str(test_file))

    # Verify index.refresh_ref_docs was called
    assert mock_index.refresh_ref_docs.called
