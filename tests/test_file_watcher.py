import os
import pytest
from unittest.mock import MagicMock, patch
from zettlr_rag.file_watcher import NewPaperHandler

def test_process_new_file():
    path = "/Users/awhitworth/Library/CloudStorage/ProtonDrive-whitworth.alex@protonmail.com-folder/Zettlr-Papers/data_methods/imputation/poisson"
    # Find a markdown file in that directory
    files = [f for f in os.listdir(path) if f.endswith(".md")]
    if not files:
        pytest.skip(f"No markdown files found in {path}")
    
    test_file = os.path.join(path, files[0])
    
    mock_index = MagicMock()
    mock_node_parser = MagicMock()
    mock_node_parser.get_nodes_from_documents.return_value = [MagicMock()]
    
    handler = NewPaperHandler(index=mock_index, node_parser=mock_node_parser)
    handler.process_new_file(test_file)
    
    # Verify node parser was called
    assert mock_node_parser.get_nodes_from_documents.called
    # Verify index.insert_nodes was called
    assert mock_index.insert_nodes.called
