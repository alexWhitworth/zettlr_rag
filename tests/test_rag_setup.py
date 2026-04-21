import os
import pytest
from unittest.mock import patch, MagicMock
from zettlr_rag.rag_setup import setup_settings, load_academic_markdown, main_async

@pytest.mark.asyncio
async def test_load_academic_markdown():
    path = "/Users/awhitworth/Library/CloudStorage/ProtonDrive-whitworth.alex@protonmail.com-folder/Zettlr-Papers/data_methods/surveys"
    documents = load_academic_markdown(path)
    assert len(documents) > 0
    # Check if we got markdown files
    assert all(doc.doc_id for doc in documents)

from llama_index.core.llms.mock import MockLLM
from llama_index.core.embeddings.mock_embed_model import MockEmbedding

@pytest.mark.asyncio
async def test_main_async_survey(temp_chroma_db, temp_metadata_path):
    path = "/Users/awhitworth/Library/CloudStorage/ProtonDrive-whitworth.alex@protonmail.com-folder/Zettlr-Papers/data_methods/surveys"
    
    # Mock GoogleGenAI and RateLimitedEmbedding to return MockLLM and MockEmbedding
    with patch("zettlr_rag.rag_setup.GoogleGenAI") as mock_llm_class, \
         patch("zettlr_rag.rag_setup.RateLimitedEmbedding") as mock_embed_class:
        
        mock_llm_class.return_value = MockLLM()
        mock_embed_class.return_value = MockEmbedding(embed_dim=768)
        
        await main_async(
            base_path=path,
            chroma_path=temp_chroma_db,
            metadata_path=temp_metadata_path,
            run_verification=False
        )
        
        # Verify indexing happened
        assert os.path.exists(temp_chroma_db)
        assert os.path.exists(temp_metadata_path)
        assert len(os.listdir(temp_metadata_path)) > 0
