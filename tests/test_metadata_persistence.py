from llama_index.core import Document

from zettlr_rag.rag_setup import process_documents_metadata


def test_process_documents_metadata_extraction(tmp_path):
    """Test that YAML metadata is extracted and content is cleaned."""
    # Create a dummy file with YAML frontmatter
    category_dir = tmp_path / "machine_learning"
    category_dir.mkdir()
    file_path = category_dir / "paper_2024.md"

    yaml_content = """---
title: "Test Paper"
year: 2024
category: "ml_custom"
tags: ["test", "rag"]
---
# Actual Content
This is the body of the paper."""

    file_path.write_text(yaml_content)

    # Create a LlamaIndex Document as SimpleDirectoryReader would
    doc = Document(
        text=yaml_content, metadata={"file_path": str(file_path), "file_name": "paper_2024.md"}
    )

    # Process it
    processed_docs = process_documents_metadata([doc], str(tmp_path))
    processed_doc = processed_docs[0]

    # 1. Check Stable ID
    assert processed_doc.id_ == str(file_path)

    # 2. Check Content Cleaning (no YAML in text)
    assert "---" not in processed_doc.text
    assert 'title: "Test Paper"' not in processed_doc.text
    assert "# Actual Content" in processed_doc.text

    # 3. Check Metadata Extraction
    assert processed_doc.metadata["year"] == 2024
    assert processed_doc.metadata["category"] == "ml_custom"
    assert "test" in processed_doc.metadata["tags"]


def test_process_documents_metadata_fallbacks(tmp_path):
    """Test that category and year fallbacks work when YAML is missing."""
    # Case: No YAML frontmatter, file in a subfolder
    category_dir = tmp_path / "causal_inference"
    category_dir.mkdir()
    file_path = category_dir / "experiment_2023.md"

    content = "# Just Content\nNo YAML here."
    file_path.write_text(content)

    doc = Document(
        text=content, metadata={"file_path": str(file_path), "file_name": "experiment_2023.md"}
    )

    processed_docs = process_documents_metadata([doc], str(tmp_path))
    processed_doc = processed_docs[0]

    # 1. Check Stable ID
    assert processed_doc.id_ == str(file_path)

    # 2. Check Fallback Category (from folder name)
    assert processed_doc.metadata["category"] == "causal_inference"

    # 3. Check Fallback Year (from filename)
    assert processed_doc.metadata["year"] == 2023


def test_id_consistency_across_runs(tmp_path):
    """Ensure that the ID remains identical even if content changes slightly."""
    file_path = tmp_path / "persistent_id.md"
    file_path.write_text("---\nyear: 2025\n---\nContent v1")

    doc1 = Document(
        text=file_path.read_text(),
        metadata={"file_path": str(file_path), "file_name": "persistent_id.md"},
    )

    processed1 = process_documents_metadata([doc1], str(tmp_path))[0]
    id1 = processed1.id_

    # Simulate a file modification
    file_path.write_text("---\nyear: 2025\n---\nContent v2 (Modified)")

    doc2 = Document(
        text=file_path.read_text(),
        metadata={"file_path": str(file_path), "file_name": "persistent_id.md"},
    )

    processed2 = process_documents_metadata([doc2], str(tmp_path))[0]
    id2 = processed2.id_

    assert id1 == id2
    assert id1 == str(file_path)
    assert processed2.text.strip() == "Content v2 (Modified)"
