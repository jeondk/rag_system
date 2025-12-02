"""
Verification tests for TOC-based section splitter.
"""
import asyncio
import logging
from unittest.mock import MagicMock

from fastapi import UploadFile
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langconnect.services.document_processor import DocumentProcessor
from langconnect.services.toc_splitter import (
    HTMLTOCSplitter,
    MarkdownTOCSplitter,
    Section,
)

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

# Sample Markdown content with headers
SAMPLE_MARKDOWN = """# Chapter 1: Introduction

This is the introduction section with some content.
It has multiple paragraphs.

## 1.1 Background

Background information goes here.

## 1.2 Objectives

The objectives of this document.

# Chapter 2: Implementation

This section describes the implementation.

## 2.1 Architecture

Architecture details here.

### 2.1.1 Components

Component breakdown.

## 2.2 Testing

Testing approach.
"""

# Sample HTML content
SAMPLE_HTML = """
<html>
<body>
    <h1>Chapter 1: Introduction</h1>
    <p>This is the introduction section.</p>
    
    <h2>1.1 Background</h2>
    <p>Background information.</p>
    
    <h2>1.2 Objectives</h2>
    <p>The objectives.</p>
    
    <h1>Chapter 2: Implementation</h1>
    <p>Implementation details.</p>
    
    <h2>2.1 Architecture</h2>
    <p>Architecture details.</p>
</body>
</html>
"""


def test_markdown_toc_splitter():
    """Test Markdown TOC splitter."""
    LOGGER.info("Testing Markdown TOC Splitter...")

    doc = Document(page_content=SAMPLE_MARKDOWN, metadata={"source": "test.md"})
    splitter = MarkdownTOCSplitter()

    sections = splitter.extract_sections(doc)

    LOGGER.info(f"Extracted {len(sections)} sections")
    for section in sections:
        LOGGER.info(
            f"  - Level {section.level}: {section.title} "
            f"(path: {section.section_path})"
        )

    # Verify sections
    assert len(sections) > 0, "Should extract sections"
    assert sections[0].title == "Chapter 1: Introduction"
    assert sections[0].level == 1
    assert sections[0].section_path == ["Chapter 1: Introduction"]

    # Check hierarchical path
    background_section = next(s for s in sections if "Background" in s.title)
    assert "Chapter 1: Introduction" in background_section.parent_path

    LOGGER.info("✓ Markdown TOC splitter test passed")


def test_html_toc_splitter():
    """Test HTML TOC splitter."""
    LOGGER.info("Testing HTML TOC Splitter...")

    doc = Document(page_content=SAMPLE_HTML, metadata={"source": "test.html"})
    splitter = HTMLTOCSplitter()

    sections = splitter.extract_sections(doc)

    LOGGER.info(f"Extracted {len(sections)} sections")
    for section in sections:
        LOGGER.info(
            f"  - Level {section.level}: {section.title} "
            f"(path: {section.section_path})"
        )

    # Verify sections
    assert len(sections) > 0, "Should extract sections"
    assert any("Chapter 1" in s.title for s in sections)

    LOGGER.info("✓ HTML TOC splitter test passed")


def test_section_chunking():
    """Test TOC splitting with chunking."""
    LOGGER.info("Testing TOC splitting with chunking...")

    doc = Document(page_content=SAMPLE_MARKDOWN, metadata={"source": "test.md"})
    toc_splitter = MarkdownTOCSplitter()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)

    chunked_docs = toc_splitter.split_with_chunking([doc], text_splitter)

    LOGGER.info(f"Created {len(chunked_docs)} chunks")

    # Group chunks by section
    sections_seen = {}
    
    # Verify all chunks have section metadata
    for i, chunk in enumerate(chunked_docs):
        assert "section_title" in chunk.metadata, f"Chunk {i} missing section_title"
        assert "section_level" in chunk.metadata, f"Chunk {i} missing section_level"
        assert "section_path" in chunk.metadata, f"Chunk {i} missing section_path"
        
        section_title = chunk.metadata["section_title"]
        
        # Track if this is the first chunk for this section
        if section_title not in sections_seen:
            sections_seen[section_title] = chunk
            # First chunk of a section should contain the title
            assert section_title in chunk.page_content, (
                f"First chunk of section '{section_title}' doesn't contain title"
            )

        LOGGER.info(
            f"  Chunk {i}: section='{chunk.metadata['section_title']}' "
            f"level={chunk.metadata['section_level']}"
        )

    LOGGER.info(f"✓ Section chunking test passed ({len(sections_seen)} unique sections)")



async def test_document_processor_integration():
    """Test TOC splitting integration with DocumentProcessor."""
    LOGGER.info("Testing DocumentProcessor integration...")

    # Create processor with TOC splitting enabled
    processor = DocumentProcessor(use_toc_splitting=True)

    # Mock file
    mock_file = MagicMock(spec=UploadFile)
    mock_file.filename = "test.md"
    mock_file.content_type = "text/markdown"
    mock_file.read.return_value = SAMPLE_MARKDOWN.encode("utf-8")

    # Process document
    chunks = await processor.process(mock_file, chunk_size=100, chunk_overlap=20)

    LOGGER.info(f"Processed into {len(chunks)} chunks with TOC splitting")

    # Verify chunks have section metadata
    for chunk in chunks:
        assert "section_title" in chunk.metadata
        assert "section_path" in chunk.metadata
        # Also check auto-injected metadata
        assert "filename" in chunk.metadata
        assert "mime_type" in chunk.metadata

    LOGGER.info("✓ DocumentProcessor integration test passed")


async def test_document_processor_without_toc():
    """Test DocumentProcessor without TOC splitting (fallback)."""
    LOGGER.info("Testing DocumentProcessor without TOC splitting...")

    # Create processor with TOC splitting disabled
    processor = DocumentProcessor(use_toc_splitting=False)

    mock_file = MagicMock(spec=UploadFile)
    mock_file.filename = "test.md"
    mock_file.content_type = "text/markdown"
    mock_file.read.return_value = SAMPLE_MARKDOWN.encode("utf-8")

    chunks = await processor.process(mock_file, chunk_size=100, chunk_overlap=20)

    LOGGER.info(f"Processed into {len(chunks)} chunks without TOC splitting")

    # These chunks should NOT have section metadata
    for chunk in chunks:
        assert "section_title" not in chunk.metadata
        # But should still have auto-injected metadata
        assert "filename" in chunk.metadata

    LOGGER.info("✓ DocumentProcessor without TOC test passed")


async def main():
    """Run all tests."""
    LOGGER.info("=" * 60)
    LOGGER.info("TOC Splitter Verification Tests")
    LOGGER.info("=" * 60)

    try:
        # Unit tests
        test_markdown_toc_splitter()
        test_html_toc_splitter()
        test_section_chunking()

        # Integration tests
        await test_document_processor_integration()
        await test_document_processor_without_toc()

        LOGGER.info("=" * 60)
        LOGGER.info("✅ All tests passed!")
        LOGGER.info("=" * 60)

    except AssertionError as e:
        LOGGER.error(f"❌ Test failed: {e}")
        raise
    except Exception as e:
        LOGGER.error(f"❌ Unexpected error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
