import asyncio
import logging
import os
import sys
from unittest.mock import MagicMock

# Debug: Print sys.path
print(f"sys.path: {sys.path}")

try:
    import langconnect.services.document_processor as dp
    print(f"Module file: {dp.__file__}")
    print(f"Module attributes: {dir(dp)}")
    from langconnect.services.document_processor import (
        DocumentProcessor,
        ParserRegistry,
        process_document,
    )
except ImportError as e:
    print(f"ImportError: {e}")
    print(f"Current directory: {os.getcwd()}")
    print(f"Directory contents: {os.listdir('.')}")
    if os.path.exists('langconnect'):
        print(f"langconnect contents: {os.listdir('langconnect')}")
    raise e

from fastapi import UploadFile
from langchain_community.document_loaders.base import BaseBlobParser
from langchain_core.documents import Document

# Configure logging
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


class MockParser(BaseBlobParser):
    """A mock parser for testing."""

    def lazy_parse(self, blob):
        yield Document(page_content="Mock content", metadata={"source": "mock"})


async def verify_document_processor():
    LOGGER.info("Starting verification...")

    # 1. Test Default Functionality via process_document wrapper
    LOGGER.info("Testing default process_document wrapper...")
    mock_file = MagicMock(spec=UploadFile)
    mock_file.filename = "test.txt"
    mock_file.content_type = "text/plain"
    mock_file.read.return_value = b"Hello World"

    docs = await process_document(mock_file)
    assert len(docs) > 0
    assert docs[0].page_content == "Hello World"
    LOGGER.info("Default process_document wrapper passed.")

    # 2. Test ParserRegistry and Custom Parser
    LOGGER.info("Testing ParserRegistry and Custom Parser...")
    registry = ParserRegistry()
    registry.register("application/mock", MockParser())
    
    processor = DocumentProcessor(registry=registry)
    
    mock_custom_file = MagicMock(spec=UploadFile)
    mock_custom_file.filename = "test.mock"
    mock_custom_file.content_type = "application/mock"
    mock_custom_file.read.return_value = b"Mock Data"

    docs = await processor.process(mock_custom_file)
    assert len(docs) > 0
    assert docs[0].page_content == "Mock content"
    LOGGER.info("Custom Parser registration passed.")

    # 3. Test Fallback Mechanism
    LOGGER.info("Testing Fallback Mechanism...")
    registry.set_fallback(MockParser())
    
    mock_unknown_file = MagicMock(spec=UploadFile)
    mock_unknown_file.filename = "unknown.xyz"
    mock_unknown_file.content_type = "application/xyz"
    mock_unknown_file.read.return_value = b"Unknown Data"

    # Note: MimeTypeBasedParser usually raises an error if no handler is found.
    # Our DocumentProcessor catches this and tries the fallback.
    docs = await processor.process(mock_unknown_file)
    assert len(docs) > 0
    assert docs[0].page_content == "Mock content"
    LOGGER.info("Fallback Mechanism passed.")

    LOGGER.info("All verification tests passed!")

if __name__ == "__main__":
    asyncio.run(verify_document_processor())
