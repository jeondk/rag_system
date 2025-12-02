import importlib
import json
import logging
import uuid
from typing import Any, Callable, Dict, List, Optional, Type, Union

import yaml
from fastapi import UploadFile
from langchain_community.document_loaders.base import BaseBlobParser
from langchain_community.document_loaders.parsers import (
    BS4HTMLParser,
    PDFPlumberParser,
)
from langchain_community.document_loaders.parsers.generic import MimeTypeBasedParser
from langchain_community.document_loaders.parsers.msword import MsWordParser
from langchain_community.document_loaders.parsers.txt import TextParser
from langchain_core.documents.base import Blob, Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

LOGGER = logging.getLogger(__name__)

# Default max file size: 100MB
DEFAULT_MAX_FILE_SIZE = 100 * 1024 * 1024


class ParserRegistry:
    """Registry for document parsers with immutable handler exposure."""

    def __init__(self):
        self._handlers: Dict[str, BaseBlobParser] = {
            "application/pdf": PDFPlumberParser(),
            "text/plain": TextParser(),
            "text/html": BS4HTMLParser(),
            "text/markdown": TextParser(),
            "text/x-markdown": TextParser(),
            "application/msword": MsWordParser(),
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": (
                MsWordParser()
            ),
        }
        self._fallback_parser: Optional[BaseBlobParser] = None

    def register(self, mime_type: str, parser: BaseBlobParser) -> None:
        """Register a parser for a specific mime type."""
        self._handlers[mime_type] = parser
        LOGGER.info(f"Registered parser {parser} for mime type {mime_type}")

    def get(self, mime_type: str) -> Optional[BaseBlobParser]:
        """Get a parser for a specific mime type."""
        return self._handlers.get(mime_type)

    def set_fallback(self, parser: BaseBlobParser) -> None:
        """Set a fallback parser."""
        self._fallback_parser = parser
        LOGGER.info(f"Set fallback parser to {parser}")

    def get_handlers_snapshot(self) -> Dict[str, BaseBlobParser]:
        """Get an immutable snapshot of registered handlers."""
        return self._handlers.copy()

    def load_from_config(self, config_path: str) -> None:
        """
        Load parser configuration from a YAML or JSON file.
        
        Expected config format:
        ```yaml
        parsers:
          text/plain:
            class: "langchain_community.document_loaders.parsers.txt.TextParser"
            kwargs: {}
          application/pdf:
            class: "langchain_community.document_loaders.parsers.PDFPlumberParser"
            kwargs: {}
        fallback:
          class: "langchain_community.document_loaders.parsers.txt.TextParser"
          kwargs: {}
        ```
        """
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                if config_path.endswith(".yaml") or config_path.endswith(".yml"):
                    config = yaml.safe_load(f)
                elif config_path.endswith(".json"):
                    config = json.load(f)
                else:
                    raise ValueError("Unsupported config file format")

            # Load parsers
            if "parsers" in config:
                for mime_type, parser_config in config["parsers"].items():
                    parser = self._instantiate_parser(parser_config)
                    if parser:
                        self.register(mime_type, parser)

            # Load fallback parser
            if "fallback" in config:
                fallback = self._instantiate_parser(config["fallback"])
                if fallback:
                    self.set_fallback(fallback)

            LOGGER.info(f"Successfully loaded configuration from {config_path}")

        except Exception as e:
            LOGGER.error(f"Failed to load config from {config_path}: {e}")
            raise

    def _instantiate_parser(
        self, parser_config: Dict[str, Any]
    ) -> Optional[BaseBlobParser]:
        """Dynamically instantiate a parser from config."""
        try:
            class_path = parser_config.get("class")
            kwargs = parser_config.get("kwargs", {})

            if not class_path:
                LOGGER.error("Parser config missing 'class' field")
                return None

            # Split module path and class name
            module_path, class_name = class_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            parser_class = getattr(module, class_name)

            # Instantiate parser
            return parser_class(**kwargs)

        except Exception as e:
            LOGGER.error(f"Failed to instantiate parser from {parser_config}: {e}")
            return None

    @property
    def handlers(self) -> Dict[str, BaseBlobParser]:
        """Get handlers (returns a copy for immutability)."""
        return self.get_handlers_snapshot()

    @property
    def fallback_parser(self) -> Optional[BaseBlobParser]:
        return self._fallback_parser


class DocumentProcessor:
    """Processor for handling document uploads and parsing."""

    def __init__(
        self,
        registry: Optional[ParserRegistry] = None,
        max_file_size: Optional[int] = DEFAULT_MAX_FILE_SIZE,
        splitter_factory: Optional[Callable[[int, int], Any]] = None,
        use_toc_splitting: bool = False,
        toc_splitter_factory: Optional[Callable[[str], Any]] = None,
    ):
        """
        Initialize DocumentProcessor.
        
        Args:
            registry: Parser registry instance
            max_file_size: Maximum allowed file size in bytes (None = no limit)
            splitter_factory: Factory function to create text splitters
                             Signature: (chunk_size, chunk_overlap) -> TextSplitter
            use_toc_splitting: Enable TOC-based section splitting
            toc_splitter_factory: Factory function to create TOC splitters
                                 Signature: (mime_type) -> TOCSplitter or None
        """
        self.registry = registry or ParserRegistry()
        self.max_file_size = max_file_size
        self.splitter_factory = splitter_factory or self._default_splitter_factory
        self.use_toc_splitting = use_toc_splitting
        self.toc_splitter_factory = (
            toc_splitter_factory or self._default_toc_splitter_factory
        )

    def _default_splitter_factory(
        self, chunk_size: int, chunk_overlap: int
    ) -> RecursiveCharacterTextSplitter:
        """Default text splitter factory."""
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    def _default_toc_splitter_factory(self, mime_type: str) -> Optional[Any]:
        """
        Default TOC splitter factory based on mime type.
        
        Args:
            mime_type: MIME type of the document
            
        Returns:
            TOCSplitter instance or None if not supported
        """
        try:
            from langconnect.services.toc_splitter import (
                HTMLTOCSplitter,
                MarkdownTOCSplitter,
            )

            if mime_type in ["text/markdown", "text/x-markdown"]:
                return MarkdownTOCSplitter()
            elif mime_type == "text/html":
                return HTMLTOCSplitter()
            # PDF/Word not supported yet
            return None
        except ImportError as e:
            LOGGER.warning(f"Failed to import TOC splitter: {e}")
            return None

    async def process(
        self,
        file: UploadFile,
        metadata: Optional[dict] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> List[Document]:
        """Process an uploaded file into LangChain documents."""
        file_id = uuid.uuid4()
        contents = await file.read()
        file_size = len(contents)

        # File size validation
        if self.max_file_size and file_size > self.max_file_size:
            raise ValueError(
                f"File size ({file_size} bytes) exceeds maximum allowed size "
                f"({self.max_file_size} bytes)"
            )

        mime_type = self._determine_mime_type(file)

        blob = Blob(data=contents, mimetype=mime_type)

        # Create parser per request for concurrency safety
        parser = MimeTypeBasedParser(
            handlers=self.registry.get_handlers_snapshot(),
            fallback_parser=self.registry.fallback_parser,
        )

        try:
            docs = parser.parse(blob)
        except Exception as e:
            LOGGER.warning(
                f"Parsing failed for {file.filename} ({mime_type}): {e}. "
                "Attempting fallback if available."
            )
            if self.registry.fallback_parser:
                docs = self.registry.fallback_parser.parse(blob)
            else:
                raise e

        # Auto-inject basic metadata
        auto_metadata = {
            "filename": file.filename or "unknown",
            "mime_type": mime_type,
            "file_size": file_size,
        }

        self._enrich_metadata(docs, auto_metadata)
        self._enrich_metadata(docs, metadata)

        # Use splitter factory
        text_splitter = self.splitter_factory(chunk_size, chunk_overlap)

        # Apply TOC splitting if enabled
        if self.use_toc_splitting:
            toc_splitter = self.toc_splitter_factory(mime_type)
            if toc_splitter:
                LOGGER.debug(f"Using TOC splitter for {mime_type}")
                split_docs = toc_splitter.split_with_chunking(docs, text_splitter)
            else:
                LOGGER.debug(f"No TOC splitter available for {mime_type}, using regular chunking")
                split_docs = text_splitter.split_documents(docs)
        else:
            split_docs = text_splitter.split_documents(docs)

        self._add_file_id(split_docs, file_id)

        return split_docs

    def _determine_mime_type(self, file: UploadFile) -> str:
        """Determine the mime type of the file."""
        mime_type = file.content_type or "text/plain"

        if mime_type == "application/octet-stream" and file.filename:
            filename_lower = file.filename.lower()
            if filename_lower.endswith((".md", ".markdown")):
                return "text/markdown"
            elif filename_lower.endswith(".txt"):
                return "text/plain"
            elif filename_lower.endswith((".html", ".htm")):
                return "text/html"
            elif filename_lower.endswith(".pdf"):
                return "application/pdf"
            elif filename_lower.endswith(".doc"):
                return "application/msword"
            elif filename_lower.endswith(".docx"):
                return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

        return mime_type

    def _enrich_metadata(self, docs: List[Document], metadata: Optional[dict]) -> None:
        """Add provided metadata to documents."""
        if metadata:
            for doc in docs:
                if not hasattr(doc, "metadata") or not isinstance(doc.metadata, dict):
                    doc.metadata = {}
                doc.metadata.update(metadata)

    def _add_file_id(self, docs: List[Document], file_id: uuid.UUID) -> None:
        """Add file_id to document metadata."""
        for doc in docs:
            if not hasattr(doc, "metadata") or not isinstance(doc.metadata, dict):
                doc.metadata = {}
            doc.metadata["file_id"] = str(file_id)


# Global instance for backward compatibility
_processor = DocumentProcessor()


async def process_document(
    file: UploadFile,
    metadata: Optional[dict] = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Document]:
    """
    Process an uploaded file into LangChain documents.
    Wrapper around DocumentProcessor for backward compatibility.
    """
    return await _processor.process(
        file, metadata, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
