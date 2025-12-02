import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from bs4 import BeautifulSoup
from langchain_core.documents import Document

LOGGER = logging.getLogger(__name__)


@dataclass
class Section:
    """Represents a document section extracted from TOC/headers."""

    title: str
    level: int  # 1 for h1/#, 2 for h2/##, etc.
    content: str
    parent_path: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def section_path(self) -> List[str]:
        """Get full hierarchical path including this section."""
        return self.parent_path + [self.title]


class TOCSplitter(ABC):
    """Abstract base class for TOC-based document splitters."""

    @abstractmethod
    def extract_sections(self, doc: Document) -> List[Section]:
        """
        Extract sections from a document based on TOC/headers.

        Args:
            doc: LangChain Document to extract sections from

        Returns:
            List of Section objects
        """
        pass

    def split_with_chunking(
        self,
        docs: List[Document],
        text_splitter,
    ) -> List[Document]:
        """
        Two-stage splitting: extract sections, then chunk each section.

        Args:
            docs: List of documents to process
            text_splitter: Text splitter to use for chunking sections

        Returns:
            List of chunked documents with section metadata
        """
        result_docs = []

        for doc in docs:
            sections = self.extract_sections(doc)

            if not sections:
                # No sections found, use regular chunking
                LOGGER.debug(f"No sections found in document, using regular chunking")
                chunked = text_splitter.split_documents([doc])
                result_docs.extend(chunked)
                continue

            # Process each section
            for section in sections:
                # Create a temporary document for this section
                section_doc = Document(
                    page_content=section.content,
                    metadata={**doc.metadata, **section.metadata},
                )

                # Chunk the section
                chunked_sections = text_splitter.split_documents([section_doc])

                # Add section metadata to each chunk
                for chunk in chunked_sections:
                    chunk.metadata.update(
                        {
                            "section_title": section.title,
                            "section_level": section.level,
                            "section_path": section.section_path,
                        }
                    )

                result_docs.extend(chunked_sections)

        return result_docs


class MarkdownTOCSplitter(TOCSplitter):
    """TOC splitter for Markdown documents."""

    # Regex to match Markdown headers: ^#{1,6}\s+(.+)$
    HEADER_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

    def extract_sections(self, doc: Document) -> List[Section]:
        """
        Extract sections from Markdown document based on # headers.

        Returns:
            List of Section objects with hierarchical structure
        """
        content = doc.page_content
        sections = []

        # Find all headers
        matches = list(self.HEADER_PATTERN.finditer(content))

        if not matches:
            return []

        # Track parent hierarchy
        parent_stack: List[tuple[int, str]] = []  # (level, title)

        for i, match in enumerate(matches):
            hashes = match.group(1)
            title = match.group(2).strip()
            level = len(hashes)

            # Determine section body content (excluding the header line itself)
            start_pos = match.end()
            end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(content)
            section_body = content[start_pos:end_pos].strip()

            # Build parent path
            # Remove items from stack that are at same or deeper level
            while parent_stack and parent_stack[-1][0] >= level:
                parent_stack.pop()

            parent_path = [title for _, title in parent_stack]

            # Include title in content
            section_content = f"{title}\n\n{section_body}" if section_body else title

            # Create section
            section = Section(
                title=title,
                level=level,
                content=section_content,
                parent_path=parent_path,
                metadata=doc.metadata.copy(),
            )
            sections.append(section)

            # Add current section to parent stack
            parent_stack.append((level, title))

        return sections


class HTMLTOCSplitter(TOCSplitter):
    """TOC splitter for HTML documents."""

    def extract_sections(self, doc: Document) -> List[Section]:
        """
        Extract sections from HTML document based on <h1> through <h6> tags.

        Returns:
            List of Section objects with hierarchical structure
        """
        content = doc.page_content

        try:
            soup = BeautifulSoup(content, "html.parser")
        except Exception as e:
            LOGGER.error(f"Failed to parse HTML: {e}")
            return []

        # Find all header tags
        headers = soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])

        if not headers:
            return []

        sections = []
        parent_stack: List[tuple[int, str]] = []  # (level, title)

        for i, header in enumerate(headers):
            title = header.get_text().strip()
            level = int(header.name[1])  # h1 -> 1, h2 -> 2, etc.

            # Find content from current header to next header using document order
            section_content_parts = []
            
            # Start from elements after the current header
            for element in header.next_elements:
                # Skip the header itself and its descendants (to avoid duplicate title text)
                if element == header or element in header.descendants:
                    continue
                
                # Stop if we encounter the next header
                if hasattr(element, 'name') and element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                    # Check if this is actually the next header in our list
                    if i + 1 < len(headers) and element == headers[i + 1]:
                        break
                    # Or if it's any header tag, stop (covers nested/unexpected headers)
                    elif element in headers[i+1:]:
                        break
                
                # Collect text content
                if isinstance(element, str):
                    text = element.strip()
                    if text:
                        section_content_parts.append(text)

            section_body = " ".join(section_content_parts).strip()

            # Build parent path
            while parent_stack and parent_stack[-1][0] >= level:
                parent_stack.pop()

            parent_path = [title for _, title in parent_stack]

            # Include title in content
            section_content = f"{title}\n\n{section_body}" if section_body else title

            # Create section
            section = Section(
                title=title,
                level=level,
                content=section_content,
                parent_path=parent_path,
                metadata=doc.metadata.copy(),
            )
            sections.append(section)

            # Add to parent stack
            parent_stack.append((level, title))

        return sections
