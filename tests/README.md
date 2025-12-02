# Tests

This directory contains test files for the langconnect services.

## Test Files

- `test_toc_splitter.py`: Tests for TOC-based section splitter (Markdown and HTML)
- `verify_document_processor.py`: Verification tests for document processor enhancements

## Running Tests

Run tests from the **project root directory**:

```bash
# Run TOC splitter tests
python -m tests.test_toc_splitter

# Run document processor verification
python -m tests.verify_document_processor
```

Or run directly with full path:

```bash
# From project root
python tests/test_toc_splitter.py
python tests/verify_document_processor.py
```

