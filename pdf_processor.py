"""PDF text extraction and chunking."""

from __future__ import annotations

import hashlib
import os
import tempfile
from typing import Any

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import CHUNK_OVERLAP, CHUNK_SIZE

def clean_text(text: str) -> str:
    """Remove NUL characters, which Postgres text columns cannot store."""
    return (text or "").replace("\x00", "").replace("\u0000", "")
    
def file_hash(data: bytes) -> str:
    """Short stable id for a file, used to detect re-uploads."""
    return hashlib.sha256(data).hexdigest()[:16]


def _splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )


def load_pdf(file_bytes: bytes, filename: str) -> list[Document]:
    """Extract one Document per page from an in-memory PDF."""
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        pages = PyPDFLoader(tmp_path).load()
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    for page in pages:
        # PyPDFLoader pages are 0-indexed; humans count from 1.
        page.page_content = clean_text(page.page_content)
        page.metadata["page"] = int(page.metadata.get("page", 0)) + 1
        page.metadata["source"] = filename
    return pages


def chunk_pdf(
    file_bytes: bytes,
    filename: str,
    storage_path: str | None = None,
    public_url: str | None = None,
) -> list[dict[str, Any]]:
    """Turn a PDF into embed-ready chunk records.

    Returns dicts of {"content": str, "metadata": {...}}. Empty or whitespace-only
    chunks are dropped so we never store rows that cannot be retrieved usefully.
    """
    pages = load_pdf(file_bytes, filename)
    chunks = _splitter().split_documents(pages)

    records: list[dict[str, Any]] = []
    doc_id = file_hash(file_bytes)
    for index, chunk in enumerate(chunks):
        text = clean_text(chunk.page_content).strip()
        if len(text) < 20:
            continue
        records.append(
            {
                "content": text,
                "metadata": {
                    "source": filename,
                    "page": chunk.metadata.get("page", 1),
                    "chunk_index": index,
                    "doc_id": doc_id,
                    "storage_path": storage_path,
                    "public_url": public_url,
                    "char_count": len(text),
                },
            }
        )
    return records


def pdf_stats(records: list[dict[str, Any]]) -> dict[str, int]:
    pages = {r["metadata"]["page"] for r in records}
    return {
        "chunks": len(records),
        "pages": len(pages),
        "characters": sum(r["metadata"]["char_count"] for r in records),
    }
