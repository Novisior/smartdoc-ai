"""SmartDoc AI — RAG document Q&A over Supabase + Groq."""

from .config import Settings, load_settings
from .evaluation import SAMPLE_QUESTIONS, Rating, ratings_frame, record_rating, summarise
from .pdf_processor import chunk_pdf, file_hash, pdf_stats
from .rag_chain import RAGResponse, Source, answer_question, format_citation_block
from .vector_store import SupabaseStore, VectorStoreError, get_embedder

__all__ = [
    "Settings",
    "load_settings",
    "SAMPLE_QUESTIONS",
    "Rating",
    "ratings_frame",
    "record_rating",
    "summarise",
    "chunk_pdf",
    "file_hash",
    "pdf_stats",
    "RAGResponse",
    "Source",
    "answer_question",
    "format_citation_block",
    "SupabaseStore",
    "VectorStoreError",
    "get_embedder",
]
