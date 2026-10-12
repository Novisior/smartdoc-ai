"""Configuration and secret loading for SmartDoc AI.

Secrets are resolved in this order:
1. Values typed into the sidebar (passed in by the caller).
2. `st.secrets` (Streamlit Community Cloud deployment).
3. Environment variables / `.env` file (local development).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# --- Embedding model -------------------------------------------------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# --- Chunking --------------------------------------------------------------
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# --- Supabase --------------------------------------------------------------
TABLE_NAME = "documents"
MATCH_FUNCTION = "match_documents"
STORAGE_BUCKET = "documents"

# --- LLM -------------------------------------------------------------------
AVAILABLE_MODELS = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
]
DEFAULT_MODEL = "llama-3.1-8b-instant"
DEFAULT_TEMPERATURE = 0.1
DEFAULT_TOP_K = 4

# Free-tier guardrails
MAX_FILE_MB = 20
INSERT_BATCH_SIZE = 100


def get_secret(key: str, default: str = "") -> str:
    """Read a secret from Streamlit secrets, falling back to the environment."""
    try:
        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        # st.secrets raises if no secrets.toml exists locally.
        pass
    return os.getenv(key, default)


@dataclass
class Settings:
    """Runtime settings assembled from the sidebar and stored secrets."""

    groq_api_key: str = ""
    supabase_url: str = ""
    supabase_key: str = ""
    model: str = DEFAULT_MODEL
    temperature: float = DEFAULT_TEMPERATURE
    top_k: int = DEFAULT_TOP_K

    @property
    def supabase_ready(self) -> bool:
        return bool(self.supabase_url and self.supabase_key)

    @property
    def groq_ready(self) -> bool:
        return bool(self.groq_api_key)

    def missing(self) -> list[str]:
        gaps = []
        if not self.supabase_url:
            gaps.append("Supabase URL")
        if not self.supabase_key:
            gaps.append("Supabase key")
        if not self.groq_api_key:
            gaps.append("Groq API key")
        return gaps


def load_settings(
    groq_api_key: str = "",
    supabase_url: str = "",
    supabase_key: str = "",
    **kwargs,
) -> Settings:
    """Build a Settings object, preferring sidebar input over stored secrets."""
    return Settings(
        groq_api_key=groq_api_key or get_secret("GROQ_API_KEY"),
        supabase_url=supabase_url or get_secret("SUPABASE_URL"),
        supabase_key=supabase_key or get_secret("SUPABASE_KEY"),
        **kwargs,
    )
