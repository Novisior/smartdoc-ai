"""Supabase-backed vector store and file storage."""

from __future__ import annotations

from typing import Any

import streamlit as st
from supabase import Client, create_client

from .config import (
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    INSERT_BATCH_SIZE,
    MATCH_FUNCTION,
    STORAGE_BUCKET,
    TABLE_NAME,
)


class VectorStoreError(RuntimeError):
    """Raised when Supabase rejects a read or write."""


@st.cache_resource(show_spinner=False)
def get_embedder(model_name: str = EMBEDDING_MODEL):
    """Load the sentence-transformers model once per session.

    Downloads ~90 MB the first time it runs, then stays in memory.
    """
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:  # older installs
        from langchain_community.embeddings import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


@st.cache_resource(show_spinner=False)
def get_client(url: str, key: str) -> Client:
    return create_client(url, key)


class SupabaseStore:
    """Thin wrapper over the `documents` table, the RPC search, and the bucket."""

    def __init__(self, url: str, key: str):
        self.client = get_client(url, key)

    # --- Storage ----------------------------------------------------------
    def upload_pdf(self, filename: str, data: bytes, doc_id: str) -> tuple[str, str]:
        """Upload the original PDF and return (storage_path, public_url)."""
        path = f"{doc_id}/{filename}"
        try:
            self.client.storage.from_(STORAGE_BUCKET).upload(
                path=path,
                file=data,
                file_options={
                    "content-type": "application/pdf",
                    "upsert": "true",
                },
            )
        except Exception as exc:  # duplicate uploads are not fatal
            if "Duplicate" not in str(exc) and "already exists" not in str(exc):
                raise VectorStoreError(
                    f"Could not upload '{filename}' to the '{STORAGE_BUCKET}' bucket: {exc}"
                ) from exc
        public_url = self.client.storage.from_(STORAGE_BUCKET).get_public_url(path)
        return path, public_url

    def delete_all_files(self) -> int:
        """Remove every object in the bucket. Returns the number removed."""
        bucket = self.client.storage.from_(STORAGE_BUCKET)
        removed = 0
        try:
            folders = bucket.list()
        except Exception as exc:
            raise VectorStoreError(f"Could not list storage objects: {exc}") from exc

        for entry in folders:
            name = entry.get("name")
            if not name:
                continue
            children = bucket.list(name)
            paths = [f"{name}/{c['name']}" for c in children if c.get("name")]
            if paths:
                bucket.remove(paths)
                removed += len(paths)
            elif entry.get("id"):
                bucket.remove([name])
                removed += 1
        return removed

    # --- Vectors ----------------------------------------------------------
    def add_chunks(self, records: list[dict[str, Any]], embeddings: list[list[float]]) -> int:
        if len(records) != len(embeddings):
            raise VectorStoreError("Chunk and embedding counts do not match.")

        rows = [
            {
                "content": record["content"],
                "metadata": record["metadata"],
                "embedding": vector,
            }
            for record, vector in zip(records, embeddings)
        ]
        inserted = 0
        for start in range(0, len(rows), INSERT_BATCH_SIZE):
            batch = rows[start : start + INSERT_BATCH_SIZE]
            try:
                self.client.table(TABLE_NAME).insert(batch).execute()
            except Exception as exc:
                raise VectorStoreError(
                    f"Insert failed at rows {start}-{start + len(batch)}: {exc}"
                ) from exc
            inserted += len(batch)
        return inserted

    def similarity_search(
        self,
        query_embedding: list[float],
        k: int = 4,
        source_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        if len(query_embedding) != EMBEDDING_DIM:
            raise VectorStoreError(
                f"Query embedding has {len(query_embedding)} dimensions, expected {EMBEDDING_DIM}."
            )
        payload = {
            "query_embedding": query_embedding,
            "match_count": k,
            "filter": {"source": source_filter} if source_filter else {},
        }
        try:
            response = self.client.rpc(MATCH_FUNCTION, payload).execute()
        except Exception as exc:
            raise VectorStoreError(
                f"Similarity search failed. Check that the '{MATCH_FUNCTION}' function "
                f"exists in Supabase: {exc}"
            ) from exc
        return response.data or []

    # --- Bookkeeping ------------------------------------------------------
    def count_chunks(self) -> int:
        try:
            response = (
                self.client.table(TABLE_NAME)
                .select("id", count="exact")
                .limit(1)
                .execute()
            )
            return response.count or 0
        except Exception as exc:
            raise VectorStoreError(f"Could not count stored chunks: {exc}") from exc

    def list_sources(self) -> list[dict[str, Any]]:
        """Return one row per indexed file with its chunk count."""
        try:
            response = self.client.table(TABLE_NAME).select("metadata").execute()
        except Exception as exc:
            raise VectorStoreError(f"Could not list documents: {exc}") from exc

        tally: dict[str, dict[str, Any]] = {}
        for row in response.data or []:
            meta = row.get("metadata") or {}
            source = meta.get("source", "unknown")
            entry = tally.setdefault(
                source, {"source": source, "chunks": 0, "pages": set(), "url": meta.get("public_url")}
            )
            entry["chunks"] += 1
            entry["pages"].add(meta.get("page", 1))
        return [
            {"source": v["source"], "chunks": v["chunks"], "pages": len(v["pages"]), "url": v["url"]}
            for v in sorted(tally.values(), key=lambda x: x["source"])
        ]

    def delete_source(self, source: str) -> None:
        try:
            self.client.table(TABLE_NAME).delete().contains(
                "metadata", {"source": source}
            ).execute()
        except Exception as exc:
            raise VectorStoreError(f"Could not delete '{source}': {exc}") from exc

    def delete_all_chunks(self) -> None:
        try:
            self.client.table(TABLE_NAME).delete().gte("id", 0).execute()
        except Exception as exc:
            raise VectorStoreError(f"Could not clear the documents table: {exc}") from exc
