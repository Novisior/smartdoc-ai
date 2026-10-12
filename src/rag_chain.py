"""Retrieval-augmented answering with Groq."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from .vector_store import SupabaseStore, get_embedder

SYSTEM_PROMPT = """You answer questions using only the numbered context passages below.

Rules:
- Use only what the passages say. Do not add outside knowledge.
- Cite the passage number inline as [1], [2] after each claim it supports.
- When passages disagree, say so and cite both.
- If the passages do not contain the answer, reply exactly: "I could not find this in the uploaded documents." Then name what the documents do cover.
- Write in plain prose. Use short bullets only when listing several distinct items.

Context passages:
{context}"""

USER_PROMPT = """Question: {question}

Answer with inline citations."""


@dataclass
class Source:
    number: int
    filename: str
    page: int
    snippet: str
    similarity: float
    url: str | None = None

    @property
    def label(self) -> str:
        return f"[{self.number}] {self.filename} — page {self.page}"


@dataclass
class RAGResponse:
    answer: str
    sources: list[Source] = field(default_factory=list)
    cited: list[int] = field(default_factory=list)
    latency_s: float = 0.0
    model: str = ""

    @property
    def cited_sources(self) -> list[Source]:
        if not self.cited:
            return self.sources
        return [s for s in self.sources if s.number in self.cited]


def _snippet(text: str, limit: int = 320) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + "…"


def build_sources(matches: list[dict[str, Any]]) -> list[Source]:
    sources = []
    for i, match in enumerate(matches, start=1):
        meta = match.get("metadata") or {}
        sources.append(
            Source(
                number=i,
                filename=meta.get("source", "unknown.pdf"),
                page=int(meta.get("page", 1)),
                snippet=_snippet(match.get("content", "")),
                similarity=round(float(match.get("similarity", 0.0)), 3),
                url=meta.get("public_url"),
            )
        )
    return sources


def format_context(matches: list[dict[str, Any]]) -> str:
    blocks = []
    for i, match in enumerate(matches, start=1):
        meta = match.get("metadata") or {}
        header = f"[{i}] {meta.get('source', 'unknown.pdf')}, page {meta.get('page', 1)}"
        blocks.append(f"{header}\n{match.get('content', '').strip()}")
    return "\n\n---\n\n".join(blocks)


def extract_citations(answer: str, max_number: int) -> list[int]:
    found = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
    return sorted(n for n in found if 1 <= n <= max_number)


def get_llm(api_key: str, model: str, temperature: float = 0.1) -> ChatGroq:
    return ChatGroq(
        groq_api_key=api_key,
        model_name=model,
        temperature=temperature,
        max_tokens=1024,
        timeout=60,
        max_retries=2,
    )


def answer_question(
    question: str,
    store: SupabaseStore,
    groq_api_key: str,
    model: str,
    temperature: float = 0.1,
    top_k: int = 4,
    source_filter: str | None = None,
) -> RAGResponse:
    """Retrieve context, ask Groq, and return the answer with its citations."""
    started = time.time()

    embedder = get_embedder()
    query_vector = embedder.embed_query(question)
    matches = store.similarity_search(query_vector, k=top_k, source_filter=source_filter)

    if not matches:
        return RAGResponse(
            answer=(
                "Nothing in the index matched that question. Upload a PDF on the "
                "Documents tab, or try wording the question with terms that appear "
                "in your files."
            ),
            latency_s=round(time.time() - started, 2),
            model=model,
        )

    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", USER_PROMPT)]
    )
    chain = prompt | get_llm(groq_api_key, model, temperature)
    result = chain.invoke({"context": format_context(matches), "question": question})

    answer = result.content.strip()
    sources = build_sources(matches)
    return RAGResponse(
        answer=answer,
        sources=sources,
        cited=extract_citations(answer, len(sources)),
        latency_s=round(time.time() - started, 2),
        model=model,
    )


def format_citation_block(response: RAGResponse) -> str:
    """Markdown list of the sources the answer actually used."""
    if not response.sources:
        return ""
    lines = []
    for source in response.cited_sources:
        lines.append(f"- **{source.label}** · match {source.similarity:.2f}")
    return "\n".join(lines)
