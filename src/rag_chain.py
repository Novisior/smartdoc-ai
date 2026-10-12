"""Retrieval-augmented answering with Groq."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from .vector_store import SupabaseStore, get_embedder

SYSTEM_PROMPT = """You are a study assistant answering from the numbered context passages below.

Rules:
- Base every factual claim on the passages and cite it inline as [1], [2].
- You may organise, compare, summarise or prioritise what the passages say. For judgment questions (easiest topic, most important, what to study first), reason from the passages, for example by pointing to topics that are short definitions or lists, and say clearly that this is your assessment.
- If you add general knowledge that is not in the passages, put it in a final line starting with "Beyond the document:" and do not cite it.
- When passages disagree, say so and cite both.
- Only if the passages are unrelated to the question, reply: "I could not find this in the uploaded documents." Then name what the documents do cover.
- Write in plain prose. Use short bullets only when listing several distinct items.

Math formatting:
- Write inline math with single dollar signs, like $\\phi_A = 1.33$, and display equations with double dollar signs on their own lines, like $$\\phi_i = \\sum_{{S}} w(S)\\,[v(S \\cup \\{{i\\}}) - v(S)]$$.
- Never use \\( \\) or \\[ \\] for math.
- Never put formulas or math inside markdown tables, because the | symbol breaks the table. Use a numbered list instead, with one calculation step per line.
- For worked calculations (such as Shapley values), show each step on its own line and finish with a clear final answer.

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
        max_tokens=2048,
        timeout=60,
        max_retries=2,
    )

_MATH_SPAN = re.compile(r"(\$\$.+?\$\$|\$[^$\n]+?\$)", re.S)
_LATEX_CMD = re.compile(r"\\[A-Za-z]+")


def _wrap_bare_latex(segment: str) -> str:
    """Wrap (...) or [...] groups that contain LaTeX commands in $...$."""
    out, i, n = [], 0, len(segment)
    while i < n:
        ch = segment[i]
        if ch in "([":
            close = ")" if ch == "(" else "]"
            depth, j = 0, i
            while j < n:
                if segment[j] == ch:
                    depth += 1
                elif segment[j] == close:
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            if j < n:
                inner = segment[i + 1 : j]
                if _LATEX_CMD.search(inner) and "\n" not in inner:
                    out.append(f"${inner.strip()}$")
                    i = j + 1
                    continue
        out.append(ch)
        i += 1
    return "".join(out)


def normalize_math(text: str) -> str:
    """Make LaTeX from the model render in Streamlit."""
    # \[ ... \] and \( ... \) -> $$ ... $$ and $ ... $
    text = re.sub(
        r"\\\[(.+?)\\\]",
        lambda m: f"\n$$\n{m.group(1).strip()}\n$$\n",
        text,
        flags=re.S,
    )
    text = re.sub(r"\\\((.+?)\\\)", lambda m: f"${m.group(1).strip()}$", text, flags=re.S)
    # Bare (\phi_A) or [ \phi_A = 1.33 ] that lost its backslashes or delimiters
    parts = _MATH_SPAN.split(text)
    return "".join(p if k % 2 else _wrap_bare_latex(p) for k, p in enumerate(parts))
    
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

    answer = normalize_math(result.content.strip())
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
