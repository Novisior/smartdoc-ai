"""SmartDoc AI — ask questions across your PDFs and get cited answers."""

from __future__ import annotations

import sys
from pathlib import Path

# Make sure "src" is importable no matter which directory `streamlit run`
# was launched from (Windows in particular can otherwise resolve the
# working directory differently from the script's own folder).
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from src.config import (
    AVAILABLE_MODELS,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DEFAULT_MODEL,
    MAX_FILE_MB,
    STORAGE_BUCKET,
    load_settings,
)
from src.evaluation import SAMPLE_QUESTIONS, Rating, ratings_frame, record_rating, summarise
from src.pdf_processor import chunk_pdf, file_hash, pdf_stats
from src.rag_chain import answer_question
from src.vector_store import SupabaseStore, VectorStoreError, get_embedder

st.set_page_config(
    page_title="SmartDoc AI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

STYLE = """
<style>
    .block-container { padding-top: 2.2rem; max-width: 1150px; }
    h1, h2, h3 { letter-spacing: -0.015em; }
    .sd-tag {
        display: inline-block; padding: 0.15rem 0.55rem; border-radius: 4px;
        background: #EAF2EF; color: #1F6F5C; font-size: 0.78rem;
        font-weight: 600; margin-right: 0.35rem;
    }
    .sd-snippet {
        border-left: 3px solid #1F6F5C; padding: 0.45rem 0.8rem; margin: 0.4rem 0;
        background: #F7F8F7; font-size: 0.88rem; line-height: 1.5; color: #333;
    }
    .sd-lead { color: #5B6660; font-size: 0.95rem; margin-top: -0.6rem; }
    div[data-testid="stMetricValue"] { font-size: 1.5rem; }
</style>
"""
st.markdown(STYLE, unsafe_allow_html=True)


# ---------------------------------------------------------------- state ----
def init_state() -> None:
    defaults = {
        "messages": [],
        "ratings": [],
        "indexed_files": {},
        "pending_question": "",
        "last_eval": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


init_state()


@st.cache_data(ttl=20, show_spinner=False)
def cached_sources(url: str, key: str, nonce: int) -> list[dict]:
    """Source list for the sidebar and document tab. `nonce` busts the cache."""
    return SupabaseStore(url, key).list_sources()


def bump_cache() -> None:
    st.session_state["cache_nonce"] = st.session_state.get("cache_nonce", 0) + 1


st.session_state.setdefault("cache_nonce", 0)


# -------------------------------------------------------------- sidebar ----
def render_sidebar():
    with st.sidebar:
        st.markdown("### SmartDoc AI")
        st.caption("Cited answers from your own PDFs.")

        with st.expander("Connection", expanded=False):
            st.caption(
                "Leave these blank to use the values in your .env file or Streamlit secrets."
            )
            groq_key = st.text_input("Groq API key", type="password", key="in_groq")
            supabase_url = st.text_input("Supabase project URL", key="in_url")
            supabase_key = st.text_input("Supabase anon key", type="password", key="in_key")

        model = st.selectbox(
            "Model",
            AVAILABLE_MODELS,
            index=AVAILABLE_MODELS.index(DEFAULT_MODEL),
            help="8b is faster and cheaper on rate limits; 70b reasons better over long context.",
        )
        top_k = st.slider("Passages per answer", 2, 10, 4)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.05)

        settings = load_settings(
            groq_api_key=groq_key,
            supabase_url=supabase_url,
            supabase_key=supabase_key,
            model=model,
            temperature=temperature,
            top_k=top_k,
        )

        st.divider()
        if settings.supabase_ready:
            try:
                sources = cached_sources(
                    settings.supabase_url,
                    settings.supabase_key,
                    st.session_state["cache_nonce"],
                )
                left, right = st.columns(2)
                left.metric("Files", len(sources))
                right.metric("Chunks", sum(s["chunks"] for s in sources))
            except VectorStoreError as exc:
                st.error(str(exc))
        else:
            st.info("Add your keys to start indexing.")

        st.divider()
        col_a, col_b = st.columns(2)
        if col_a.button("Clear chat", use_container_width=True):
            st.session_state["messages"] = []
            st.rerun()
        if col_b.button("Clear documents", use_container_width=True, type="secondary"):
            st.session_state["confirm_wipe"] = True

        if st.session_state.get("confirm_wipe"):
            st.warning("This deletes every chunk and every stored PDF.")
            yes, no = st.columns(2)
            if yes.button("Delete everything", use_container_width=True):
                try:
                    store = SupabaseStore(settings.supabase_url, settings.supabase_key)
                    store.delete_all_chunks()
                    store.delete_all_files()
                    st.session_state["indexed_files"] = {}
                    st.session_state["confirm_wipe"] = False
                    bump_cache()
                    st.success("Index and storage cleared.")
                    st.rerun()
                except VectorStoreError as exc:
                    st.error(str(exc))
            if no.button("Keep them", use_container_width=True):
                st.session_state["confirm_wipe"] = False
                st.rerun()

        st.divider()
        st.caption(
            f"Chunks of {CHUNK_SIZE} characters with {CHUNK_OVERLAP} overlap · "
            "all-MiniLM-L6-v2 (384d) · Groq"
        )
    return settings


settings = render_sidebar()


def get_store() -> SupabaseStore | None:
    if not settings.supabase_ready:
        return None
    try:
        return SupabaseStore(settings.supabase_url, settings.supabase_key)
    except Exception as exc:
        st.error(f"Could not reach Supabase: {exc}")
        return None


# ---------------------------------------------------------------- header ---
st.title("SmartDoc AI")
st.markdown(
    '<p class="sd-lead">Upload PDFs, then ask questions. Every answer points back to the '
    "file and page it came from.</p>",
    unsafe_allow_html=True,
)

missing = settings.missing()
if missing:
    st.warning(
        "Missing " + ", ".join(missing) + ". Add them in the sidebar under Connection, "
        "or set them in `.env` / Streamlit secrets."
    )

tab_docs, tab_chat, tab_eval, tab_setup = st.tabs(
    ["Documents", "Ask", "Evaluation", "Setup"]
)


# ------------------------------------------------------------ documents ----
with tab_docs:
    st.subheader("Add documents")
    uploads = st.file_uploader(
        "PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help=f"Up to {MAX_FILE_MB} MB per file.",
    )

    col_run, col_note = st.columns([1, 3])
    start = col_run.button(
        "Index documents",
        type="primary",
        disabled=not uploads or not settings.supabase_ready,
        use_container_width=True,
    )
    if uploads:
        col_note.caption(
            f"{len(uploads)} file(s) selected. Indexing uploads the original to the "
            f"'{STORAGE_BUCKET}' bucket and stores each chunk with its page number."
        )

    if start:
        store = get_store()
        if store:
            progress = st.progress(0.0, text="Loading the embedding model…")
            try:
                embedder = get_embedder()
            except Exception as exc:
                progress.empty()
                st.error(f"Could not load the embedding model: {exc}")
                embedder = None

            if embedder:
                total = len(uploads)
                summary_rows = []
                for i, upload in enumerate(uploads):
                    name = upload.name
                    step = i / total
                    progress.progress(step, text=f"Reading {name}…")
                    data = upload.getvalue()

                    if len(data) > MAX_FILE_MB * 1024 * 1024:
                        st.error(f"{name} is larger than {MAX_FILE_MB} MB. Skipped.")
                        continue

                    doc_id = file_hash(data)
                    try:
                        storage_path, public_url = store.upload_pdf(name, data, doc_id)

                        progress.progress(step + 0.25 / total, text=f"Splitting {name}…")
                        records = chunk_pdf(data, name, storage_path, public_url)
                        if not records:
                            st.error(
                                f"{name} produced no readable text. It may be a scanned "
                                "PDF — run OCR on it first."
                            )
                            continue

                        progress.progress(
                            step + 0.5 / total,
                            text=f"Embedding {len(records)} chunks from {name}…",
                        )
                        vectors = embedder.embed_documents([r["content"] for r in records])

                        progress.progress(step + 0.8 / total, text=f"Saving {name}…")
                        store.add_chunks(records, vectors)

                        stats = pdf_stats(records)
                        stats["file"] = name
                        summary_rows.append(stats)
                        st.session_state["indexed_files"][name] = doc_id
                    except VectorStoreError as exc:
                        st.error(str(exc))
                    except Exception as exc:
                        st.error(f"{name} failed: {exc}")

                progress.progress(1.0, text="Done")
                bump_cache()
                if summary_rows:
                    st.success(
                        f"Indexed {len(summary_rows)} file(s) · "
                        f"{sum(r['chunks'] for r in summary_rows)} chunks."
                    )
                    st.dataframe(
                        [
                            {
                                "File": r["file"],
                                "Pages": r["pages"],
                                "Chunks": r["chunks"],
                                "Characters": r["characters"],
                            }
                            for r in summary_rows
                        ],
                        use_container_width=True,
                        hide_index=True,
                    )

    st.divider()
    st.subheader("Indexed library")
    if not settings.supabase_ready:
        st.info("Connect Supabase to see what is already indexed.")
    else:
        try:
            sources = cached_sources(
                settings.supabase_url, settings.supabase_key, st.session_state["cache_nonce"]
            )
        except VectorStoreError as exc:
            sources = []
            st.error(str(exc))

        if not sources:
            st.info("Nothing indexed yet. Upload a PDF above to get started.")
        for source in sources:
            row = st.columns([5, 1.2, 1.2, 1.3])
            row[0].markdown(f"**{source['source']}**")
            row[1].markdown(f"<span class='sd-tag'>{source['pages']} pages</span>", unsafe_allow_html=True)
            row[2].markdown(f"<span class='sd-tag'>{source['chunks']} chunks</span>", unsafe_allow_html=True)
            if row[3].button("Remove", key=f"rm_{source['source']}", use_container_width=True):
                store = get_store()
                if store:
                    try:
                        store.delete_source(source["source"])
                        bump_cache()
                        st.rerun()
                    except VectorStoreError as exc:
                        st.error(str(exc))


# ------------------------------------------------------------------ ask ----
def render_sources(response, key_prefix: str) -> None:
    if not response.sources:
        return
    shown = response.cited_sources
    with st.expander(f"Sources ({len(shown)})", expanded=False):
        for source in shown:
            st.markdown(f"**{source.label}** · similarity {source.similarity:.2f}")
            st.markdown(
                f"<div class='sd-snippet'>{source.snippet}</div>", unsafe_allow_html=True
            )
            if source.url:
                st.markdown(f"[Open the original PDF]({source.url})")


def run_question(question: str, source_filter: str | None):
    store = get_store()
    if not store:
        st.error("Connect Supabase first.")
        return None
    if not settings.groq_ready:
        st.error("Add a Groq API key in the sidebar to generate answers.")
        return None
    try:
        return answer_question(
            question=question,
            store=store,
            groq_api_key=settings.groq_api_key,
            model=settings.model,
            temperature=settings.temperature,
            top_k=settings.top_k,
            source_filter=source_filter,
        )
    except VectorStoreError as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error(f"The model call failed: {exc}")
    return None


with tab_chat:
    filter_choice = None
    if settings.supabase_ready:
        try:
            names = [s["source"] for s in cached_sources(
                settings.supabase_url, settings.supabase_key, st.session_state["cache_nonce"]
            )]
        except VectorStoreError:
            names = []
        if names:
            picked = st.selectbox("Search in", ["All documents"] + names)
            filter_choice = None if picked == "All documents" else picked

    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("response") is not None:
                render_sources(message["response"], key_prefix=str(id(message)))
                st.caption(
                    f"{message['response'].model} · {message['response'].latency_s}s · "
                    f"{len(message['response'].sources)} passages retrieved"
                )

    with st.form("ask_form", clear_on_submit=True):
        question = st.text_area(
            "Your question",
            value=st.session_state.pop("pending_question", ""),
            placeholder="What does the report say about revenue in Q3?",
            height=90,
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Ask", type="primary")

    if submitted and question.strip():
        st.session_state["messages"].append({"role": "user", "content": question.strip()})
        with st.chat_message("user"):
            st.markdown(question.strip())
        with st.chat_message("assistant"):
            with st.spinner("Searching your documents…"):
                response = run_question(question.strip(), filter_choice)
            if response:
                st.markdown(response.answer)
                render_sources(response, key_prefix="live")
                st.caption(
                    f"{response.model} · {response.latency_s}s · "
                    f"{len(response.sources)} passages retrieved"
                )
                st.session_state["messages"].append(
                    {"role": "assistant", "content": response.answer, "response": response}
                )


# ----------------------------------------------------------- evaluation ----
with tab_eval:
    st.subheader("Check the answers")
    st.caption(
        "Run a sample question, read the sources, and score what came back. "
        "Scores stay in this session and can be exported."
    )

    cols = st.columns(len(SAMPLE_QUESTIONS[:3]))
    for col, sample in zip(cols, SAMPLE_QUESTIONS[:3]):
        if col.button(sample, use_container_width=True, key=f"s_{sample}"):
            st.session_state["eval_question"] = sample

    more = st.selectbox("Or pick another", SAMPLE_QUESTIONS, index=0, key="eval_pick")
    custom = st.text_input("Or write your own", key="eval_custom")

    chosen = custom.strip() or st.session_state.get("eval_question") or more
    st.markdown(f"**Question:** {chosen}")

    if st.button("Run this question", type="primary"):
        with st.spinner("Answering…"):
            st.session_state["last_eval"] = run_question(chosen, None)
            st.session_state["last_eval_q"] = chosen

    result = st.session_state.get("last_eval")
    if result:
        st.markdown(result.answer)
        render_sources(result, key_prefix="eval")
        st.caption(f"{result.model} · {result.latency_s}s · {len(result.cited)} citations used")

        with st.form("rating_form"):
            score_col, ground_col = st.columns([2, 1])
            accuracy = score_col.slider("Accuracy (1 poor – 5 excellent)", 1, 5, 4)
            grounded = ground_col.checkbox("Supported by the cited passages", value=True)
            note = st.text_input("Note (optional)", placeholder="Missed the table on page 7")
            if st.form_submit_button("Save rating"):
                record_rating(
                    st.session_state["ratings"],
                    Rating(
                        question=st.session_state.get("last_eval_q", chosen),
                        answer=result.answer[:500],
                        model=result.model,
                        accuracy=accuracy,
                        grounded=grounded,
                        citations=len(result.cited),
                        latency_s=result.latency_s,
                        note=note,
                    ),
                )
                st.success("Rating saved.")

    st.divider()
    metrics = summarise(st.session_state["ratings"])
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Answers rated", metrics["rated"])
    m2.metric("Average accuracy", metrics["avg_accuracy"])
    m3.metric("Grounded", f"{metrics['grounded_rate']}%")
    m4.metric("Average latency", f"{metrics['avg_latency']}s")

    frame = ratings_frame(st.session_state["ratings"])
    if not frame.empty:
        st.dataframe(frame, use_container_width=True, hide_index=True)
        st.download_button(
            "Download ratings as CSV",
            frame.to_csv(index=False).encode("utf-8"),
            file_name="smartdoc_ratings.csv",
            mime="text/csv",
        )


# ---------------------------------------------------------------- setup ----
with tab_setup:
    st.subheader("One-time Supabase setup")
    st.markdown(
        "Run this in the Supabase SQL editor, then create a **public** storage bucket "
        f"named `{STORAGE_BUCKET}`."
    )
    sql_path = Path(__file__).parent / "sql" / "setup.sql"
    sql_text = sql_path.read_text() if sql_path.exists() else "-- sql/setup.sql not found"
    st.code(sql_text, language="sql")
    st.download_button("Download setup.sql", sql_text, file_name="setup.sql", mime="text/plain")

    st.markdown(
        """
**Where the keys come from**

- Supabase URL and anon key: Project settings → API.
- Groq API key: console.groq.com → API Keys.

Put them in `.env` locally, or in Streamlit secrets when you deploy:

```toml
GROQ_API_KEY = "gsk_..."
SUPABASE_URL = "https://xxxx.supabase.co"
SUPABASE_KEY = "eyJ..."
```
"""
    )
