# SmartDoc AI

Ask questions across your own PDFs and get answers that point back to the file and page they came from.

Built entirely on free tiers: Streamlit for the interface, Supabase (Postgres + pgvector) for the vectors and the original files, local MiniLM embeddings, and Groq for generation.

New here? **[SETUP.md](SETUP.md)** walks through every account, key and click in order. The sections below are the reference version.

---

## How it works

```
PDF upload
   ├─ original file  ──────────────────▶ Supabase Storage (bucket: documents)
   └─ pypdf text → RecursiveCharacterTextSplitter (800 / 150)
         → all-MiniLM-L6-v2 (384-d, runs locally, no API cost)
         → documents table (content, metadata jsonb, embedding vector(384))

Question
   → embed the question
   → match_documents() cosine search over the HNSW index
   → top-k passages numbered [1]..[k] into the prompt
   → Groq (llama-3.1-8b-instant / llama-3.3-70b-versatile)
   → answer with inline [n] citations + a source panel with page numbers and snippets
```

## Project structure

```
smartdoc-ai/
├── app.py                      Streamlit UI: Documents, Ask, Evaluation, Setup tabs
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
├── SETUP.md                    Click-by-click setup walkthrough
├── sql/
│   └── setup.sql               Everything to run in the Supabase SQL editor
├── src/
│   ├── __init__.py
│   ├── config.py               Secrets, chunking constants, model list
│   ├── pdf_processor.py        Load PDFs, split, attach source + page metadata
│   ├── vector_store.py         Supabase table, RPC search, storage bucket
│   ├── rag_chain.py            Prompt, Groq call, citation parsing
│   └── evaluation.py           Sample questions, ratings, summary metrics
└── .streamlit/
    ├── config.toml
    └── secrets.toml.example
```

---

## 1. Supabase setup

Create a free project at [supabase.com](https://supabase.com), open the **SQL Editor**, and run the contents of `sql/setup.sql`. It is reproduced here in full.

```sql
-- 1. Vector support
create extension if not exists vector;

-- 2. Chunk table
create table if not exists documents (
    id         bigserial primary key,
    content    text not null,
    metadata   jsonb not null default '{}'::jsonb,
    embedding  vector(384),
    created_at timestamptz not null default now()
);

create index if not exists documents_metadata_idx
    on documents using gin (metadata jsonb_path_ops);

-- 3. Similarity search function
create or replace function match_documents(
    query_embedding vector(384),
    match_count int default 4,
    filter jsonb default '{}'::jsonb
)
returns table (
    id bigint,
    content text,
    metadata jsonb,
    similarity float
)
language plpgsql
as $$
begin
    return query
    select
        documents.id,
        documents.content,
        documents.metadata,
        1 - (documents.embedding <=> query_embedding) as similarity
    from documents
    where documents.metadata @> filter
    order by documents.embedding <=> query_embedding
    limit match_count;
end;
$$;

-- 4. HNSW index for cosine distance
create index if not exists documents_embedding_hnsw_idx
    on documents using hnsw (embedding vector_cosine_ops)
    with (m = 16, ef_construction = 64);

-- 5. Storage bucket
insert into storage.buckets (id, name, public)
values ('documents', 'documents', true)
on conflict (id) do nothing;

-- 6. Access policies (anon read/write — demo settings, tighten for real data)
alter table documents enable row level security;

drop policy if exists "anon read documents" on documents;
create policy "anon read documents" on documents
    for select to anon, authenticated using (true);

drop policy if exists "anon write documents" on documents;
create policy "anon write documents" on documents
    for insert to anon, authenticated with check (true);

drop policy if exists "anon delete documents" on documents;
create policy "anon delete documents" on documents
    for delete to anon, authenticated using (true);

drop policy if exists "public read bucket" on storage.objects;
create policy "public read bucket" on storage.objects
    for select to anon, authenticated using (bucket_id = 'documents');

drop policy if exists "public write bucket" on storage.objects;
create policy "public write bucket" on storage.objects
    for insert to anon, authenticated with check (bucket_id = 'documents');

drop policy if exists "public delete bucket" on storage.objects;
create policy "public delete bucket" on storage.objects
    for delete to anon, authenticated using (bucket_id = 'documents');
```

Then confirm under **Storage** that a public bucket named `documents` exists. Step 5 creates it; if your project blocks inserts into `storage.buckets`, create it by hand with **New bucket → name `documents` → Public**.

## 2. Keys

| Key | Where to find it |
| --- | --- |
| `SUPABASE_URL` | Supabase → Project Settings → API → Project URL |
| `SUPABASE_KEY` | Supabase → Project Settings → API → `anon` `public` key |
| `GROQ_API_KEY` | [console.groq.com](https://console.groq.com) → API Keys |

## 3. Run it locally

```bash
git clone <your-repo-url> smartdoc-ai
cd smartdoc-ai

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env               # then fill in your three keys
streamlit run app.py
```

The first question downloads the MiniLM model (about 90 MB) and caches it. Everything after that is instant.

You can also skip `.env` entirely and paste the keys into the sidebar under **Connection**.

## 4. Deploy to Streamlit Community Cloud

1. Push the repo to GitHub (`.env` is gitignored).
2. On [share.streamlit.io](https://share.streamlit.io), create an app pointing at `app.py`.
3. In **App settings → Secrets**, paste:

```toml
GROQ_API_KEY = "gsk_..."
SUPABASE_URL = "https://xxxx.supabase.co"
SUPABASE_KEY = "eyJ..."
```

The free Streamlit container has about 1 GB of RAM. MiniLM plus CPU torch fits, but keep the app to one embedding model and avoid indexing very large PDFs in a single run. If the container runs out of memory during install, pin a CPU-only torch wheel in `requirements.txt`:

```
--extra-index-url https://download.pytorch.org/whl/cpu
torch==2.2.2+cpu
```

---

## Using the app

**Documents** — drop in one or more PDFs and press *Index documents*. You see per-file progress through upload, splitting, embedding and saving, then a table of pages and chunks. The indexed library below lists everything stored, with a Remove button per file.

**Ask** — type a question, optionally scoped to a single file. The answer carries inline `[1]` `[2]` markers, and the Sources panel expands to show each cited passage: filename, page number, similarity score, the snippet itself, and a link to the original PDF in Supabase Storage. Conversation history stays in the session.

**Evaluation** — five sample questions, a Run button, and a short rating form (accuracy 1–5, a grounded checkbox, a free-text note). Ratings roll up into average accuracy, grounded rate and latency, and export as CSV.

**Setup** — the SQL above, viewable and downloadable from inside the app.

### Sidebar controls

- Connection: optional key overrides.
- Model: `llama-3.1-8b-instant` (fast, generous rate limits) or `llama-3.3-70b-versatile` (better at multi-passage reasoning).
- Passages per answer (`top_k`) and temperature.
- Clear chat, and Clear documents with a confirmation step that wipes both the table and the bucket.

---

## Design notes

**Why page numbers survive.** `PyPDFLoader` returns one document per page, and splitting happens *after* loading, so each chunk inherits the page it came from. Pages are shifted from pypdf's 0-index to human 1-index at load time.

**Why the citations are checkable.** Passages go into the prompt pre-numbered with their filename and page. The answer's `[n]` markers are parsed back out with a regex, and only the sources the model actually cited are shown by default — so an uncited answer is visibly uncited rather than silently decorated with every retrieved chunk.

**Why normalized embeddings.** MiniLM vectors are L2-normalized at encode time, which makes the `<=>` cosine distance in `match_documents` line up with the `1 - distance` similarity score shown in the UI.

**Chunk size.** 800 characters with 150 overlap keeps roughly a paragraph of context per chunk while leaving room for 4–10 passages inside Groq's context window.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `function match_documents does not exist` | Step 3 of the SQL did not run. Re-run it and check the signature matches `vector(384), int, jsonb`. |
| `expected 384 dimensions` | The table was created with a different dimension. Drop and recreate `documents`. |
| Upload fails with a bucket error | The `documents` bucket is missing or not public. Create it under Storage. |
| "produced no readable text" | The PDF is scanned images. Run OCR (e.g. `ocrmypdf`) before uploading. |
| `rate_limit_exceeded` from Groq | Free-tier limit. Wait a minute, or switch to `llama-3.1-8b-instant`. |
| Answers say nothing was found | Index at least one document, raise *Passages per answer*, or use wording closer to the document's own. |
| Slow first question | One-time model download. Subsequent calls use the cached model. |

## Limits worth knowing

- RLS is open to the anon key so the demo works without auth. Add Supabase Auth and per-user policies before storing anything sensitive.
- Retrieval is dense-only. Exact identifiers (part numbers, dates) sometimes need keyword search; adding a `tsvector` column and blending the two scores is the natural next step.
- Evaluation ratings live in session state and disappear on refresh unless exported to CSV.
