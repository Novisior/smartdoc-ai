-- SmartDoc AI — Supabase setup
-- Run top to bottom in the Supabase SQL editor.

-- 1. Vector support -------------------------------------------------------
create extension if not exists vector;

-- 2. Chunk table ----------------------------------------------------------
create table if not exists documents (
    id         bigserial primary key,
    content    text not null,
    metadata   jsonb not null default '{}'::jsonb,
    embedding  vector(384),
    created_at timestamptz not null default now()
);

-- Metadata lookups (filtering by source filename)
create index if not exists documents_metadata_idx
    on documents using gin (metadata jsonb_path_ops);

-- 3. Similarity search function ------------------------------------------
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

-- 4. HNSW index for cosine distance --------------------------------------
create index if not exists documents_embedding_hnsw_idx
    on documents using hnsw (embedding vector_cosine_ops)
    with (m = 16, ef_construction = 64);

-- 5. Storage bucket -------------------------------------------------------
insert into storage.buckets (id, name, public)
values ('documents', 'documents', true)
on conflict (id) do nothing;

-- 6. Access policies ------------------------------------------------------
-- These allow the anon key to read and write. Fine for a personal demo;
-- tighten them (or switch to authenticated-only) before putting real data in.
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

-- Optional: start over
-- truncate table documents restart identity;
