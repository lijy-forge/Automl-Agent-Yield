CREATE TABLE IF NOT EXISTS yieldmind_documents (
    document_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    source_path TEXT NOT NULL,
    source_type TEXT NOT NULL,
    corpus TEXT NOT NULL DEFAULT 'project',
    source_url TEXT NOT NULL DEFAULT '',
    doi TEXT NOT NULL DEFAULT '',
    license TEXT NOT NULL DEFAULT '',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    document_hash TEXT NOT NULL,
    document_version TEXT NOT NULL,
    status TEXT NOT NULL,
    index_version TEXT NOT NULL,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    error TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_yieldmind_documents_source_hash
    ON yieldmind_documents(source_path, document_hash, index_version);

CREATE TABLE IF NOT EXISTS yieldmind_document_chunks (
    chunk_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    document_version TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    title TEXT NOT NULL,
    source_path TEXT NOT NULL,
    section TEXT NOT NULL DEFAULT '',
    page_start INTEGER,
    page_end INTEGER,
    text_hash TEXT NOT NULL,
    index_version TEXT NOT NULL,
    text TEXT NOT NULL,
    created_at REAL NOT NULL,
    FOREIGN KEY (document_id) REFERENCES yieldmind_documents(document_id)
);

CREATE INDEX IF NOT EXISTS idx_yieldmind_chunks_document
    ON yieldmind_document_chunks(document_id, document_version);

CREATE TABLE IF NOT EXISTS yieldmind_sessions (
    session_id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    current_run_id TEXT NOT NULL DEFAULT '',
    constraints_json TEXT NOT NULL DEFAULT '{}',
    summary TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS yieldmind_turns (
    turn_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    idempotency_key TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at REAL NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    FOREIGN KEY (session_id) REFERENCES yieldmind_sessions(session_id)
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_yieldmind_turn_idempotency
    ON yieldmind_turns(session_id, idempotency_key);

CREATE TABLE IF NOT EXISTS yieldmind_memories (
    memory_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL DEFAULT '',
    scope TEXT NOT NULL,
    kind TEXT NOT NULL,
    content TEXT NOT NULL,
    source_ref TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_yieldmind_memories_scope
    ON yieldmind_memories(scope, kind, status);
