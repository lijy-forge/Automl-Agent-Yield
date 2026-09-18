PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS yieldmind_runs (
    run_id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    mode TEXT NOT NULL,
    source TEXT NOT NULL,
    started_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    completed_at REAL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    result_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS yieldmind_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    ts REAL NOT NULL,
    stage TEXT NOT NULL,
    level TEXT NOT NULL DEFAULT 'info',
    message TEXT NOT NULL,
    payload_json TEXT NOT NULL DEFAULT '{}',
    FOREIGN KEY (run_id) REFERENCES yieldmind_runs(run_id)
);

CREATE INDEX IF NOT EXISTS idx_yieldmind_events_run_ts
    ON yieldmind_events(run_id, event_id);

CREATE TABLE IF NOT EXISTS yieldmind_tool_calls (
    call_id TEXT PRIMARY KEY,
    run_id TEXT,
    tool_name TEXT NOT NULL,
    mode TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at REAL NOT NULL,
    completed_at REAL,
    args_json TEXT NOT NULL DEFAULT '{}',
    result_json TEXT NOT NULL DEFAULT '{}',
    error TEXT NOT NULL DEFAULT '',
    idempotency_key TEXT NOT NULL DEFAULT '',
    FOREIGN KEY (run_id) REFERENCES yieldmind_runs(run_id)
);

CREATE INDEX IF NOT EXISTS idx_yieldmind_tool_calls_run
    ON yieldmind_tool_calls(run_id, started_at);

CREATE TABLE IF NOT EXISTS yieldmind_eval_runs (
    eval_id TEXT PRIMARY KEY,
    mode TEXT NOT NULL,
    started_at REAL NOT NULL,
    completed_at REAL,
    summary_json TEXT NOT NULL DEFAULT '{}',
    report_path TEXT NOT NULL DEFAULT ''
);
