CREATE TABLE IF NOT EXISTS yieldmind_stage_executions (
    stage_execution_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    thread_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    status TEXT NOT NULL,
    input_hash TEXT NOT NULL,
    started_at REAL NOT NULL,
    completed_at REAL NOT NULL,
    payload_json TEXT NOT NULL DEFAULT '{}',
    error TEXT NOT NULL DEFAULT '',
    FOREIGN KEY (run_id) REFERENCES yieldmind_runs(run_id)
);

CREATE INDEX IF NOT EXISTS idx_yieldmind_stage_executions_run
    ON yieldmind_stage_executions(run_id, started_at);

CREATE INDEX IF NOT EXISTS idx_yieldmind_stage_executions_thread
    ON yieldmind_stage_executions(thread_id, started_at);
