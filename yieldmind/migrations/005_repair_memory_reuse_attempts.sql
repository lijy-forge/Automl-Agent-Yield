CREATE TABLE IF NOT EXISTS yieldmind_repair_memory_reuse_attempts (
    reuse_id TEXT PRIMARY KEY,
    memory_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    workspace_id TEXT NOT NULL,
    execution_mode TEXT NOT NULL,
    matched_round INTEGER NOT NULL,
    applied_round INTEGER NOT NULL,
    status TEXT NOT NULL,
    operation_rcode INTEGER,
    manager_passed INTEGER,
    manager_decision TEXT NOT NULL DEFAULT '',
    matched_at REAL NOT NULL,
    applied_at REAL NOT NULL,
    completed_at REAL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    FOREIGN KEY (memory_id) REFERENCES yieldmind_memories(memory_id),
    FOREIGN KEY (run_id) REFERENCES yieldmind_runs(run_id),
    UNIQUE (memory_id, run_id, applied_round)
);

CREATE INDEX IF NOT EXISTS idx_yieldmind_repair_reuse_workspace
    ON yieldmind_repair_memory_reuse_attempts(workspace_id, status, applied_at);

CREATE INDEX IF NOT EXISTS idx_yieldmind_repair_reuse_memory
    ON yieldmind_repair_memory_reuse_attempts(memory_id, applied_at);
