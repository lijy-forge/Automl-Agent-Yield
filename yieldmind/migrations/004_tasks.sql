CREATE TABLE IF NOT EXISTS yieldmind_tasks (
    task_id TEXT PRIMARY KEY,
    task_type TEXT NOT NULL,
    status TEXT NOT NULL,
    idempotency_key TEXT NOT NULL UNIQUE,
    payload_json TEXT NOT NULL DEFAULT '{}',
    result_json TEXT NOT NULL DEFAULT '{}',
    run_id TEXT NOT NULL DEFAULT '',
    error TEXT NOT NULL DEFAULT '',
    cancel_requested_at REAL,
    cancelled_at REAL,
    cancel_reason TEXT NOT NULL DEFAULT '',
    cancel_requested_by TEXT NOT NULL DEFAULT '',
    worker_id TEXT NOT NULL DEFAULT '',
    heartbeat_at REAL,
    lease_expires_at REAL,
    interrupted_at REAL,
    recovery_reason TEXT NOT NULL DEFAULT '',
    recovery_requested_by TEXT NOT NULL DEFAULT '',
    recovery_of_task_id TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_yieldmind_tasks_status
    ON yieldmind_tasks(status, updated_at);
