"""SQLAlchemy metadata shared by Alembic and PostgreSQL integration checks."""

from __future__ import annotations

from sqlalchemy import Column, Float, ForeignKey, Index, Integer, MetaData, String, Table, Text, UniqueConstraint, text


metadata = MetaData()

runs = Table(
    "yieldmind_runs",
    metadata,
    Column("run_id", String(64), primary_key=True),
    Column("status", String(64), nullable=False),
    Column("mode", String(64), nullable=False),
    Column("source", String(128), nullable=False),
    Column("started_at", Float, nullable=False),
    Column("updated_at", Float, nullable=False),
    Column("completed_at", Float),
    Column("metadata_json", Text, nullable=False, server_default="{}"),
    Column("result_json", Text, nullable=False, server_default="{}"),
)

events = Table(
    "yieldmind_events",
    metadata,
    Column("event_id", Integer, primary_key=True, autoincrement=True),
    Column("run_id", String(64), ForeignKey("yieldmind_runs.run_id"), nullable=False),
    Column("ts", Float, nullable=False),
    Column("stage", String(128), nullable=False),
    Column("level", String(32), nullable=False, server_default="info"),
    Column("message", Text, nullable=False),
    Column("payload_json", Text, nullable=False, server_default="{}"),
)
Index("idx_yieldmind_events_run_ts", events.c.run_id, events.c.event_id)

tool_calls = Table(
    "yieldmind_tool_calls",
    metadata,
    Column("call_id", String(64), primary_key=True),
    Column("run_id", String(64), ForeignKey("yieldmind_runs.run_id")),
    Column("tool_name", String(128), nullable=False),
    Column("mode", String(64), nullable=False),
    Column("status", String(64), nullable=False),
    Column("started_at", Float, nullable=False),
    Column("completed_at", Float),
    Column("args_json", Text, nullable=False, server_default="{}"),
    Column("result_json", Text, nullable=False, server_default="{}"),
    Column("error", Text, nullable=False, server_default=""),
    Column("session_id", String(64), nullable=False, server_default=""),
    Column("turn_id", String(64), nullable=False, server_default=""),
    Column("idempotency_key", String(128), nullable=False, server_default=""),
)
Index("idx_yieldmind_tool_calls_run", tool_calls.c.run_id, tool_calls.c.started_at)
Index(
    "uq_yieldmind_tool_calls_idempotency",
    tool_calls.c.idempotency_key,
    unique=True,
    sqlite_where=text("idempotency_key <> ''"),
    postgresql_where=text("idempotency_key <> ''"),
)

eval_runs = Table(
    "yieldmind_eval_runs",
    metadata,
    Column("eval_id", String(64), primary_key=True),
    Column("mode", String(64), nullable=False),
    Column("started_at", Float, nullable=False),
    Column("completed_at", Float),
    Column("summary_json", Text, nullable=False, server_default="{}"),
    Column("report_path", Text, nullable=False, server_default=""),
)

documents = Table(
    "yieldmind_documents",
    metadata,
    Column("document_id", String(64), primary_key=True),
    Column("title", Text, nullable=False),
    Column("source_path", Text, nullable=False),
    Column("source_type", String(32), nullable=False),
    Column("document_hash", String(128), nullable=False),
    Column("document_version", String(128), nullable=False),
    Column("status", String(64), nullable=False),
    Column("index_version", String(128), nullable=False),
    Column("chunk_count", Integer, nullable=False, server_default="0"),
    Column("error", Text, nullable=False, server_default=""),
    Column("created_at", Float, nullable=False),
    Column("updated_at", Float, nullable=False),
    UniqueConstraint("source_path", "document_hash", "index_version", name="uq_yieldmind_documents_source_hash"),
)

document_chunks = Table(
    "yieldmind_document_chunks",
    metadata,
    Column("chunk_id", String(64), primary_key=True),
    Column("document_id", String(64), ForeignKey("yieldmind_documents.document_id"), nullable=False),
    Column("document_version", String(128), nullable=False),
    Column("chunk_index", Integer, nullable=False),
    Column("title", Text, nullable=False),
    Column("source_path", Text, nullable=False),
    Column("section", Text, nullable=False, server_default=""),
    Column("text_hash", String(128), nullable=False),
    Column("index_version", String(128), nullable=False),
    Column("text", Text, nullable=False),
    Column("created_at", Float, nullable=False),
)
Index("idx_yieldmind_chunks_document", document_chunks.c.document_id, document_chunks.c.document_version)

sessions = Table(
    "yieldmind_sessions",
    metadata,
    Column("session_id", String(64), primary_key=True),
    Column("status", String(64), nullable=False),
    Column("created_at", Float, nullable=False),
    Column("updated_at", Float, nullable=False),
    Column("workspace_id", String(64), nullable=False, server_default="default"),
    Column("current_run_id", String(64), nullable=False, server_default=""),
    Column("constraints_json", Text, nullable=False, server_default="{}"),
    Column("constraint_version", Integer, nullable=False, server_default="0"),
    Column("summary", Text, nullable=False, server_default=""),
    Column("summary_through_turn_id", String(64), nullable=False, server_default=""),
)

turns = Table(
    "yieldmind_turns",
    metadata,
    Column("turn_id", String(64), primary_key=True),
    Column("session_id", String(64), ForeignKey("yieldmind_sessions.session_id"), nullable=False),
    Column("idempotency_key", String(128), nullable=False),
    Column("role", String(32), nullable=False),
    Column("content", Text, nullable=False),
    Column("status", String(64), nullable=False),
    Column("created_at", Float, nullable=False),
    Column("constraint_version", Integer, nullable=False, server_default="0"),
    Column("metadata_json", Text, nullable=False, server_default="{}"),
    UniqueConstraint("session_id", "idempotency_key", name="uq_yieldmind_turn_idempotency"),
)

memories = Table(
    "yieldmind_memories",
    metadata,
    Column("memory_id", String(64), primary_key=True),
    Column("session_id", String(64), nullable=False, server_default=""),
    Column("workspace_id", String(64), nullable=False, server_default="default"),
    Column("scope", String(64), nullable=False),
    Column("kind", String(64), nullable=False),
    Column("content", Text, nullable=False),
    Column("source_ref", Text, nullable=False),
    Column("status", String(64), nullable=False),
    Column("validation_status", String(32), nullable=False, server_default="confirmed"),
    Column("source_run_id", String(64), nullable=False, server_default=""),
    Column("applicability_json", Text, nullable=False, server_default="{}"),
    Column("created_at", Float, nullable=False),
    Column("updated_at", Float, nullable=False),
    Column("metadata_json", Text, nullable=False, server_default="{}"),
)
Index("idx_yieldmind_memories_scope", memories.c.scope, memories.c.kind, memories.c.status)
Index(
    "idx_yieldmind_memories_context",
    memories.c.workspace_id,
    memories.c.session_id,
    memories.c.scope,
    memories.c.validation_status,
    memories.c.status,
)

stage_executions = Table(
    "yieldmind_stage_executions",
    metadata,
    Column("stage_execution_id", String(64), primary_key=True),
    Column("run_id", String(64), ForeignKey("yieldmind_runs.run_id"), nullable=False),
    Column("thread_id", String(64), nullable=False),
    Column("stage", String(128), nullable=False),
    Column("status", String(64), nullable=False),
    Column("input_hash", String(128), nullable=False),
    Column("started_at", Float, nullable=False),
    Column("completed_at", Float, nullable=False),
    Column("payload_json", Text, nullable=False, server_default="{}"),
    Column("error", Text, nullable=False, server_default=""),
)
Index("idx_yieldmind_stage_executions_run", stage_executions.c.run_id, stage_executions.c.started_at)
Index("idx_yieldmind_stage_executions_thread", stage_executions.c.thread_id, stage_executions.c.started_at)

tasks = Table(
    "yieldmind_tasks",
    metadata,
    Column("task_id", String(64), primary_key=True),
    Column("task_type", String(128), nullable=False),
    Column("status", String(64), nullable=False),
    Column("idempotency_key", String(128), nullable=False, unique=True),
    Column("payload_json", Text, nullable=False, server_default="{}"),
    Column("result_json", Text, nullable=False, server_default="{}"),
    Column("run_id", String(64), nullable=False, server_default=""),
    Column("error", Text, nullable=False, server_default=""),
    Column("cancel_requested_at", Float),
    Column("cancelled_at", Float),
    Column("cancel_reason", Text, nullable=False, server_default=""),
    Column("cancel_requested_by", String(128), nullable=False, server_default=""),
    Column("worker_id", String(256), nullable=False, server_default=""),
    Column("heartbeat_at", Float),
    Column("lease_expires_at", Float),
    Column("interrupted_at", Float),
    Column("recovery_reason", Text, nullable=False, server_default=""),
    Column("recovery_requested_by", String(128), nullable=False, server_default=""),
    Column("recovery_of_task_id", String(64), nullable=False, server_default=""),
    Column("created_at", Float, nullable=False),
    Column("updated_at", Float, nullable=False),
)
Index("idx_yieldmind_tasks_status", tasks.c.status, tasks.c.updated_at)
Index("idx_yieldmind_tasks_lease", tasks.c.status, tasks.c.lease_expires_at)
Index("idx_yieldmind_tasks_recovery", tasks.c.recovery_of_task_id)
