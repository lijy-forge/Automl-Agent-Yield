#!/usr/bin/env python3
"""Run a real PostgreSQL/Redis integration smoke without model API calls."""

from __future__ import annotations

import json
import sys
import time
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from yieldmind.database import YieldMindStore, connect
from yieldmind.knowledge_base import KnowledgeBase, KnowledgeIngestRequest, KnowledgeSearchRequest
from yieldmind.memory import AddMessageRequest, CreateSessionRequest, SessionMemoryStore
from yieldmind.safety import RedactRequest, redact_payload
from yieldmind.task_queue import redis_health
from yieldmind.tools import ToolRegistry
from yieldmind.workflow import WorkflowRequest, run_workflow


def main() -> int:
    started = time.time()
    store = YieldMindStore()
    if store.backend != "postgresql":
        raise RuntimeError("Set YIELDMIND_DATABASE_URL to a PostgreSQL URL for this integration smoke.")

    database_ok = store.ping()
    with connect(store.db_path) as conn:
        migration_row = conn.execute("SELECT version_num FROM alembic_version").fetchone()
    redis_status = redis_health()
    workflow = run_workflow(
        WorkflowRequest(
            prompt="PostgreSQL/Redis integration smoke with deterministic tools.",
            n_samples=20,
            n_splits=2,
        ),
        store=store,
    )
    stage_rows = store.list_stage_executions(str(workflow.get("run_id") or ""))
    with connect(store.db_path) as conn:
        checkpoint_row = conn.execute(
            "SELECT COUNT(*) AS count FROM checkpoints WHERE thread_id = ?",
            (str(workflow.get("thread_id") or ""),),
        ).fetchone()

    tool_replay_key = f"pg-smoke-tool-{uuid.uuid4().hex[:12]}"
    tool_registry = ToolRegistry(store=store)
    tool_args = {"data_path": str(workflow.get("data_path") or "")}
    first_tool_result = tool_registry.execute(
        "profile_yield_data",
        tool_args,
        idempotency_key=tool_replay_key,
    )
    replayed_tool_result = tool_registry.execute(
        "profile_yield_data",
        tool_args,
        idempotency_key=tool_replay_key,
    )
    with connect(store.db_path) as conn:
        tool_replay_row = conn.execute(
            "SELECT COUNT(*) AS count FROM yieldmind_tool_calls WHERE idempotency_key = ?",
            (tool_replay_key,),
        ).fetchone()

    memory = SessionMemoryStore(store)
    session = memory.create_session(CreateSessionRequest(constraints={"target_column": "yield_stress"}))
    turn = memory.add_message(
        AddMessageRequest(
            session_id=session["session_id"],
            content="把搜索预算调小，再运行一次",
            idempotency_key=f"pg-smoke-{uuid.uuid4().hex[:12]}",
        )
    )

    kb = KnowledgeBase(store=store)
    seed_path = PROJECT_ROOT / "knowledge_sources" / "yieldmind_domain_seed.md"
    ingest = kb.ingest(KnowledgeIngestRequest(paths=[str(seed_path)]))
    search = kb.search(
        KnowledgeSearchRequest(query="YODEL packing yield stress", top_k=2, retrieval_mode="hybrid")
    )

    task_key = f"pg-smoke-task-{uuid.uuid4().hex[:12]}"
    task, idempotent_first = store.create_task(
        task_type="integration_smoke",
        idempotency_key=task_key,
        payload={"run_id": workflow.get("run_id")},
    )
    same_task, idempotent_second = store.create_task(
        task_type="integration_smoke",
        idempotency_key=task_key,
        payload={"run_id": workflow.get("run_id")},
    )
    store.transition_task(task["task_id"], from_statuses=("created",), to_status="queued")
    store.transition_task(task["task_id"], from_statuses=("queued",), to_status="running")
    store.transition_task(
        task["task_id"],
        from_statuses=("running",),
        to_status="completed",
        result={"ok": True},
    )

    checks = {
        "database_ping": database_ok,
        "alembic_revision": bool(migration_row and migration_row["version_num"] == "20260918_0004"),
        "redis_ping": bool(redis_status.get("ok")),
        "workflow_passed": workflow.get("status") == "passed",
        "langgraph_postgres_checkpoint": (
            workflow.get("checkpoint_backend") == "langgraph_postgres"
            and bool(checkpoint_row and int(checkpoint_row["count"]) > 0)
        ),
        "tool_call_idempotent_replay": (
            first_tool_result == replayed_tool_result
            and bool(tool_replay_row and int(tool_replay_row["count"]) == 1)
        ),
        "stage_rows_persisted": len(stage_rows) == len(workflow.get("stages", [])) and len(stage_rows) > 0,
        "memory_created_followup_run": turn.get("action") == "create_run" and bool(turn.get("run_id")),
        "knowledge_metadata_persisted": bool(ingest.get("documents")) and bool(search.get("hits")),
        "knowledge_hybrid_postgres": (
            search.get("retrieval_mode") == "hybrid"
            and search.get("lexical_algorithm") == "bm25_plus"
            and bool(search.get("hits"))
            and {"vector", "bm25"}.issubset(set(search["hits"][0].get("retrieval_channels", [])))
        ),
        "task_idempotency": not idempotent_first and idempotent_second and task["task_id"] == same_task["task_id"],
    }
    report = {
        "status": "passed" if all(checks.values()) else "failed",
        "mode": "postgresql_redis_integration",
        "database_backend": store.backend,
        "database_location": store.location,
        "real_llm_calls": 0,
        "simulated_model_calls": 0,
        "checks": checks,
        "workflow_run_id": workflow.get("run_id"),
        "stage_execution_count": len(stage_rows),
        "checkpoint_count": int(checkpoint_row["count"]) if checkpoint_row else 0,
        "session_id": session.get("session_id"),
        "followup_run_id": turn.get("run_id"),
        "knowledge_hit_count": len(search.get("hits", [])),
        "knowledge_retrieval_mode": search.get("retrieval_mode"),
        "task_id": task.get("task_id"),
        "duration_seconds": round(time.time() - started, 4),
    }
    report = redact_payload(RedactRequest(payload=report)).payload
    out_dir = PROJECT_ROOT / "agent_workspace" / "yieldmind" / "integrations"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"postgres_redis_smoke_{time.strftime('%Y%m%d_%H%M%S')}.json"
    report["report_path"] = str(report_path)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
