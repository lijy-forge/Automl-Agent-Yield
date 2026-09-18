#!/usr/bin/env python3
"""Publish one real Redis/Celery workflow task and verify its PostgreSQL record."""

from __future__ import annotations

import json
import sys
import time
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from yieldmind.database import YieldMindStore
from yieldmind.safety import RedactRequest, redact_payload
from yieldmind.task_queue import EnqueueWorkflowRequest, enqueue_workflow, get_task_status
from yieldmind.workflow import WorkflowRequest


TERMINAL_STATUSES = {"completed", "failed", "dispatch_failed"}


def main() -> int:
    started = time.time()
    store = YieldMindStore()
    if store.backend != "postgresql":
        raise RuntimeError("Queue integration smoke requires PostgreSQL via YIELDMIND_DATABASE_URL.")

    idempotency_key = f"queue-smoke-{uuid.uuid4().hex[:16]}"
    request = EnqueueWorkflowRequest(
        workflow=WorkflowRequest(
            prompt="Real Redis/Celery queue smoke with deterministic tools.",
            n_samples=20,
            n_splits=2,
        ),
        idempotency_key=idempotency_key,
    )
    published = enqueue_workflow(request, store=store)
    repeated = enqueue_workflow(request, store=store)
    task_id = str(published["task"]["task_id"])

    latest = get_task_status(task_id, store=store)
    deadline = time.time() + 120
    while latest and latest["task"]["status"] not in TERMINAL_STATUSES and time.time() < deadline:
        time.sleep(0.5)
        latest = get_task_status(task_id, store=store)

    task = (latest or {}).get("task") or {}
    checks = {
        "published_to_redis": bool(published.get("published")),
        "idempotent_repeat": bool(repeated.get("idempotent")) and repeated["task"]["task_id"] == task_id,
        "worker_completed": task.get("status") == "completed",
        "run_persisted": bool(task.get("run_id")) and store.get_run(str(task.get("run_id"))) is not None,
        "celery_terminal_state": (latest or {}).get("celery_state") == "SUCCESS",
    }
    report = {
        "status": "passed" if all(checks.values()) else "failed",
        "mode": "postgresql_redis_celery_integration",
        "real_llm_calls": 0,
        "simulated_model_calls": 0,
        "checks": checks,
        "task_id": task_id,
        "task_status": task.get("status"),
        "celery_state": (latest or {}).get("celery_state"),
        "run_id": task.get("run_id"),
        "duration_seconds": round(time.time() - started, 4),
    }
    report = redact_payload(RedactRequest(payload=report)).payload
    out_dir = PROJECT_ROOT / "agent_workspace" / "yieldmind" / "integrations"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"postgres_redis_celery_smoke_{time.strftime('%Y%m%d_%H%M%S')}.json"
    report["report_path"] = str(report_path)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
