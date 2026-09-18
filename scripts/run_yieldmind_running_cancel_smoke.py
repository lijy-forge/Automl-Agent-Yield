#!/usr/bin/env python3
"""Verify cooperative cancellation of a running Celery workflow."""

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
from yieldmind.task_queue import (
    EnqueueWorkflowRequest,
    cancel_queued_task,
    enqueue_workflow,
    get_task_status,
)
from yieldmind.workflow import WorkflowRequest


TERMINAL_STATUSES = {"cancelled", "completed", "failed", "dispatch_failed"}


def main() -> int:
    started = time.time()
    store = YieldMindStore()
    if store.backend != "postgresql":
        raise RuntimeError("Running cancellation smoke requires PostgreSQL via YIELDMIND_DATABASE_URL.")

    request = EnqueueWorkflowRequest(
        workflow=WorkflowRequest(
            prompt="Cooperative running cancellation integration smoke without model API calls.",
            n_samples=5000,
            n_splits=3,
        ),
        idempotency_key=f"running-cancel-{uuid.uuid4().hex[:16]}",
    )
    published = enqueue_workflow(request, store=store)
    task_id = str(published["task"]["task_id"])

    saw_running = False
    running_deadline = time.time() + 30
    task = store.get_task(task_id) or {}
    while time.time() < running_deadline:
        task = store.get_task(task_id) or {}
        if task.get("status") == "running":
            saw_running = True
            break
        if task.get("status") in TERMINAL_STATUSES:
            break
        time.sleep(0.02)

    cancellation = None
    cancellation_error = ""
    if saw_running:
        try:
            cancellation = cancel_queued_task(
                task_id,
                store=store,
                reason="integration smoke requests a cooperative node-boundary stop",
                requested_by="running_cancel_smoke",
            )
        except Exception as exc:
            cancellation_error = f"{type(exc).__name__}: {exc}"

    latest = get_task_status(task_id, store=store)
    terminal_deadline = time.time() + 120
    while latest and latest["task"]["status"] not in TERMINAL_STATUSES and time.time() < terminal_deadline:
        time.sleep(0.1)
        latest = get_task_status(task_id, store=store)

    task = (latest or {}).get("task") or {}
    run_id = str(task.get("run_id") or "")
    run = store.get_run(run_id) if run_id else None
    run_result = (run or {}).get("result") or {}
    stages = run_result.get("stages") or []
    stage_names = [str(item.get("stage") or "") for item in stages]
    cancel_index = stage_names.index("cancel") if "cancel" in stage_names else -1
    only_finish_after_cancel = cancel_index >= 0 and stage_names[cancel_index + 1 :] == ["finish"]
    response_status = str(((cancellation or {}).get("task") or {}).get("status") or "")
    checks = {
        "published_to_redis": bool(published.get("published")),
        "observed_database_running": saw_running,
        "running_cancel_request_accepted": response_status in {"cancel_requested", "cancelled"},
        "database_cancelled": task.get("status") == "cancelled",
        "cancellation_audit_persisted": (
            bool(task.get("cancel_requested_at"))
            and bool(task.get("cancelled_at"))
            and task.get("cancel_requested_by") == "running_cancel_smoke"
        ),
        "task_run_link_consistent": bool(run_id) and bool(run) and run_id == run.get("run_id"),
        "run_cancelled": (run or {}).get("status") == "cancelled" and run_result.get("status") == "cancelled",
        "cancel_node_persisted": "cancel" in stage_names,
        "no_workflow_node_after_cancel": only_finish_after_cancel,
        "celery_state_revoked": (latest or {}).get("celery_state") == "REVOKED",
    }
    report = {
        "status": "passed" if all(checks.values()) else "failed",
        "mode": "postgresql_redis_celery_running_cancellation",
        "database_backend": store.backend,
        "real_llm_calls": 0,
        "simulated_model_calls": 0,
        "checks": checks,
        "task_id": task_id,
        "task_status": task.get("status"),
        "celery_state": (latest or {}).get("celery_state"),
        "run_id": run_id,
        "run_status": (run or {}).get("status"),
        "stage_names": stage_names,
        "cancel_after_stage": next(
            (item.get("after_stage") for item in run_result.get("route_history", []) if item.get("route") == "cancel"),
            "",
        ),
        "cancellation_error": cancellation_error,
        "duration_seconds": round(time.time() - started, 4),
    }
    report = redact_payload(RedactRequest(payload=report)).payload
    out_dir = PROJECT_ROOT / "agent_workspace" / "yieldmind" / "integrations"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"postgres_redis_running_cancel_{time.strftime('%Y%m%d_%H%M%S')}.json"
    report["report_path"] = str(report_path)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
