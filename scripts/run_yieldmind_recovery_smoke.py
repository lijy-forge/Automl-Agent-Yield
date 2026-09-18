#!/usr/bin/env python3
"""Prepare and verify a real worker-crash lease recovery scenario."""

from __future__ import annotations

import argparse
import json
import os
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
    RecoverTaskRequest,
    enqueue_workflow,
    get_task_status,
    recover_stale_task,
)
from yieldmind.workflow import WorkflowRequest


CONTEXT_PATH = PROJECT_ROOT / "agent_workspace" / "yieldmind" / "integrations" / "recovery_smoke_context.json"
TERMINAL_STATUSES = {"completed", "failed", "dispatch_failed", "cancelled", "interrupted"}


def _write_demo_csv(path: Path) -> None:
    rows = ["sample_id,phi,sp_percent,yield_stress"]
    for index in range(1, 41):
        phi = 0.40 + index * 0.003
        sp = 0.15 + (index % 6) * 0.05
        stress = 8.0 + index * 1.7 + (index % 4) * 0.4
        rows.append(f"recovery_{index},{phi:.4f},{sp:.3f},{stress:.4f}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def prepare(store: YieldMindStore) -> int:
    CONTEXT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fifo_path = CONTEXT_PATH.parent / f"blocked_input_{uuid.uuid4().hex[:10]}.csv"
    os.mkfifo(fifo_path)
    request = EnqueueWorkflowRequest(
        workflow=WorkflowRequest(
            prompt="Worker crash recovery smoke; block on FIFO until the worker is terminated.",
            data_path=str(fifo_path),
            n_samples=20,
            n_splits=2,
        ),
        idempotency_key=f"recovery-parent-{uuid.uuid4().hex[:16]}",
    )
    published = enqueue_workflow(request, store=store)
    task_id = str(published["task"]["task_id"])
    deadline = time.time() + 30
    task = store.get_task(task_id) or {}
    initial_heartbeat = None
    heartbeat_renewed = False
    prepare_stage_persisted = False
    while time.time() < deadline:
        task = store.get_task(task_id) or {}
        if task.get("status") in TERMINAL_STATUSES:
            break
        if task.get("status") == "running" and task.get("heartbeat_at"):
            if initial_heartbeat is None:
                initial_heartbeat = float(task["heartbeat_at"])
            elif float(task["heartbeat_at"]) > initial_heartbeat:
                heartbeat_renewed = True
        run_id = str(task.get("run_id") or "")
        if run_id:
            stages = store.list_stage_executions(run_id)
            prepare_stage_persisted = any(item.get("stage") == "prepare_data" for item in stages)
        if heartbeat_renewed and prepare_stage_persisted:
            break
        time.sleep(0.1)

    context = {
        "task_id": task_id,
        "run_id": str(task.get("run_id") or ""),
        "fifo_path": str(fifo_path),
        "worker_id": str(task.get("worker_id") or ""),
        "initial_heartbeat": initial_heartbeat,
        "latest_heartbeat": task.get("heartbeat_at"),
        "heartbeat_renewed": heartbeat_renewed,
        "prepare_stage_persisted": prepare_stage_persisted,
        "prepared_at": time.time(),
    }
    CONTEXT_PATH.write_text(json.dumps(context, ensure_ascii=False, indent=2), encoding="utf-8")
    checks = {
        "published_to_redis": bool(published.get("published")),
        "task_running": task.get("status") == "running",
        "run_linked": bool(context["run_id"]),
        "worker_identified": bool(context["worker_id"]),
        "heartbeat_renewed_while_node_blocked": heartbeat_renewed,
        "prepare_stage_persisted": prepare_stage_persisted,
    }
    payload = {"phase": "prepare", "status": "passed" if all(checks.values()) else "failed", "checks": checks, **context}
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["status"] == "passed" else 1


def recover(store: YieldMindStore, wait_seconds: float) -> int:
    started = time.time()
    context = json.loads(CONTEXT_PATH.read_text(encoding="utf-8"))
    parent_task_id = str(context["task_id"])
    parent_run_id = str(context["run_id"])
    data_path = Path(context["fifo_path"])
    if data_path.exists():
        data_path.unlink()
    _write_demo_csv(data_path)

    deadline = time.time() + max(1.0, wait_seconds)
    stale = False
    while time.time() < deadline:
        stale = any(item["task_id"] == parent_task_id for item in store.list_stale_tasks())
        if stale:
            break
        time.sleep(0.2)

    recovery = None
    recovery_error = ""
    if stale:
        try:
            recovery = recover_stale_task(
                parent_task_id,
                RecoverTaskRequest(
                    reason="integration smoke confirmed the old worker process exited",
                    requested_by="recovery_smoke",
                    confirmed_worker_stopped=True,
                    idempotency_key=f"recovery-child-{uuid.uuid4().hex[:16]}",
                ),
                store=store,
            )
        except Exception as exc:
            recovery_error = f"{type(exc).__name__}: {exc}"

    child = ((recovery or {}).get("recovery") or {}).get("task") or {}
    child_task_id = str(child.get("task_id") or "")
    latest = get_task_status(child_task_id, store=store) if child_task_id else None
    completion_deadline = time.time() + 120
    while latest and latest["task"]["status"] not in TERMINAL_STATUSES and time.time() < completion_deadline:
        time.sleep(0.25)
        latest = get_task_status(child_task_id, store=store)

    parent = store.get_task(parent_task_id) or {}
    parent_run = store.get_run(parent_run_id) or {}
    child = (latest or {}).get("task") or child
    child_run_id = str(child.get("run_id") or "")
    child_run = store.get_run(child_run_id) if child_run_id else None
    child_metadata = (child_run or {}).get("metadata") or {}
    checks = {
        "old_worker_heartbeat_was_observed": bool(context.get("heartbeat_renewed")),
        "expired_task_detected": stale,
        "parent_task_interrupted": parent.get("status") == "interrupted",
        "parent_run_interrupted": parent_run.get("status") == "interrupted",
        "new_task_created": bool(child_task_id) and child_task_id != parent_task_id,
        "recovery_lineage_persisted": child.get("recovery_of_task_id") == parent_task_id,
        "new_task_completed": child.get("status") == "completed",
        "new_run_completed": (child_run or {}).get("status") == "passed",
        "new_run_not_parent_run": bool(child_run_id) and child_run_id != parent_run_id,
        "run_lineage_persisted": child_metadata.get("recovery_of_task_id") == parent_task_id
        and child_metadata.get("parent_run_id") == parent_run_id,
        "celery_child_success": (latest or {}).get("celery_state") == "SUCCESS",
    }
    report = {
        "status": "passed" if all(checks.values()) else "failed",
        "mode": "postgresql_redis_celery_worker_crash_recovery",
        "real_llm_calls": 0,
        "simulated_model_calls": 0,
        "checks": checks,
        "parent_task_id": parent_task_id,
        "parent_task_status": parent.get("status"),
        "parent_run_id": parent_run_id,
        "parent_run_status": parent_run.get("status"),
        "child_task_id": child_task_id,
        "child_task_status": child.get("status"),
        "child_run_id": child_run_id,
        "child_run_status": (child_run or {}).get("status"),
        "celery_state": (latest or {}).get("celery_state"),
        "recovery_error": recovery_error,
        "duration_seconds": round(time.time() - started, 4),
    }
    report = redact_payload(RedactRequest(payload=report)).payload
    report_path = CONTEXT_PATH.parent / f"postgres_redis_recovery_{time.strftime('%Y%m%d_%H%M%S')}.json"
    report["report_path"] = str(report_path)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="phase", required=True)
    subparsers.add_parser("prepare")
    recover_parser = subparsers.add_parser("recover")
    recover_parser.add_argument("--wait-seconds", type=float, default=30.0)
    args = parser.parse_args()
    store = YieldMindStore()
    if store.backend != "postgresql":
        raise RuntimeError("Recovery smoke requires PostgreSQL via YIELDMIND_DATABASE_URL.")
    if args.phase == "prepare":
        return prepare(store)
    return recover(store, args.wait_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
