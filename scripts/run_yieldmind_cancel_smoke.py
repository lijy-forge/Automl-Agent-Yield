#!/usr/bin/env python3
"""Prepare or verify a real queued-task cancellation smoke test."""

from __future__ import annotations

import argparse
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


def _write_report(payload: dict) -> dict:
    report = redact_payload(RedactRequest(payload=payload)).payload
    out_dir = PROJECT_ROOT / "agent_workspace" / "yieldmind" / "integrations"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"postgres_redis_cancel_{time.strftime('%Y%m%d_%H%M%S')}.json"
    report["report_path"] = str(report_path)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def prepare(store: YieldMindStore) -> int:
    request = EnqueueWorkflowRequest(
        workflow=WorkflowRequest(
            prompt="Queued cancellation integration smoke; this workflow must not execute.",
            n_samples=20,
            n_splits=2,
        ),
        idempotency_key=f"cancel-smoke-{uuid.uuid4().hex[:16]}",
    )
    published = enqueue_workflow(request, store=store)
    task_id = str(published["task"]["task_id"])
    queued_before_cancel = str((store.get_task(task_id) or {}).get("status") or "") == "queued"
    cancelled = cancel_queued_task(task_id, store=store)
    task = store.get_task(task_id) or {}
    checks = {
        "published_to_redis": bool(published.get("published")),
        "queued_before_cancel": queued_before_cancel,
        "database_cancelled": task.get("status") == "cancelled",
        "no_run_created": not task.get("run_id"),
        "revoke_requested": bool((cancelled or {}).get("revoke_requested")),
    }
    report = _write_report(
        {
            "phase": "prepare",
            "status": "passed" if all(checks.values()) else "failed",
            "database_backend": store.backend,
            "real_llm_calls": 0,
            "simulated_model_calls": 0,
            "checks": checks,
            "task_id": task_id,
            "task_status": task.get("status"),
        }
    )
    return 0 if report["status"] == "passed" else 1


def verify(store: YieldMindStore, task_id: str, wait_seconds: float) -> int:
    deadline = time.time() + max(0.0, wait_seconds)
    latest = get_task_status(task_id, store=store)
    while latest and time.time() < deadline and latest.get("celery_state") == "PENDING":
        time.sleep(0.25)
        latest = get_task_status(task_id, store=store)
    task = (latest or {}).get("task") or {}
    checks = {
        "database_still_cancelled": task.get("status") == "cancelled",
        "worker_did_not_create_run": not task.get("run_id"),
        "worker_did_not_write_result": not task.get("result"),
        "celery_state_revoked": (latest or {}).get("celery_state") == "REVOKED",
    }
    report = _write_report(
        {
            "phase": "verify_after_worker_start",
            "status": "passed" if all(checks.values()) else "failed",
            "database_backend": store.backend,
            "real_llm_calls": 0,
            "simulated_model_calls": 0,
            "checks": checks,
            "task_id": task_id,
            "task_status": task.get("status"),
            "celery_state": (latest or {}).get("celery_state"),
        }
    )
    return 0 if report["status"] == "passed" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="phase", required=True)
    subparsers.add_parser("prepare")
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--task-id", required=True)
    verify_parser.add_argument("--wait-seconds", type=float, default=5.0)
    args = parser.parse_args()

    store = YieldMindStore()
    if store.backend != "postgresql":
        raise RuntimeError("Cancellation integration smoke requires PostgreSQL via YIELDMIND_DATABASE_URL.")
    if args.phase == "prepare":
        return prepare(store)
    return verify(store, args.task_id, args.wait_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
