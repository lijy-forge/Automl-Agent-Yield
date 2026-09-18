#!/usr/bin/env python3
"""Verify real PostgreSQL/Redis/Celery dispatch for the domain StateGraph."""

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
from yieldmind.domain_workflow import DomainWorkflowRequest
from yieldmind.safety import RedactRequest, redact_payload
from yieldmind.task_queue import EnqueueDomainWorkflowRequest, enqueue_domain_workflow, get_task_status


TERMINAL_STATUSES = {"cancelled", "completed", "failed", "dispatch_failed"}


def main() -> int:
    started = time.time()
    store = YieldMindStore()
    if store.backend != "postgresql":
        raise RuntimeError("Domain queue smoke requires PostgreSQL via YIELDMIND_DATABASE_URL.")

    request = EnqueueDomainWorkflowRequest(
        workflow=DomainWorkflowRequest(
            run_dir=str(PROJECT_ROOT / "agent_workspace" / "runs" / f"domain_queue_gate_{uuid.uuid4().hex[:8]}"),
            allow_live_llm=False,
        ),
        idempotency_key=f"domain-queue-smoke-{uuid.uuid4().hex[:16]}",
    )
    published = enqueue_domain_workflow(request, store=store)
    repeated = enqueue_domain_workflow(request, store=store)
    task_id = str(published["task"]["task_id"])

    latest = get_task_status(task_id, store=store)
    deadline = time.time() + 60
    while latest and latest["task"]["status"] not in TERMINAL_STATUSES and time.time() < deadline:
        time.sleep(0.2)
        latest = get_task_status(task_id, store=store)

    task = (latest or {}).get("task") or {}
    run_id = str(task.get("run_id") or "")
    run = store.get_run(run_id) if run_id else None
    run_result = (run or {}).get("result") or {}
    stages = [str(item.get("stage") or "") for item in run_result.get("stages", [])]
    task_result = task.get("result") or {}
    checks = {
        "published_to_redis": bool(published.get("published")),
        "idempotent_repeat": bool(repeated.get("idempotent")) and repeated["task"]["task_id"] == task_id,
        "domain_task_type": task.get("task_type") == "domain_workflow",
        "worker_consumed_task": (latest or {}).get("celery_state") == "SUCCESS",
        "live_gate_rejected_business_run": task.get("status") == "failed" and (run or {}).get("status") == "failed",
        "run_link_persisted": bool(run_id) and bool(run),
        "stategraph_trace_persisted": stages == ["start", "finish"],
        "live_model_not_called": (
            run_result.get("model_call_mode") == "live_blocked"
            and run_result.get("real_llm_calls") == 0
            and task_result.get("real_llm_calls") == 0
        ),
        "explicit_gate_reason": any("allow_live_llm=true" in str(item) for item in run_result.get("errors", [])),
    }
    report = {
        "status": "passed" if all(checks.values()) else "failed",
        "mode": "postgresql_redis_celery_domain_authorization_gate",
        "real_llm_calls": 0,
        "simulated_model_calls": 0,
        "checks": checks,
        "task_id": task_id,
        "task_status": task.get("status"),
        "celery_state": (latest or {}).get("celery_state"),
        "run_id": run_id,
        "run_status": (run or {}).get("status"),
        "stage_names": stages,
        "duration_seconds": round(time.time() - started, 4),
    }
    report = redact_payload(RedactRequest(payload=report)).payload
    out_dir = PROJECT_ROOT / "agent_workspace" / "yieldmind" / "integrations"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"postgres_redis_domain_queue_{time.strftime('%Y%m%d_%H%M%S')}.json"
    report["report_path"] = str(report_path)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
