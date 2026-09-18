#!/usr/bin/env python3
"""Verify PostgreSQL task authority while Redis/Celery control is unavailable."""

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
from yieldmind.task_queue import cancel_queued_task, get_task_status, redis_health


def main() -> int:
    started = time.time()
    store = YieldMindStore()
    if store.backend != "postgresql":
        raise RuntimeError("Redis outage smoke requires PostgreSQL via YIELDMIND_DATABASE_URL.")

    redis_before = redis_health()
    task, _ = store.create_task(
        task_type="redis_outage_smoke",
        idempotency_key=f"redis-outage-{uuid.uuid4().hex[:16]}",
        payload={"purpose": "verify PostgreSQL authority while Redis is unavailable"},
    )
    task_id = str(task["task_id"])
    store.transition_task(task_id, from_statuses=("created",), to_status="queued")

    before_cancel = get_task_status(task_id, store=store) or {}
    cancellation = cancel_queued_task(
        task_id,
        store=store,
        reason="real Redis outage integration smoke",
        requested_by="redis_outage_smoke",
    ) or {}
    after_cancel = get_task_status(task_id, store=store) or {}
    persisted = store.get_task(task_id) or {}

    checks = {
        "postgres_available": store.ping(),
        "redis_confirmed_unavailable": redis_before.get("ok") is False,
        "celery_query_degraded_before_cancel": before_cancel.get("celery_state") == "UNKNOWN"
        and bool(before_cancel.get("celery_error")),
        "postgres_cancelled_without_redis": persisted.get("status") == "cancelled",
        "cancellation_audit_persisted": persisted.get("cancel_requested_by") == "redis_outage_smoke"
        and bool(persisted.get("cancelled_at")),
        "revoke_failure_reported": cancellation.get("revoke_requested") is False
        and bool(cancellation.get("revoke_error")),
        "celery_query_still_degraded": after_cancel.get("celery_state") == "UNKNOWN"
        and bool(after_cancel.get("celery_error")),
    }
    report = {
        "status": "passed" if all(checks.values()) else "failed",
        "mode": "postgresql_authority_during_real_redis_outage",
        "database_backend": store.backend,
        "real_llm_calls": 0,
        "simulated_model_calls": 0,
        "checks": checks,
        "task_id": task_id,
        "task_status": persisted.get("status"),
        "celery_state": after_cancel.get("celery_state"),
        "redis_error": redis_before.get("error", ""),
        "revoke_error": cancellation.get("revoke_error", ""),
        "duration_seconds": round(time.time() - started, 4),
    }
    report = redact_payload(RedactRequest(payload=report)).payload
    out_dir = PROJECT_ROOT / "agent_workspace" / "yieldmind" / "integrations"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"postgres_redis_outage_{time.strftime('%Y%m%d_%H%M%S')}.json"
    report["report_path"] = str(report_path)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
