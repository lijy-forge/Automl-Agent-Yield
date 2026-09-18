#!/usr/bin/env python3
"""Verify legal task transitions and terminal-state protection."""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from yieldmind.database import YieldMindStore


def main() -> int:
    store = YieldMindStore()
    task, _ = store.create_task(
        task_type="transition_smoke",
        idempotency_key=f"transition-smoke-{uuid.uuid4().hex}",
        payload={},
    )
    task_id = str(task["task_id"])
    transitions = [
        store.transition_task(task_id, from_statuses=("created",), to_status="queued"),
        store.transition_task(task_id, from_statuses=("queued",), to_status="running"),
        store.transition_task(task_id, from_statuses=("running",), to_status="completed"),
    ]
    reversal_error = ""
    try:
        store.transition_task(task_id, from_statuses=("completed",), to_status="running")
        reversal_blocked = False
    except ValueError as exc:
        reversal_blocked = True
        reversal_error = str(exc)
    final_status = str((store.get_task(task_id) or {}).get("status") or "")
    checks = {
        "legal_transitions_succeeded": all(transitions),
        "terminal_reversal_blocked": reversal_blocked,
        "final_status_completed": final_status == "completed",
    }
    report = {
        "status": "passed" if all(checks.values()) else "failed",
        "database_backend": store.backend,
        "checks": checks,
        "transitions": transitions,
        "reversal_error": reversal_error,
        "final_status": final_status,
        "task_id": task_id,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
