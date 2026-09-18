"""Redis/Celery dispatch with PostgreSQL/SQLite task records as status authority."""

from __future__ import annotations

import os
import socket
import threading
import time
import uuid
from typing import Any

from celery import Celery
from celery.exceptions import Ignore
from celery.result import AsyncResult
from pydantic import BaseModel, Field
from redis import Redis

from yieldmind.database import YieldMindStore
from yieldmind.domain_workflow import DomainWorkflowRequest, run_domain_workflow
from yieldmind.safety import RedactRequest, redact_payload
from yieldmind.workflow import WorkflowRequest, run_workflow


DEFAULT_REDIS_URL = "redis://:yieldmind_redis_dev_only@127.0.0.1:6379/0"


def redis_url() -> str:
    return os.environ.get("YIELDMIND_REDIS_URL", DEFAULT_REDIS_URL)


celery_app = Celery("yieldmind", broker=redis_url(), backend=redis_url())
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    result_expires=3600,
    broker_connection_retry_on_startup=True,
    worker_prefetch_multiplier=1,
)


class EnqueueWorkflowRequest(BaseModel):
    workflow: WorkflowRequest = Field(default_factory=WorkflowRequest)
    idempotency_key: str = Field(default_factory=lambda: f"workflow_{uuid.uuid4().hex[:16]}", min_length=8, max_length=128)


class EnqueueDomainWorkflowRequest(BaseModel):
    workflow: DomainWorkflowRequest = Field(default_factory=DomainWorkflowRequest)
    idempotency_key: str = Field(
        default_factory=lambda: f"domain_{uuid.uuid4().hex[:16]}",
        min_length=8,
        max_length=128,
    )


class CancelTaskRequest(BaseModel):
    reason: str = Field(default="", max_length=500)
    requested_by: str = Field(default="api_user", min_length=1, max_length=128)


class RecoverTaskRequest(BaseModel):
    reason: str = Field(min_length=1, max_length=500)
    requested_by: str = Field(default="api_user", min_length=1, max_length=128)
    confirmed_worker_stopped: bool = False
    idempotency_key: str = Field(
        default_factory=lambda: f"recovery_{uuid.uuid4().hex[:16]}",
        min_length=8,
        max_length=128,
    )


class TaskCancellationConflict(RuntimeError):
    """Raised when cancellation loses to a terminal task transition."""


class TaskRecoveryConflict(RuntimeError):
    """Raised when a task is not eligible for explicit stale-task recovery."""


def task_lease_seconds() -> float:
    return max(10.0, float(os.environ.get("YIELDMIND_TASK_LEASE_SECONDS", "30")))


def task_heartbeat_seconds(lease_seconds: float) -> float:
    configured = max(1.0, float(os.environ.get("YIELDMIND_TASK_HEARTBEAT_SECONDS", "5")))
    return min(configured, lease_seconds / 3.0)


class _TaskLeaseHeartbeat:
    def __init__(self, store: YieldMindStore, task_id: str, worker_id: str, lease_seconds: float) -> None:
        self.store = store
        self.task_id = task_id
        self.worker_id = worker_id
        self.lease_seconds = lease_seconds
        self.interval_seconds = task_heartbeat_seconds(lease_seconds)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"lease-{task_id}", daemon=True)
        self.lost = False
        self.last_error = ""

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=self.interval_seconds + 1.0)

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            try:
                if not self.store.renew_task_lease(
                    self.task_id,
                    worker_id=self.worker_id,
                    lease_seconds=self.lease_seconds,
                ):
                    self.lost = True
                    return
                self.last_error = ""
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"


def redis_client() -> Redis:
    return Redis.from_url(redis_url(), decode_responses=True, socket_connect_timeout=2, socket_timeout=2)


def redis_health() -> dict[str, Any]:
    try:
        client = redis_client()
        return {"ok": bool(client.ping()), "backend": "redis", "url_configured": True}
    except Exception as exc:
        return {"ok": False, "backend": "redis", "url_configured": True, "error": f"{type(exc).__name__}: {exc}"}


def _run_claimed_workflow_task(
    self: Any,
    task_id: str,
    workflow_payload: dict[str, Any],
    *,
    domain_workflow: bool,
) -> dict[str, Any]:
    store = YieldMindStore()
    lease_seconds = task_lease_seconds()
    hostname = str(getattr(self.request, "hostname", "") or socket.gethostname())
    worker_id = f"{hostname}:{os.getpid()}:{task_id[-8:]}"
    claimed = store.claim_task(task_id, worker_id=worker_id, lease_seconds=lease_seconds)
    if not claimed:
        task = store.get_task(task_id)
        summary = {
            "run_id": str((task or {}).get("run_id") or ""),
            "status": str((task or {}).get("status") or "not_claimed"),
            "errors": ["Task was not in queued state; worker did not execute it."],
            "artifacts": {},
            "workflow_backend": "",
        }
        if summary["status"] == "cancelled":
            self.backend.mark_as_revoked(
                task_id,
                reason="Task was cancelled before worker claim.",
                request=self.request,
            )
        raise Ignore()
    heartbeat = _TaskLeaseHeartbeat(store, task_id, worker_id, lease_seconds)
    heartbeat.start()
    try:
        def attach_run(run_id: str) -> None:
            if not store.attach_task_run(task_id, run_id):
                raise RuntimeError("Task could not be linked to its workflow run.")
            task = store.get_task(task_id) or {}
            recovery_of_task_id = str(task.get("recovery_of_task_id") or "")
            if recovery_of_task_id:
                parent_task = store.get_task(recovery_of_task_id) or {}
                run = store.get_run(run_id) or {}
                metadata = dict(run.get("metadata") or {})
                metadata.update(
                    {
                        "recovery_of_task_id": recovery_of_task_id,
                        "parent_run_id": str(parent_task.get("run_id") or ""),
                    }
                )
                store.update_run(run_id, metadata=metadata)

        def cancellation_requested() -> bool:
            if heartbeat.lost:
                raise RuntimeError("Task lease ownership was lost while the workflow was running.")
            task = store.get_task(task_id)
            return str((task or {}).get("status") or "") == "cancel_requested"

        try:
            if domain_workflow:
                result = run_domain_workflow(
                    DomainWorkflowRequest.model_validate(workflow_payload),
                    store=store,
                    cancel_check=cancellation_requested,
                    on_run_started=attach_run,
                )
            else:
                result = run_workflow(
                    WorkflowRequest.model_validate(workflow_payload),
                    store=store,
                    cancel_check=cancellation_requested,
                    on_run_started=attach_run,
                )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            store.transition_task(task_id, from_statuses=("running", "cancel_requested"), to_status="failed", error=error)
            raise

        summary = {
            "run_id": result.get("run_id", ""),
            "status": result.get("status", "failed"),
            "errors": result.get("errors", []),
            "artifacts": result.get("artifacts", {}),
            "workflow_backend": result.get("workflow_backend", ""),
            "process_control": result.get("process_control", {}),
            "model_call_mode": result.get("model_call_mode", "offline_deterministic"),
            "real_llm_calls": result.get("real_llm_calls", 0),
            "simulated_model_calls": result.get("simulated_model_calls", 0),
        }
        summary = redact_payload(RedactRequest(payload=summary)).payload
        run_id = str(result.get("run_id") or "")
        if result.get("status") == "cancelled":
            if not store.complete_task_cancellation(task_id, result=summary, run_id=run_id):
                raise RuntimeError("Task cancellation could not be finalized from cancel_requested state.")
            try:
                self.backend.mark_as_revoked(
                    task_id,
                    reason="Task stopped cooperatively at a workflow node boundary.",
                    request=self.request,
                )
            finally:
                raise Ignore()

        task_status = "completed" if result.get("status") == "passed" else "failed"
        updated = store.transition_task(
            task_id,
            from_statuses=("running", "cancel_requested"),
            to_status=task_status,
            result=summary,
            run_id=run_id,
            error="; ".join(result.get("errors", [])),
        )
        if not updated:
            raise RuntimeError("Task left an active state before workflow completion could be recorded.")
        return summary
    finally:
        heartbeat.stop()


@celery_app.task(bind=True, name="yieldmind.run_offline_workflow")
def run_offline_workflow_task(self: Any, task_id: str, workflow_payload: dict[str, Any]) -> dict[str, Any]:
    return _run_claimed_workflow_task(
        self,
        task_id,
        workflow_payload,
        domain_workflow=False,
    )


@celery_app.task(bind=True, name="yieldmind.run_domain_workflow")
def run_domain_workflow_task(self: Any, task_id: str, workflow_payload: dict[str, Any]) -> dict[str, Any]:
    return _run_claimed_workflow_task(
        self,
        task_id,
        workflow_payload,
        domain_workflow=True,
    )


def enqueue_workflow(
    request: EnqueueWorkflowRequest,
    *,
    store: YieldMindStore,
    recovery_of_task_id: str = "",
) -> dict[str, Any]:
    health = redis_health()
    if not health["ok"]:
        raise ConnectionError(health.get("error", "Redis is unavailable."))
    task, idempotent = store.create_task(
        task_type="offline_workflow",
        idempotency_key=request.idempotency_key,
        payload=request.workflow.model_dump(),
        recovery_of_task_id=recovery_of_task_id,
    )
    if idempotent:
        return {
            "task": task,
            "idempotent": True,
            "published": task.get("status") in {
                "queued",
                "running",
                "cancel_requested",
                "cancelled",
                "completed",
                "failed",
            },
        }
    task_id = str(task["task_id"])
    if not store.transition_task(task_id, from_statuses=("created",), to_status="queued"):
        raise RuntimeError(f"Task {task_id} could not transition from created to queued.")
    try:
        run_offline_workflow_task.apply_async(
            args=[task_id, request.workflow.model_dump()],
            task_id=task_id,
            queue="yieldmind",
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        store.transition_task(task_id, from_statuses=("queued",), to_status="dispatch_failed", error=error)
        raise ConnectionError(error) from exc
    return {"task": store.get_task(task_id), "idempotent": False, "published": True}


def enqueue_domain_workflow(
    request: EnqueueDomainWorkflowRequest,
    *,
    store: YieldMindStore,
    recovery_of_task_id: str = "",
) -> dict[str, Any]:
    health = redis_health()
    if not health["ok"]:
        raise ConnectionError(health.get("error", "Redis is unavailable."))
    payload = request.workflow.model_dump()
    task, idempotent = store.create_task(
        task_type="domain_workflow",
        idempotency_key=request.idempotency_key,
        payload=payload,
        recovery_of_task_id=recovery_of_task_id,
    )
    if idempotent:
        return {
            "task": task,
            "idempotent": True,
            "published": task.get("status") in {
                "queued",
                "running",
                "cancel_requested",
                "cancelled",
                "completed",
                "failed",
            },
        }
    task_id = str(task["task_id"])
    if not store.transition_task(task_id, from_statuses=("created",), to_status="queued"):
        raise RuntimeError(f"Task {task_id} could not transition from created to queued.")
    try:
        run_domain_workflow_task.apply_async(
            args=[task_id, payload],
            task_id=task_id,
            queue="yieldmind",
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        store.transition_task(task_id, from_statuses=("queued",), to_status="dispatch_failed", error=error)
        raise ConnectionError(error) from exc
    return {"task": store.get_task(task_id), "idempotent": False, "published": True}


def get_task_status(task_id: str, *, store: YieldMindStore) -> dict[str, Any] | None:
    task = store.get_task(task_id)
    if not task:
        return None
    try:
        celery_state = AsyncResult(task_id, app=celery_app).state
        celery_error = ""
    except Exception as exc:
        celery_state = "UNKNOWN"
        celery_error = f"{type(exc).__name__}: {exc}"
    return {"task": task, "celery_state": celery_state, "celery_error": celery_error}


def cancel_queued_task(
    task_id: str,
    *,
    store: YieldMindStore,
    reason: str = "",
    requested_by: str = "api_user",
) -> dict[str, Any] | None:
    """Cancel queued work immediately or request cooperative running cancellation."""
    task = store.get_task(task_id)
    if task is None:
        return None
    status = str(task.get("status") or "")
    if status == "cancelled":
        return {
            "task": task,
            "idempotent": True,
            "cancellation_pending": False,
            "revoke_requested": False,
            "revoke_error": "",
        }
    if status == "cancel_requested":
        return {
            "task": task,
            "idempotent": True,
            "cancellation_pending": True,
            "revoke_requested": False,
            "revoke_error": "",
        }
    if status == "queued":
        changed = store.cancel_queued_task_record(task_id, reason=reason, requested_by=requested_by)
    elif status == "running":
        changed = store.request_running_task_cancellation(task_id, reason=reason, requested_by=requested_by)
    else:
        raise TaskCancellationConflict(f"Task cannot be cancelled from terminal status {status}.")

    if not changed:
        latest = store.get_task(task_id)
        latest_status = str((latest or {}).get("status") or "unknown")
        if latest_status in {"cancelled", "cancel_requested"}:
            return {
                "task": latest,
                "idempotent": True,
                "cancellation_pending": latest_status == "cancel_requested",
                "revoke_requested": False,
                "revoke_error": "",
            }
        raise TaskCancellationConflict(
            f"Task left cancellable state before cancellation completed; current status is {latest_status}."
        )

    revoke_error = ""
    try:
        celery_app.control.revoke(task_id, terminate=False)
        revoke_requested = True
    except Exception as exc:
        revoke_requested = False
        revoke_error = f"{type(exc).__name__}: {exc}"
    return {
        "task": store.get_task(task_id),
        "idempotent": False,
        "cancellation_pending": status == "running",
        "revoke_requested": revoke_requested,
        "revoke_error": revoke_error,
    }


def recover_stale_task(
    task_id: str,
    request: RecoverTaskRequest,
    *,
    store: YieldMindStore,
) -> dict[str, Any] | None:
    """Interrupt one expired task and enqueue a new, explicitly linked attempt."""
    task = store.get_task(task_id)
    if task is None:
        return None

    existing = store.get_task_by_idempotency_key(request.idempotency_key)
    if existing is not None:
        if str(existing.get("recovery_of_task_id") or "") != task_id:
            raise TaskRecoveryConflict("Recovery idempotency key is already used by another task.")
        return {
            "interrupted_task": task,
            "recovery": {"task": existing, "idempotent": True, "published": existing.get("status") != "created"},
            "idempotent": True,
        }

    if not request.confirmed_worker_stopped:
        raise TaskRecoveryConflict("confirmed_worker_stopped=true is required before stale-task recovery.")
    status = str(task.get("status") or "")
    if status not in {"running", "cancel_requested"}:
        raise TaskRecoveryConflict(f"Task cannot be recovered from status {status}.")
    lease_expires_at = task.get("lease_expires_at")
    if lease_expires_at is None or float(lease_expires_at) >= time.time():
        raise TaskRecoveryConflict("Task lease has not expired; recovery would risk overlapping execution.")
    health = redis_health()
    if not health.get("ok"):
        raise ConnectionError(health.get("error", "Redis is unavailable."))

    interrupted = store.interrupt_stale_task(
        task_id,
        reason=request.reason,
        requested_by=request.requested_by,
        confirmed_worker_stopped=True,
    )
    if not interrupted:
        latest = store.get_task(task_id) or {}
        raise TaskRecoveryConflict(
            f"Task changed before recovery acquired it; current status is {latest.get('status', 'unknown')}."
        )

    interrupted_task = store.get_task(task_id) or {}
    run_id = str(interrupted_task.get("run_id") or "")
    if run_id:
        run = store.get_run(run_id)
        if run is not None:
            result = dict(run.get("result") or {})
            result["interruption"] = {
                "reason": request.reason,
                "requested_by": request.requested_by,
                "task_id": task_id,
            }
            store.add_event(
                run_id,
                stage="recovery",
                level="error",
                message="Task lease expired; operator confirmed the old worker stopped.",
                payload=result["interruption"],
            )
            store.update_run(run_id, status="interrupted", result=result, completed=True)

    if str(interrupted_task.get("task_type") or "") == "domain_workflow":
        recovery = enqueue_domain_workflow(
            EnqueueDomainWorkflowRequest(
                workflow=DomainWorkflowRequest.model_validate(interrupted_task.get("payload") or {}),
                idempotency_key=request.idempotency_key,
            ),
            store=store,
            recovery_of_task_id=task_id,
        )
    else:
        recovery = enqueue_workflow(
            EnqueueWorkflowRequest(
                workflow=WorkflowRequest.model_validate(interrupted_task.get("payload") or {}),
                idempotency_key=request.idempotency_key,
            ),
            store=store,
            recovery_of_task_id=task_id,
        )
    return {
        "interrupted_task": store.get_task(task_id),
        "recovery": recovery,
        "idempotent": False,
    }
