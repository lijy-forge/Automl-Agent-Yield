from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import yieldmind.api as yieldmind_api
from yieldmind.checkpointing import psycopg_connection_string
from yieldmind.database import YieldMindStore, connect
from yieldmind.function_calling import (
    SIMULATED_TEST_ADAPTER,
    ToolCallProtocolError,
    ToolPlanRequest,
    execute_plan,
    local_rule_plan,
    plan_tools,
)
from yieldmind.knowledge_base import EmbeddingProfile, KnowledgeBase, KnowledgeIngestRequest, KnowledgeSearchRequest
from yieldmind.memory import AddMessageRequest, CreateSessionRequest, SessionMemoryStore, UpsertMemoryRequest
from yieldmind.sandbox import DockerSandbox, DockerSandboxCommand, build_docker_argv
from yieldmind.task_queue import (
    EnqueueWorkflowRequest,
    RecoverTaskRequest,
    TaskRecoveryConflict,
    cancel_queued_task,
    enqueue_workflow,
    get_task_status,
    recover_stale_task,
)
from yieldmind.tools import ToolRegistry, ToolResult
from yieldmind.workflow import WorkflowRequest, YieldMindWorkflow, run_workflow


def _write_small_yield_csv(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "sample_id,phi,sp_percent,yield_stress",
                "s1,0.42,0.2,12.0",
                "s2,0.44,0.2,14.2",
                "s3,0.46,0.3,18.1",
                "s4,0.48,0.4,25.4",
                "s5,0.50,0.5,35.0",
                "s6,0.52,0.5,48.0",
            ]
        ),
        encoding="utf-8",
    )


def test_database_run_lifecycle(tmp_path: Path) -> None:
    store = YieldMindStore(tmp_path / "yieldmind.sqlite3")
    run_id = store.create_run(mode="offline_deterministic", source="pytest")
    store.add_event(run_id, stage="test", message="hello")
    store.update_run(run_id, status="passed", result={"ok": True}, completed=True)
    run = store.get_run(run_id)
    assert run is not None
    assert run["status"] == "passed"
    assert store.list_events(run_id)[0]["message"] == "hello"


def test_postgres_checkpoint_url_uses_psycopg_compatible_scheme() -> None:
    assert (
        psycopg_connection_string("postgresql+psycopg://user:pass@localhost/db")
        == "postgresql://user:pass@localhost/db"
    )


def test_task_record_idempotency(tmp_path: Path) -> None:
    store = YieldMindStore(tmp_path / "yieldmind.sqlite3")
    first, first_idempotent = store.create_task(
        task_type="offline_workflow",
        idempotency_key="pytest-task-key",
        payload={"n_samples": 20},
    )
    second, second_idempotent = store.create_task(
        task_type="offline_workflow",
        idempotency_key="pytest-task-key",
        payload={"n_samples": 999},
    )
    assert first_idempotent is False
    assert second_idempotent is True
    assert first["task_id"] == second["task_id"]
    assert second["payload"] == {"n_samples": 20}
    assert store.transition_task(first["task_id"], from_statuses=("created",), to_status="queued")
    assert store.transition_task(first["task_id"], from_statuses=("queued",), to_status="running")
    assert store.transition_task(
        first["task_id"],
        from_statuses=("running",),
        to_status="completed",
        result={"ok": True},
        run_id="run_test",
    )
    updated = store.get_task(first["task_id"])
    assert updated is not None
    assert updated["status"] == "completed"
    assert updated["result"] == {"ok": True}


def test_task_transitions_are_atomic_and_terminal_safe(tmp_path: Path) -> None:
    store = YieldMindStore(tmp_path / "yieldmind.sqlite3")
    task, _ = store.create_task(task_type="offline_workflow", idempotency_key="transition-key", payload={})
    task_id = task["task_id"]

    assert store.transition_task(task_id, from_statuses=("created",), to_status="queued")
    assert not store.transition_task(task_id, from_statuses=("created",), to_status="queued")
    assert store.transition_task(task_id, from_statuses=("queued",), to_status="running")
    assert store.transition_task(task_id, from_statuses=("running",), to_status="completed", result={"ok": True})
    with pytest.raises(ValueError, match="Illegal task status transition"):
        store.transition_task(task_id, from_statuses=("completed",), to_status="running")
    assert store.get_task(task_id)["status"] == "completed"


def test_task_lease_heartbeat_and_manual_interruption_guard(tmp_path: Path) -> None:
    store = YieldMindStore(tmp_path / "yieldmind.sqlite3")
    task, _ = store.create_task(task_type="offline_workflow", idempotency_key="lease-key", payload={})
    task_id = str(task["task_id"])
    assert store.transition_task(task_id, from_statuses=("created",), to_status="queued")
    assert store.claim_task(task_id, worker_id="worker-a", lease_seconds=30, claimed_at=100)
    claimed = store.get_task(task_id)
    assert claimed is not None
    assert claimed["worker_id"] == "worker-a"
    assert claimed["heartbeat_at"] == 100
    assert claimed["lease_expires_at"] == 130
    assert not store.renew_task_lease(task_id, worker_id="worker-b", lease_seconds=30, heartbeat_at=110)
    assert store.renew_task_lease(task_id, worker_id="worker-a", lease_seconds=30, heartbeat_at=110)
    assert store.list_stale_tasks(now=139) == []
    assert store.list_stale_tasks(now=141)[0]["task_id"] == task_id
    with pytest.raises(ValueError, match="confirmed_worker_stopped=true"):
        store.interrupt_stale_task(
            task_id,
            reason="worker disappeared",
            requested_by="pytest",
            confirmed_worker_stopped=False,
            interrupted_at=141,
        )
    assert store.interrupt_stale_task(
        task_id,
        reason="worker disappeared",
        requested_by="pytest",
        confirmed_worker_stopped=True,
        interrupted_at=141,
    )
    interrupted = store.get_task(task_id)
    assert interrupted is not None and interrupted["status"] == "interrupted"
    assert interrupted["recovery_reason"] == "worker disappeared"
    assert interrupted["lease_expires_at"] is None
    with pytest.raises(ValueError, match="Illegal task status transition"):
        store.transition_task(task_id, from_statuses=("interrupted",), to_status="running")


def test_stale_task_recovery_creates_linked_new_task_and_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = YieldMindStore(tmp_path / "yieldmind.sqlite3")
    workflow = WorkflowRequest(n_samples=20, n_splits=2)
    parent, _ = store.create_task(
        task_type="offline_workflow",
        idempotency_key="stale-parent",
        payload=workflow.model_dump(),
    )
    parent_id = str(parent["task_id"])
    assert store.transition_task(parent_id, from_statuses=("created",), to_status="queued")
    assert store.claim_task(parent_id, worker_id="dead-worker", lease_seconds=1, claimed_at=1)
    parent_run_id = store.create_run(mode="offline_deterministic", source="pytest", status="running")
    assert store.attach_task_run(parent_id, parent_run_id)
    monkeypatch.setattr("yieldmind.task_queue.redis_health", lambda: {"ok": True})
    published: list[str] = []
    monkeypatch.setattr(
        "yieldmind.task_queue.run_offline_workflow_task.apply_async",
        lambda *, args, task_id, queue: published.append(task_id),
    )
    request = RecoverTaskRequest(
        reason="worker process exited",
        requested_by="pytest",
        confirmed_worker_stopped=True,
        idempotency_key="recovery-child-key",
    )

    result = recover_stale_task(parent_id, request, store=store)
    repeated = recover_stale_task(parent_id, request, store=store)

    assert result is not None and result["interrupted_task"]["status"] == "interrupted"
    child = result["recovery"]["task"]
    assert child["status"] == "queued"
    assert child["task_id"] != parent_id
    assert child["recovery_of_task_id"] == parent_id
    assert published == [child["task_id"]]
    assert repeated is not None and repeated["idempotent"] is True
    assert repeated["recovery"]["task"]["task_id"] == child["task_id"]
    assert store.get_run(parent_run_id)["status"] == "interrupted"


def test_stale_task_recovery_rejects_live_lease_and_missing_confirmation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = YieldMindStore(tmp_path / "yieldmind.sqlite3")
    task, _ = store.create_task(
        task_type="offline_workflow",
        idempotency_key="live-parent",
        payload=WorkflowRequest(n_samples=20, n_splits=2).model_dump(),
    )
    task_id = str(task["task_id"])
    assert store.transition_task(task_id, from_statuses=("created",), to_status="queued")
    assert store.claim_task(task_id, worker_id="live-worker", lease_seconds=3600)
    monkeypatch.setattr("yieldmind.task_queue.redis_health", lambda: {"ok": True})

    with pytest.raises(TaskRecoveryConflict, match="confirmed_worker_stopped=true"):
        recover_stale_task(
            task_id,
            RecoverTaskRequest(reason="unsafe", idempotency_key="missing-confirmation"),
            store=store,
        )
    with pytest.raises(TaskRecoveryConflict, match="lease has not expired"):
        recover_stale_task(
            task_id,
            RecoverTaskRequest(
                reason="unsafe",
                confirmed_worker_stopped=True,
                idempotency_key="live-lease-recovery",
            ),
            store=store,
        )


def test_stale_task_recovery_api_lists_and_recovers_expired_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = YieldMindStore(tmp_path / "yieldmind.sqlite3")
    monkeypatch.setattr(yieldmind_api, "store", store)
    monkeypatch.setattr("yieldmind.task_queue.redis_health", lambda: {"ok": True})
    monkeypatch.setattr("yieldmind.task_queue.run_offline_workflow_task.apply_async", lambda **kwargs: None)
    task, _ = store.create_task(
        task_type="offline_workflow",
        idempotency_key="api-stale-parent",
        payload=WorkflowRequest(n_samples=20, n_splits=2).model_dump(),
    )
    task_id = str(task["task_id"])
    assert store.transition_task(task_id, from_statuses=("created",), to_status="queued")
    assert store.claim_task(task_id, worker_id="dead-api-worker", lease_seconds=1, claimed_at=1)
    client = TestClient(yieldmind_api.app)

    stale_response = client.get("/api/task-recovery/stale")
    rejected = client.post(
        f"/api/tasks/{task_id}/recover",
        json={"reason": "operator has not checked worker", "idempotency_key": "api-recovery-rejected"},
    )
    accepted = client.post(
        f"/api/tasks/{task_id}/recover",
        json={
            "reason": "worker confirmed stopped",
            "requested_by": "pytest-api",
            "confirmed_worker_stopped": True,
            "idempotency_key": "api-recovery-accepted",
        },
    )

    assert stale_response.status_code == 200
    assert [item["task_id"] for item in stale_response.json()["tasks"]] == [task_id]
    assert rejected.status_code == 409
    assert "confirmed_worker_stopped=true" in rejected.json()["detail"]
    assert accepted.status_code == 202
    body = accepted.json()
    assert body["interrupted_task"]["status"] == "interrupted"
    assert body["recovery"]["task"]["recovery_of_task_id"] == task_id


def test_enqueue_cannot_overwrite_fast_worker_terminal_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = YieldMindStore(tmp_path / "yieldmind.sqlite3")
    monkeypatch.setattr("yieldmind.task_queue.redis_health", lambda: {"ok": True})

    def complete_synchronously(*, args: list, task_id: str, queue: str) -> None:
        assert queue == "yieldmind"
        persisted_task_id = str(args[0])
        assert persisted_task_id == task_id
        assert store.transition_task(task_id, from_statuses=("queued",), to_status="running")
        assert store.transition_task(task_id, from_statuses=("running",), to_status="completed", result={"ok": True})

    monkeypatch.setattr("yieldmind.task_queue.run_offline_workflow_task.apply_async", complete_synchronously)
    response = enqueue_workflow(
        EnqueueWorkflowRequest(
            workflow=WorkflowRequest(n_samples=20, n_splits=2),
            idempotency_key="fast-worker-key",
        ),
        store=store,
    )

    assert response["task"]["status"] == "completed"


def test_cancel_queued_task_is_idempotent_and_blocks_worker_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = YieldMindStore(tmp_path / "yieldmind.sqlite3")
    task, _ = store.create_task(task_type="offline_workflow", idempotency_key="cancel-queued-key", payload={})
    task_id = str(task["task_id"])
    assert store.transition_task(task_id, from_statuses=("created",), to_status="queued")
    revoked: list[tuple[str, bool]] = []
    monkeypatch.setattr(
        "yieldmind.task_queue.celery_app.control.revoke",
        lambda revoked_id, terminate=False: revoked.append((revoked_id, terminate)),
    )

    first = cancel_queued_task(task_id, store=store, reason="user changed scope", requested_by="pytest")
    second = cancel_queued_task(task_id, store=store)

    assert first is not None and first["task"]["status"] == "cancelled"
    assert first["revoke_requested"] is True
    assert first["task"]["cancel_reason"] == "user changed scope"
    assert first["task"]["cancel_requested_by"] == "pytest"
    assert first["task"]["cancel_requested_at"] is not None
    assert first["task"]["cancelled_at"] is not None
    assert second is not None and second["idempotent"] is True
    assert revoked == [(task_id, False)]
    with pytest.raises(ValueError, match="Illegal task status transition"):
        store.transition_task(task_id, from_statuses=("cancelled",), to_status="running")


def test_cancel_running_task_requests_cooperative_stop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = YieldMindStore(tmp_path / "yieldmind.sqlite3")
    task, _ = store.create_task(task_type="offline_workflow", idempotency_key="cancel-running-key", payload={})
    task_id = str(task["task_id"])
    assert store.transition_task(task_id, from_statuses=("created",), to_status="queued")
    assert store.transition_task(task_id, from_statuses=("queued",), to_status="running")
    monkeypatch.setattr("yieldmind.task_queue.celery_app.control.revoke", lambda task_id, terminate=False: None)

    result = cancel_queued_task(task_id, store=store, reason="stop after node", requested_by="pytest")
    repeated = cancel_queued_task(task_id, store=store)

    assert result is not None and result["task"]["status"] == "cancel_requested"
    assert result["cancellation_pending"] is True
    assert result["task"]["cancelled_at"] is None
    assert repeated is not None and repeated["idempotent"] is True
    assert store.complete_task_cancellation(task_id, result={"status": "cancelled"}, run_id="run_cancelled")
    completed = store.get_task(task_id)
    assert completed is not None and completed["status"] == "cancelled"
    assert completed["run_id"] == "run_cancelled"
    assert completed["cancelled_at"] is not None


def test_cancelled_database_state_survives_revoke_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = YieldMindStore(tmp_path / "yieldmind.sqlite3")
    task, _ = store.create_task(task_type="offline_workflow", idempotency_key="cancel-revoke-failure", payload={})
    task_id = str(task["task_id"])
    assert store.transition_task(task_id, from_statuses=("created",), to_status="queued")

    def fail_revoke(task_id: str, terminate: bool = False) -> None:
        raise ConnectionError("redis control unavailable")

    monkeypatch.setattr("yieldmind.task_queue.celery_app.control.revoke", fail_revoke)
    result = cancel_queued_task(task_id, store=store)

    assert result is not None
    assert result["task"]["status"] == "cancelled"
    assert result["revoke_requested"] is False
    assert "redis control unavailable" in result["revoke_error"]
    with pytest.raises(ValueError, match="Illegal task status transition"):
        store.transition_task(task_id, from_statuses=("cancelled",), to_status="running")


def test_running_cancellation_request_survives_revoke_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = YieldMindStore(tmp_path / "yieldmind.sqlite3")
    task, _ = store.create_task(task_type="offline_workflow", idempotency_key="running-revoke-failure", payload={})
    task_id = str(task["task_id"])
    assert store.transition_task(task_id, from_statuses=("created",), to_status="queued")
    assert store.transition_task(task_id, from_statuses=("queued",), to_status="running")

    def fail_revoke(task_id: str, terminate: bool = False) -> None:
        raise ConnectionError("redis control unavailable")

    monkeypatch.setattr("yieldmind.task_queue.celery_app.control.revoke", fail_revoke)
    result = cancel_queued_task(task_id, store=store, requested_by="pytest")

    assert result is not None
    assert result["task"]["status"] == "cancel_requested"
    assert result["cancellation_pending"] is True
    assert result["revoke_requested"] is False
    assert "redis control unavailable" in result["revoke_error"]
    assert store.get_task(task_id)["cancel_requested_by"] == "pytest"


def test_cancel_task_api_handles_queued_running_and_terminal_states(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = YieldMindStore(tmp_path / "yieldmind.sqlite3")
    monkeypatch.setattr(yieldmind_api, "store", store)
    monkeypatch.setattr("yieldmind.task_queue.celery_app.control.revoke", lambda task_id, terminate=False: None)
    client = TestClient(yieldmind_api.app)

    queued, _ = store.create_task(task_type="offline_workflow", idempotency_key="api-cancel-queued", payload={})
    assert store.transition_task(queued["task_id"], from_statuses=("created",), to_status="queued")
    cancelled_response = client.post(
        f"/api/tasks/{queued['task_id']}/cancel",
        json={"reason": "cancel from API test", "requested_by": "pytest-api"},
    )
    assert cancelled_response.status_code == 200
    assert cancelled_response.json()["task"]["status"] == "cancelled"
    assert cancelled_response.json()["task"]["cancel_requested_by"] == "pytest-api"

    running, _ = store.create_task(task_type="offline_workflow", idempotency_key="api-cancel-running", payload={})
    assert store.transition_task(running["task_id"], from_statuses=("created",), to_status="queued")
    assert store.transition_task(running["task_id"], from_statuses=("queued",), to_status="running")
    running_response = client.post(
        f"/api/tasks/{running['task_id']}/cancel",
        json={"reason": "stop after current node", "requested_by": "pytest-api"},
    )
    assert running_response.status_code == 200
    assert running_response.json()["task"]["status"] == "cancel_requested"
    assert running_response.json()["cancellation_pending"] is True

    terminal, _ = store.create_task(task_type="offline_workflow", idempotency_key="api-cancel-terminal", payload={})
    assert store.transition_task(terminal["task_id"], from_statuses=("created",), to_status="queued")
    assert store.transition_task(terminal["task_id"], from_statuses=("queued",), to_status="running")
    assert store.transition_task(terminal["task_id"], from_statuses=("running",), to_status="completed")
    conflict_response = client.post(
        f"/api/tasks/{terminal['task_id']}/cancel",
        json={"reason": "too late", "requested_by": "pytest-api"},
    )
    assert conflict_response.status_code == 409
    assert "terminal status completed" in conflict_response.json()["detail"]


def test_task_status_keeps_postgres_authority_when_celery_backend_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = YieldMindStore(tmp_path / "yieldmind.sqlite3")
    task, _ = store.create_task(task_type="offline_workflow", idempotency_key="backend-failure", payload={})

    class BrokenAsyncResult:
        def __init__(self, task_id: str, app: object) -> None:
            self.task_id = task_id

        @property
        def state(self) -> str:
            raise ValueError("corrupt celery metadata")

    monkeypatch.setattr("yieldmind.task_queue.AsyncResult", BrokenAsyncResult)
    result = get_task_status(task["task_id"], store=store)

    assert result is not None
    assert result["task"]["status"] == "created"
    assert result["celery_state"] == "UNKNOWN"
    assert "corrupt celery metadata" in result["celery_error"]


def test_static_console_exposes_real_task_cancel_controls() -> None:
    html = (Path(__file__).resolve().parents[1] / "yieldmind" / "static" / "index.html").read_text(encoding="utf-8")
    assert "/api/tasks/workflows/offline" in html
    assert "/cancel`" in html
    assert "/api/task-recovery/stale" in html
    assert "/recover`" in html
    assert "confirmed_worker_stopped:true" in html
    assert "cancel_requested_by" in html
    assert "依赖检查" in html
    assert 'celery_state:"REVOKED"' not in html
    assert "REVOKE_REQUESTED" in html


def test_profile_and_baseline_tools(tmp_path: Path) -> None:
    csv_path = tmp_path / "yield.csv"
    _write_small_yield_csv(csv_path)
    store = YieldMindStore(tmp_path / "yieldmind.sqlite3")
    registry = ToolRegistry(store=store)
    profile = registry.execute("profile_yield_data", {"data_path": str(csv_path)})
    assert profile.ok, profile.error
    assert profile.result["row_count"] == 6
    baseline = registry.execute("run_fixed_baseline_eval", {"data_path": str(csv_path), "n_splits": 3})
    assert baseline.ok, baseline.error
    assert Path(baseline.artifacts["baseline_report"]).exists()
    assert baseline.result["best_baseline"]["name"] == baseline.result["report"]["best_baseline"]
    candidate = registry.execute("run_candidate_benchmark", {"data_path": str(csv_path), "n_splits": 3, "max_rows": 20})
    assert candidate.ok, candidate.error
    assert Path(candidate.artifacts["candidate_benchmark_report"]).exists()


def test_tool_call_idempotency_reuses_completed_result(tmp_path: Path) -> None:
    csv_path = tmp_path / "yield.csv"
    _write_small_yield_csv(csv_path)
    store = YieldMindStore(tmp_path / "yieldmind.sqlite3")
    registry = ToolRegistry(store=store)
    args = {"data_path": str(csv_path)}

    first = registry.execute("profile_yield_data", args, idempotency_key="profile-once")
    second = registry.execute("profile_yield_data", args, idempotency_key="profile-once")

    assert first == second
    with connect(store.db_path) as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS count FROM yieldmind_tool_calls WHERE idempotency_key=?",
            ("profile-once",),
        ).fetchone()
    assert row is not None and int(row["count"]) == 1


def test_tool_call_idempotency_refuses_ambiguous_running_replay(tmp_path: Path) -> None:
    csv_path = tmp_path / "yield.csv"
    _write_small_yield_csv(csv_path)
    store = YieldMindStore(tmp_path / "yieldmind.sqlite3")
    store.claim_tool_call(
        tool_name="profile_yield_data",
        mode="offline_deterministic",
        args={"data_path": str(csv_path)},
        idempotency_key="profile-running",
    )

    result = ToolRegistry(store=store).execute(
        "profile_yield_data",
        {"data_path": str(csv_path)},
        idempotency_key="profile-running",
    )

    assert not result.ok
    assert "refusing automatic replay" in result.error


def test_sandboxed_python_tool() -> None:
    registry = ToolRegistry()
    result = registry.execute(
        "run_sandboxed_python",
        {"argv": [sys.executable, "-c", "print('sandbox-ok')"], "timeout_seconds": 10},
    )
    assert result.ok, result.error
    assert "sandbox-ok" in result.result["stdout"]


def test_docker_sandbox_requires_explicit_authorization() -> None:
    registry = ToolRegistry()
    result = registry.execute(
        "run_docker_sandboxed_python",
        {"argv": ["python", "-c", "print('not-started')"]},
    )
    assert not result.ok
    assert result.mode == "docker_no_network"
    assert result.result["authorized"] is False
    assert result.result["docker_cli_available"] is False
    assert "allow_docker=true" in result.error


def test_docker_command_has_required_isolation_controls(tmp_path: Path) -> None:
    workspace_root = tmp_path.resolve()
    command = DockerSandboxCommand(
        argv=["python", "-c", "print('isolated')"],
        output_dir="outputs/docker-test",
        allow_docker=True,
    )
    argv, output_dir = build_docker_argv(
        command,
        workspace_root=workspace_root,
        docker_executable="/usr/local/bin/docker",
        container_name="yieldmind-test-container",
    )
    joined = " ".join(argv)
    assert output_dir == workspace_root / "outputs" / "docker-test"
    assert "--network none" in joined
    assert "--read-only" in argv
    assert "--cap-drop ALL" in joined
    assert "no-new-privileges:true" in argv
    assert "--pull=never" in argv
    assert f"type=bind,src={workspace_root},dst=/workspace,readonly" in argv
    assert f"type=bind,src={output_dir},dst=/output" in argv


def test_docker_sandbox_rejects_escape_and_unapproved_image(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="allowlisted"):
        DockerSandboxCommand(image="untrusted:latest")
    command = DockerSandboxCommand(output_dir="../outside", allow_docker=True)
    with pytest.raises(ValueError, match="inside repository root"):
        build_docker_argv(
            command,
            workspace_root=tmp_path,
            docker_executable="docker",
            container_name="yieldmind-test-container",
        )


def test_docker_result_can_include_built_command(tmp_path: Path) -> None:
    request = DockerSandboxCommand(allow_docker=True)
    result = DockerSandbox(tmp_path)._base_result(request, command=["docker", "run"])
    assert result.command == ["docker", "run"]


def test_local_rule_planner_routes_tools() -> None:
    plan = local_rule_plan(ToolPlanRequest(prompt="请对 demo 数据做 baseline 评测", max_tool_calls=6))
    tools = {call.tool_name for call in plan.tool_calls}
    assert plan.llm_calls == 0
    assert plan.simulated_model_calls == 0
    assert {"generate_demo_yield_data", "run_fixed_baseline_eval"}.issubset(tools)
    safety_plan = local_rule_plan(ToolPlanRequest(prompt="脱敏 Authorization 并控制上下文 token budget", max_tool_calls=6))
    safety_tools = {call.tool_name for call in safety_plan.tool_calls}
    assert {"redact_sensitive_payload", "plan_token_budget"}.issubset(safety_tools)
    evidence_plan = local_rule_plan(ToolPlanRequest(prompt="校验证据引用 chunk_id 和 text_hash", max_tool_calls=6))
    evidence_tools = {call.tool_name for call in evidence_plan.tool_calls}
    assert {"search_knowledge", "validate_evidence_refs"}.issubset(evidence_tools)


def test_function_calling_smoke_script_skips_live_llm_by_default(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_yieldmind_function_calling_smoke.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--out-dir",
            str(tmp_path),
            "--prompt",
            "脱敏 Authorization 并控制上下文 token budget",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["status"] == "skipped"
    assert payload["real_llm_calls"] == 0
    assert payload["simulated_model_calls"] == 0
    report_path = Path(payload["report_path"])
    assert report_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    preview_tools = {item["tool_name"] for item in report["offline_preview_plan"]["tool_calls"]}
    assert {"redact_sensitive_payload", "plan_token_budget"}.issubset(preview_tools)


class _FakeCompletions:
    def __init__(self, messages: list[SimpleNamespace]) -> None:
        self._messages = list(messages)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=self._messages.pop(0))])


def _fake_client(messages: list[SimpleNamespace]) -> SimpleNamespace:
    return SimpleNamespace(chat=SimpleNamespace(completions=_FakeCompletions(messages)))


def _tool_message(name: str, arguments: str, *, call_id: str = "call_1") -> SimpleNamespace:
    call = SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=arguments))
    return SimpleNamespace(content=None, tool_calls=[call])


def test_function_calling_rejects_malformed_or_invalid_model_arguments(tmp_path: Path) -> None:
    registry = ToolRegistry(store=YieldMindStore(tmp_path / "yieldmind.sqlite3"))
    request = ToolPlanRequest(prompt="search", allow_live_llm=True)
    malformed = _fake_client([_tool_message("search_knowledge", "{bad json")])
    with pytest.raises(ToolCallProtocolError, match="malformed JSON"):
        plan_tools(request, registry=registry, client=malformed, simulated_test_adapter=True)

    missing_required = _fake_client([_tool_message("search_knowledge", "{}")])
    with pytest.raises(ToolCallProtocolError, match="Pydantic validation"):
        plan_tools(request, registry=registry, client=missing_required, simulated_test_adapter=True)

    unknown = _fake_client([_tool_message("invented_tool", "{}")])
    with pytest.raises(ToolCallProtocolError, match="Unknown tool"):
        plan_tools(request, registry=registry, client=unknown, simulated_test_adapter=True)


def test_function_calling_roundtrip_returns_tool_result_to_model(tmp_path: Path) -> None:
    first = _tool_message("redact_sensitive_payload", '{"payload":{"Authorization":"Bearer sk-test-secret"}}')
    second = SimpleNamespace(content="The payload was redacted by the validated tool.", tool_calls=[])
    client = _fake_client([first, second])
    store = YieldMindStore(tmp_path / "yieldmind.sqlite3")
    registry = ToolRegistry(store=store)
    result = execute_plan(
        ToolPlanRequest(prompt="Redact this payload", allow_live_llm=True),
        registry=registry,
        store=store,
        client=client,
        simulated_test_adapter=True,
    )
    assert result.passed is True
    assert result.mode == SIMULATED_TEST_ADAPTER
    assert result.llm_calls == 0
    assert result.simulated_model_calls == 2
    assert result.final_answer == "The payload was redacted by the validated tool."
    assert len(client.chat.completions.calls) == 2
    final_messages = client.chat.completions.calls[1]["messages"]
    tool_messages = [message for message in final_messages if message["role"] == "tool"]
    assert tool_messages[0]["tool_call_id"] == "call_1"
    assert "sk-test-secret" not in tool_messages[0]["content"]
    assert client.chat.completions.calls[1]["tool_choice"] == "none"


def test_workflow_runs_with_local_fallback(tmp_path: Path) -> None:
    store = YieldMindStore(tmp_path / "yieldmind.sqlite3")
    result = run_workflow(WorkflowRequest(prompt="Run deterministic demo baseline workflow.", n_samples=40, n_splits=2), store=store)
    assert result["status"] == "passed", result.get("errors")
    assert result["workflow_backend"] in {"local_workflow_fallback", "langgraph_stategraph"}
    assert result["tool_results"]
    assert "candidate_benchmark_report" in result["artifacts"]
    assert "report_json" in result["artifacts"]
    assert result["checkpoint_backend"] in {"none", "langgraph_memory_saver"}
    assert all(stage.get("stage_execution_id") and stage.get("input_hash") for stage in result["stages"])
    persisted_stages = store.list_stage_executions(result["run_id"])
    assert [item["stage"] for item in persisted_stages] == [item["stage"] for item in result["stages"]]


def test_workflow_stops_cooperatively_at_node_boundary_and_links_run(tmp_path: Path) -> None:
    store = YieldMindStore(tmp_path / "yieldmind.sqlite3")
    registry = _FlakyWorkflowRegistry()
    checks = 0
    linked_runs: list[str] = []

    def cancellation_requested() -> bool:
        nonlocal checks
        checks += 1
        return checks >= 2

    result = YieldMindWorkflow(
        store=store,
        registry=registry,
        cancel_check=cancellation_requested,
        on_run_started=linked_runs.append,
    ).run(WorkflowRequest(n_samples=20, n_splits=2))

    assert result["status"] == "cancelled"
    assert linked_runs == [result["run_id"]]
    assert registry.calls == ["generate_demo_yield_data"]
    assert [stage["stage"] for stage in result["stages"]] == ["start", "prepare_data", "cancel", "finish"]
    assert result["route_history"][-1]["route"] == "cancel"
    assert result["route_history"][-1]["after_stage"] == "prepare_data"
    assert store.get_run(result["run_id"])["status"] == "cancelled"


class _FlakyWorkflowRegistry:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.candidate_calls = 0

    def execute(
        self,
        name: str,
        args: dict,
        *,
        run_id: str | None = None,
        idempotency_key: str = "",
    ) -> ToolResult:
        self.calls.append(name)
        if name == "generate_demo_yield_data":
            return ToolResult(ok=True, result={"paths": {"synthetic_path": "generated.csv"}})
        if name == "profile_yield_data":
            return ToolResult(ok=True, result={"row_count": 20})
        if name == "run_fixed_baseline_eval":
            return ToolResult(ok=True, result={"best_baseline": {"name": "ridge"}}, artifacts={"baseline_report": "baseline.json"})
        if name == "run_candidate_benchmark":
            self.candidate_calls += 1
            if self.candidate_calls == 1:
                return ToolResult(ok=False, error="transient candidate failure")
            return ToolResult(
                ok=True,
                result={"benchmark_report": {"selected_strategy_id": "ridge"}},
                artifacts={"candidate_benchmark_report": "candidate.json"},
            )
        if name == "verify_artifacts":
            return ToolResult(ok=True, result={"reasons": [], "warnings": []})
        if name == "build_run_report":
            return ToolResult(ok=True, artifacts={"report_json": "report.json", "report_md": "report.md"})
        raise AssertionError(f"Unexpected tool: {name}")


def test_workflow_uses_conditional_local_repair(tmp_path: Path) -> None:
    store = YieldMindStore(tmp_path / "yieldmind.sqlite3")
    registry = _FlakyWorkflowRegistry()
    result = YieldMindWorkflow(store=store, registry=registry).run(
        WorkflowRequest(n_samples=20, n_splits=3, max_local_repairs=1, max_replans=0)
    )
    assert result["status"] == "passed"
    assert registry.candidate_calls == 2
    assert result["local_repair_count"] == 1
    assert result["route_history"][0]["route"] == "local_repair"
    assert result["failure_history"][0]["stage"] == "candidate_benchmark"
    persisted = store.list_stage_executions(result["run_id"])
    assert [item["stage"] for item in persisted].count("candidate_benchmark") == 2
    assert any(item["stage"] == "local_repair" for item in persisted)


def test_workflow_rejects_missing_user_data_without_demo_fallback(tmp_path: Path) -> None:
    store = YieldMindStore(tmp_path / "yieldmind.sqlite3")
    registry = _FlakyWorkflowRegistry()
    missing = tmp_path / "missing.csv"
    result = YieldMindWorkflow(store=store, registry=registry).run(
        WorkflowRequest(data_path=str(missing), max_local_repairs=0, max_replans=0)
    )
    assert result["status"] == "failed"
    assert not registry.calls
    assert result["failed_stage"] == "prepare_data"
    assert "does not exist" in result["errors"][0]


def test_knowledge_ingest_and_search(tmp_path: Path) -> None:
    doc = tmp_path / "domain.md"
    doc.write_text(
        "# Yield notes\n\nYODEL links yield stress to packing density phi_m and contact networks.\n\n"
        "Mechanism ablation compares with_mechanism and without_mechanism under the same OOF protocol.\n",
        encoding="utf-8",
    )
    store = YieldMindStore(tmp_path / "yieldmind.sqlite3")
    kb = KnowledgeBase(store=store, chroma_dir=tmp_path / "chroma", collection_name="test_collection")
    ingest = kb.ingest(KnowledgeIngestRequest(paths=[str(doc)], chunk_size=300, chunk_overlap=20))
    assert ingest["documents"][0]["status"] == "available"
    again = kb.ingest(KnowledgeIngestRequest(paths=[str(doc)], chunk_size=300, chunk_overlap=20))
    assert again["documents"][0]["idempotent"] is True
    result = kb.search(KnowledgeSearchRequest(query="YODEL packing phi_m yield stress", top_k=2))
    assert result["hits"]
    assert result["hits"][0]["chunk_id"].startswith("chk_")


def test_knowledge_bm25_hybrid_and_profile_isolation(tmp_path: Path) -> None:
    mechanism = tmp_path / "mechanism.md"
    mechanism.write_text("YODEL maximum packing phi_m and contact network mechanism.", encoding="utf-8")
    safety = tmp_path / "safety.md"
    safety.write_text("Docker sandbox networking disabled with memory and process limits.", encoding="utf-8")
    store = YieldMindStore(tmp_path / "yieldmind.sqlite3")
    kb = KnowledgeBase(store=store, chroma_dir=tmp_path / "chroma", collection_name="profile_test")
    kb.ingest(KnowledgeIngestRequest(paths=[str(mechanism), str(safety)], chunk_size=300, chunk_overlap=20))

    bm25 = kb.search(
        KnowledgeSearchRequest(query="YODEL packing phi_m", retrieval_mode="bm25", top_k=2)
    )
    hybrid = kb.search(
        KnowledgeSearchRequest(query="Docker network memory sandbox", retrieval_mode="hybrid", top_k=2)
    )
    assert bm25["hits"][0]["source_path"].endswith("mechanism.md")
    assert hybrid["hits"][0]["source_path"].endswith("safety.md")
    assert set(hybrid["hits"][0]["retrieval_channels"]) == {"vector", "bm25"}

    rebuilt_kb = KnowledgeBase(store=store, chroma_dir=tmp_path / "chroma", collection_name="profile_test_rebuilt")
    rebuilt = rebuilt_kb.ingest(KnowledgeIngestRequest(paths=[str(mechanism)], chunk_size=300, chunk_overlap=20))
    assert rebuilt["documents"][0]["idempotent"] is False
    assert rebuilt_kb.search(KnowledgeSearchRequest(query="YODEL phi_m", top_k=1))["hits"]

    alternate = EmbeddingProfile(dimensions=512, index_version="yieldmind-hashing-512-v1")
    alternate_kb = KnowledgeBase(
        store=store,
        chroma_dir=tmp_path / "chroma",
        collection_name="profile_test",
        embedding_profile=alternate,
    )
    assert alternate_kb.collection_name != kb.collection_name
    with pytest.raises(ValueError, match="index_version"):
        alternate_kb.search(KnowledgeSearchRequest(query="packing"))
    with pytest.raises(ValueError, match="immutable model revision"):
        EmbeddingProfile(provider="sentence_transformers", model_id="Qwen/Qwen3-Embedding-0.6B", revision="main")


def test_retrieval_eval_dataset_has_distinct_labeled_cases() -> None:
    payload = json.loads(Path("evals/yieldmind_retrieval_cases_v1.json").read_text(encoding="utf-8"))
    cases = payload["cases"]
    assert len(cases) >= 30
    assert len({case["id"] for case in cases}) == len(cases)
    assert len({source for case in cases for source in case["expected_source_paths"]}) >= 6
    assert any(any("\u4e00" <= char <= "\u9fff" for char in case["query"]) for case in cases)


def test_safety_redaction_and_budget_tools(tmp_path: Path) -> None:
    store = YieldMindStore(tmp_path / "yieldmind.sqlite3")
    registry = ToolRegistry(store=store)
    redacted = registry.execute(
        "redact_sensitive_payload",
        {
            "payload": {
                "Authorization": "Bearer sk-test1234567890",
                "owner": "alice@example.com",
                "nested": {"database_url": "postgresql://user:pass@localhost:5432/app"},
            }
        },
    )
    assert redacted.ok, redacted.error
    rendered = json.dumps(redacted.result, ensure_ascii=False)
    assert "sk-test1234567890" not in rendered
    assert "alice@example.com" not in rendered
    assert "postgresql://user:pass" not in rendered
    assert redacted.result["redaction_count"] >= 3
    with connect(store.db_path) as conn:
        logged = conn.execute(
            "SELECT args_json, result_json FROM yieldmind_tool_calls WHERE tool_name=? ORDER BY started_at DESC LIMIT 1",
            ("redact_sensitive_payload",),
        ).fetchone()
    assert logged is not None
    logged_text = f"{logged['args_json']}\n{logged['result_json']}"
    assert "sk-test1234567890" not in logged_text
    assert "alice@example.com" not in logged_text
    assert "postgresql://user:pass" not in logged_text

    budget = registry.execute(
        "plan_token_budget",
        {
            "max_input_tokens": 120,
            "reserved_output_tokens": 20,
            "sections": [
                {"name": "constraints", "text": "target=tau0", "tokens": 40, "required": True, "priority": 100},
                {"name": "evidence", "text": "chunk evidence", "tokens": 50, "priority": 80},
                {"name": "history", "text": "older turns", "tokens": 80, "priority": 1},
            ],
        },
    )
    assert budget.ok, budget.error
    selected = {item["name"] for item in budget.result["selected_sections"]}
    dropped = {item["name"] for item in budget.result["dropped_sections"]}
    assert {"constraints", "evidence"}.issubset(selected)
    assert "history" in dropped
    assert budget.result["total_selected_tokens"] <= budget.result["available_input_tokens"]


def test_evidence_validation_checks_chunk_metadata(tmp_path: Path) -> None:
    doc = tmp_path / "domain.md"
    doc.write_text(
        "# Yield notes\n\nYODEL evidence chunk for packing density and yield stress.\n",
        encoding="utf-8",
    )
    store = YieldMindStore(tmp_path / "yieldmind.sqlite3")
    kb = KnowledgeBase(store=store, chroma_dir=tmp_path / "chroma", collection_name="evidence_collection")
    kb.ingest(KnowledgeIngestRequest(paths=[str(doc)], chunk_size=300, chunk_overlap=20))
    hit = kb.search(KnowledgeSearchRequest(query="packing yield stress", top_k=1))["hits"][0]
    registry = ToolRegistry(store=store)

    valid = registry.execute(
        "validate_evidence_refs",
        {"refs": [{"chunk_id": hit["chunk_id"], "text_hash": hit["text_hash"], "index_version": hit["index_version"]}]},
    )
    assert valid.ok, valid.error
    assert valid.result["valid_count"] == 1

    invalid = registry.execute(
        "validate_evidence_refs",
        {"refs": [{"chunk_id": hit["chunk_id"], "text_hash": "wrong", "index_version": hit["index_version"]}]},
    )
    assert not invalid.ok
    assert invalid.result["invalid_count"] == 1


def test_session_memory_constraints_and_delete(tmp_path: Path) -> None:
    store = YieldMindStore(tmp_path / "yieldmind.sqlite3")
    memory = SessionMemoryStore(store)
    session = memory.create_session(CreateSessionRequest(constraints={"target_column": "yield_stress"}))
    session_id = session["session_id"]
    first = memory.add_message(
        AddMessageRequest(session_id=session_id, content="目标列改成 tau0，请不要重新训练", idempotency_key="turn-1")
    )
    assert first["action"] == "answer_only"
    assert memory.get_session(session_id)["constraints"]["target_column"] == "tau0"
    again = memory.add_message(
        AddMessageRequest(session_id=session_id, content="目标列改成 tau0，请不要重新训练", idempotency_key="turn-1")
    )
    assert again["idempotent"] is True
    created = memory.add_message(AddMessageRequest(session_id=session_id, content="把搜索预算调小，再运行一次", idempotency_key="turn-2"))
    assert created["action"] == "create_run"
    assert created["run_id"].startswith("run_")
    mem = memory.upsert_memory(
        UpsertMemoryRequest(session_id=session_id, content="用户偏好小搜索预算", source_ref="turn-2")
    )
    assert memory.search_memories(session_id=session_id)
    deleted = memory.delete_memory(type("Req", (), {"memory_id": mem["memory_id"]})())
    assert deleted["ok"] is True
    assert not memory.search_memories(session_id=session_id)
