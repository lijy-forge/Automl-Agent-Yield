from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

import yieldmind.api as yieldmind_api
from yieldmind.database import YieldMindStore
from yieldmind.domain_workflow import DomainWorkflowRequest, RealYieldDomainAdapter, YieldDomainWorkflow
from yieldmind.memory import AddMessageRequest, CreateSessionRequest, SessionMemoryStore
from yieldmind.process_control import ManagedProcessOutcome
from yieldmind.repair_memory import RepairMemoryStore
from yieldmind.task_queue import EnqueueDomainWorkflowRequest, enqueue_domain_workflow


class FakeDomainAdapter:
    uses_live_llm = False

    def __init__(self, *, review_passes: list[bool] | None = None, cancel_operation: bool = False) -> None:
        self.calls: list[str] = []
        self.review_passes = list(review_passes or [True])
        self.cancel_operation = cancel_operation
        self.operation_calls = 0
        self.candidate_calls = 0

    def _ok(self, name: str, **extra: Any) -> dict[str, Any]:
        self.calls.append(name)
        return {"ok": True, "message": f"fake {name}", **extra}

    def prepare(self, state: dict[str, Any]) -> dict[str, Any]:
        Path(state["run_dir"]).mkdir(parents=True, exist_ok=True)
        return self._ok("prepare", artifacts={"manager_run_config": str(Path(state["run_dir"]) / "config.json")})

    def data(self, state: dict[str, Any]) -> dict[str, Any]:
        return self._ok("data", runtime_env={"YIELD_DATA_PATH": "fake.csv"})

    def requirements(self, state: dict[str, Any]) -> dict[str, Any]:
        return self._ok("requirements")

    def search(self, state: dict[str, Any]) -> dict[str, Any]:
        return self._ok("search")

    def candidate(self, state: dict[str, Any]) -> dict[str, Any]:
        self.candidate_calls += 1
        return self._ok("candidate", simulated_model_calls=self.candidate_calls)

    def model(self, state: dict[str, Any]) -> dict[str, Any]:
        return self._ok("model", simulated_model_calls=self.candidate_calls * 2)

    def pre_execution(self, state: dict[str, Any]) -> dict[str, Any]:
        return self._ok("pre_execution", pre_execution_passed=True)

    def operation(self, state: dict[str, Any]) -> dict[str, Any]:
        self.operation_calls += 1
        if self.cancel_operation:
            return self._ok(
                "operation",
                cancelled=True,
                operation_result={"rcode": 130, "cancelled": True},
                process_control={"status": "cancelled", "terminate_sent": True},
            )
        return self._ok(
            "operation",
            operation_result={"rcode": 0, "manager_revision_round": state["manager_revision_round"]},
            process_control={"status": "completed"},
        )

    def review(self, state: dict[str, Any]) -> dict[str, Any]:
        passed = self.review_passes.pop(0)
        return self._ok(
            "review",
            post_execution_review={
                "passed": passed,
                "decision": "accepted" if passed else "revise_candidate_and_model",
            },
        )

    def revision_feedback(self, state: dict[str, Any]) -> dict[str, Any]:
        return self._ok("revision_feedback", manager_feedback=f"revision-{state['manager_revision_round']}")


class RepairingDomainAdapter(FakeDomainAdapter):
    def __init__(self) -> None:
        super().__init__(review_passes=[False, True])
        self.candidate_repair_contexts: list[str] = []

    def candidate(self, state: dict[str, Any]) -> dict[str, Any]:
        self.candidate_repair_contexts.append(str(state.get("repair_memory_context") or ""))
        return super().candidate(state)

    def operation(self, state: dict[str, Any]) -> dict[str, Any]:
        self.operation_calls += 1
        if self.operation_calls == 1:
            return self._ok(
                "operation",
                ok=False,
                error="Generated code failed deterministic artifact verification.",
                operation_result={
                    "rcode": 1,
                    "action_result": "metrics.json was not written",
                    "error_logs": ["Missing required artifact metrics.json"],
                },
                process_control={"status": "completed"},
            )
        return self._ok(
            "operation",
            operation_result={
                "rcode": 0,
                "action_result": "Training completed and metrics.json was verified.",
                "error_logs": [],
            },
            process_control={"status": "completed"},
        )


def _run(
    tmp_path: Path,
    adapter: FakeDomainAdapter,
    *,
    n_revise: int = 1,
    progress_events: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    store = YieldMindStore(tmp_path / "yieldmind.sqlite3")
    return YieldDomainWorkflow(
        store=store,
        adapter=adapter,
        on_stage_progress=progress_events.append if progress_events is not None else None,
    ).run(
        DomainWorkflowRequest(
            run_dir=str(tmp_path / "run"),
            synthetic_data=True,
            external_search=False,
            require_search_results=False,
            n_revise=n_revise,
        )
    )


def test_domain_stategraph_runs_original_phase_order_with_fake_adapter(tmp_path: Path) -> None:
    adapter = FakeDomainAdapter()
    progress_events: list[dict[str, Any]] = []
    result = _run(tmp_path, adapter, progress_events=progress_events)

    assert result["status"] == "passed"
    assert result["workflow_backend"] == "langgraph_stategraph"
    assert adapter.calls == [
        "prepare",
        "data",
        "requirements",
        "search",
        "candidate",
        "model",
        "pre_execution",
        "operation",
        "review",
    ]
    assert [item["stage"] for item in result["stages"]] == [
        "start",
        *adapter.calls,
        "finish",
    ]
    assert result["real_llm_calls"] == 0
    assert result["simulated_model_calls"] == 2
    assert result["model_call_mode"] == "simulated_test_adapter"
    assert [item["next_node"] for item in result["manager_decisions"]] == [
        "prepare",
        "data",
        "requirements",
        "search",
        "candidate",
        "model",
        "pre_execution",
        "operation",
        "review",
        "finish",
    ]
    assert all(item["validated"] for item in result["manager_decisions"])
    assert result["manager_decisions"][-1]["reason_code"] == "review_accepted"
    assert [item["route"] for item in result["route_history"]] == [
        item["next_node"] for item in result["manager_decisions"]
    ]
    assert any(
        event["type"] == "agent_progress"
        and event["agent"] == "CandidateAgent"
        and event["stage"] == "candidate"
        and event["status"] == "running"
        for event in progress_events
    )
    assert any(
        event["type"] == "agent_progress"
        and event["agent"] == "OperationAgent"
        and event["stage"] == "operation"
        and event["status"] == "passed"
        for event in progress_events
    )
    candidate_completed = next(
        event
        for event in progress_events
        if event["type"] == "agent_progress"
        and event["stage"] == "candidate"
        and event["status"] == "passed"
    )
    assert candidate_completed["input_summary"]["remaining_revision_budget"] == 1
    assert candidate_completed["output_summary"]["stage"] == "candidate"
    assert candidate_completed["output_summary"]["simulated_model_calls"] == 1
    manager_events = [event for event in progress_events if event["type"] == "manager_decision"]
    assert len(manager_events) == len(result["manager_decisions"])
    assert manager_events[-1]["decision"]["next_node"] == "finish"
    json.dumps(result)


def test_domain_stategraph_revises_candidate_and_model_with_bounded_loop(tmp_path: Path) -> None:
    adapter = FakeDomainAdapter(review_passes=[False, True])
    result = _run(tmp_path, adapter, n_revise=1)

    assert result["status"] == "passed"
    assert result["manager_revision_round"] == 1
    assert adapter.candidate_calls == 2
    assert adapter.operation_calls == 2
    assert adapter.calls.count("revision_feedback") == 1
    assert adapter.calls.index("revision_feedback") < adapter.calls.index("candidate", 5)
    assert result["manager_feedback"] == "revision-1"
    review_routes = [item for item in result["manager_decisions"] if item["after_stage"] == "review"]
    assert [item["reason_code"] for item in review_routes] == ["review_requires_revision", "review_accepted"]
    revision_route = next(item for item in result["manager_decisions"] if item["after_stage"] == "revision_feedback")
    assert revision_route["next_agent"] == "CandidateAgent"
    assert revision_route["feedback"] == "revision-1"
    assert revision_route["remaining_revision_budget"] == 0


def test_verified_repair_memory_is_reused_by_next_domain_run(tmp_path: Path) -> None:
    store = YieldMindStore(tmp_path / "yieldmind.sqlite3")
    first_adapter = RepairingDomainAdapter()
    first = YieldDomainWorkflow(store=store, adapter=first_adapter).run(
        DomainWorkflowRequest(
            run_dir=str(tmp_path / "first"),
            synthetic_data=True,
            external_search=False,
            require_search_results=False,
            n_revise=1,
            workspace_id="repair_workspace",
        )
    )

    assert first["status"] == "passed"
    assert len(first["verified_repair_memory_ids"]) == 1

    second_adapter = RepairingDomainAdapter()
    second_adapter.review_passes = [True]
    second_adapter.operation_calls = 1  # Make this run's first operation succeed.
    second = YieldDomainWorkflow(store=store, adapter=second_adapter).run(
        DomainWorkflowRequest(
            run_dir=str(tmp_path / "second"),
            synthetic_data=True,
            external_search=False,
            require_search_results=False,
            n_revise=0,
            workspace_id="repair_workspace",
        )
    )

    assert second["status"] == "passed"
    assert len(second["repair_memories"]) == 1
    assert "Verified cross-run repair memories" in second["repair_memory_context"]
    assert "metrics.json" in second_adapter.candidate_repair_contexts[0]

    third_adapter = RepairingDomainAdapter()
    third_events: list[dict[str, Any]] = []
    third = YieldDomainWorkflow(
        store=store,
        adapter=third_adapter,
        on_stage_progress=third_events.append,
    ).run(
        DomainWorkflowRequest(
            run_dir=str(tmp_path / "third"),
            synthetic_data=True,
            external_search=False,
            require_search_results=False,
            n_revise=1,
            workspace_id="repair_workspace",
        )
    )

    assert third["status"] == "passed"
    assert third["repair_memory_retrieval"]["phase"] == "operation_error"
    assert third["repair_memory_retrieval"]["selected_count"] == 1
    assert third["repair_memory_retrieval"]["top_score"] > 0
    assert third["reconfirmed_repair_memory_ids"] == first["verified_repair_memory_ids"]
    assert "metrics.json" in third_adapter.candidate_repair_contexts[1]
    retrieval_event = next(
        event
        for event in third_events
        if event.get("action") == "retrieve_verified_repair_memory"
    )
    assert retrieval_event["agent"] == "Agent Manager"
    assert retrieval_event["output_summary"]["phase"] == "operation_error"
    revised_candidate = [
        event
        for event in third_events
        if event.get("stage") == "candidate" and event.get("status") == "running"
    ][1]
    assert revised_candidate["input_summary"]["repair_memory_retrieval"]["selected_count"] == 1
    reconfirm_event = next(
        event
        for event in third_events
        if event.get("action") == "reconfirm_verified_repair_memory"
    )
    assert reconfirm_event["agent"] == "Agent Manager"
    refreshed = RepairMemoryStore(store).memories.get_memory(first["verified_repair_memory_ids"][0])
    assert refreshed is not None
    assert refreshed["metadata"]["verification_count"] == 2
    reuse_store = RepairMemoryStore(store)
    reuse_summary = reuse_store.reuse_outcome_summary(workspace_id="repair_workspace")
    reuse_attempts = reuse_store.list_reuse_attempts(workspace_id="repair_workspace")
    assert reuse_summary["observed_count"] == 1
    assert reuse_summary["success_count"] == 1
    assert reuse_summary["success_rate"] == 1.0
    assert reuse_attempts[0]["run_id"] == third["run_id"]
    assert reuse_attempts[0]["matched_round"] == 0
    assert reuse_attempts[0]["applied_round"] == 1
    assert any(event.get("action") == "apply_verified_repair_memory" for event in third_events)
    assert any(event.get("action") == "record_repair_memory_reuse_outcome" for event in third_events)

    failed_reuse_adapter = RepairingDomainAdapter()
    failed_reuse_adapter.review_passes = [False, False]
    fourth = YieldDomainWorkflow(store=store, adapter=failed_reuse_adapter).run(
        DomainWorkflowRequest(
            run_dir=str(tmp_path / "fourth"),
            synthetic_data=True,
            external_search=False,
            require_search_results=False,
            n_revise=1,
            workspace_id="repair_workspace",
        )
    )
    updated_summary = reuse_store.reuse_outcome_summary(workspace_id="repair_workspace")

    assert fourth["status"] == "failed"
    assert updated_summary["observed_count"] == 2
    assert updated_summary["success_count"] == 1
    assert updated_summary["failure_count"] == 1
    assert updated_summary["success_rate"] == 0.5


def test_domain_workflow_loads_session_context_and_links_run(tmp_path: Path) -> None:
    store = YieldMindStore(tmp_path / "yieldmind.sqlite3")
    memory = SessionMemoryStore(store)
    session = memory.create_session(
        CreateSessionRequest(workspace_id="domain_workspace", constraints={"target_column": "tau0"})
    )
    turn = memory.add_message(
        AddMessageRequest(
            session_id=session["session_id"],
            content="请继续使用 tau0，不要改变目标列。",
            idempotency_key="domain-context-turn",
            create_run_if_requested=False,
        )
    )["turn"]
    progress_events: list[dict[str, Any]] = []

    result = YieldDomainWorkflow(
        store=store,
        adapter=FakeDomainAdapter(),
        on_stage_progress=progress_events.append,
    ).run(
        DomainWorkflowRequest(
            run_dir=str(tmp_path / "session-domain-run"),
            synthetic_data=True,
            external_search=False,
            require_search_results=False,
            workspace_id="domain_workspace",
            session_id=session["session_id"],
            turn_id=turn["turn_id"],
        )
    )

    assert result["status"] == "passed"
    assert result["session_context_summary"]["session_id"] == session["session_id"]
    assert "current_constraints" in result["session_context_summary"]["selected_sections"]
    assert "tau0" in result["session_context_prompt"]
    assert memory.get_session(session["session_id"])["current_run_id"] == result["run_id"]
    candidate_started = next(
        event for event in progress_events
        if event.get("type") == "agent_progress"
        and event.get("stage") == "candidate"
        and event.get("status") == "running"
    )
    assert candidate_started["input_summary"]["session_context"]["estimated_tokens"] > 0


def test_domain_workflow_rejects_cross_workspace_session_context(tmp_path: Path) -> None:
    store = YieldMindStore(tmp_path / "yieldmind.sqlite3")
    memory = SessionMemoryStore(store)
    session = memory.create_session(CreateSessionRequest(workspace_id="workspace_a"))

    result = YieldDomainWorkflow(store=store, adapter=FakeDomainAdapter()).run(
        DomainWorkflowRequest(
            run_dir=str(tmp_path / "cross-workspace"),
            synthetic_data=True,
            external_search=False,
            require_search_results=False,
            workspace_id="workspace_b",
            session_id=session["session_id"],
        )
    )

    assert result["status"] == "failed"
    assert [item["stage"] for item in result["stages"]] == ["start", "finish"]
    assert "workspace_id does not match" in result["errors"][0]
    assert memory.get_session(session["session_id"])["current_run_id"] == ""



def test_domain_stategraph_routes_cancelled_operation_directly_to_cancel(tmp_path: Path) -> None:
    adapter = FakeDomainAdapter(cancel_operation=True)
    result = _run(tmp_path, adapter)

    assert result["status"] == "cancelled"
    assert "review" not in adapter.calls
    assert [item["stage"] for item in result["stages"]][-2:] == ["cancel", "finish"]
    assert result["process_control"]["terminate_sent"] is True
    assert result["manager_decisions"][-2]["reason_code"] == "cancel_requested"
    assert result["manager_decisions"][-2]["next_node"] == "cancel"
    assert result["manager_decisions"][-1]["reason_code"] == "cancellation_finalized"


def test_failed_operation_is_marked_failed_but_still_reaches_manager_review(tmp_path: Path) -> None:
    class FailedOperationAdapter(FakeDomainAdapter):
        def operation(self, state: dict[str, Any]) -> dict[str, Any]:
            self.calls.append("operation")
            return {
                "ok": False,
                "error": "OperationAgent timed out.",
                "operation_result": {"rcode": 1, "timed_out": True},
            }

    adapter = FailedOperationAdapter(review_passes=[False])
    result = _run(tmp_path, adapter, n_revise=0)

    operation_stage = next(stage for stage in result["stages"] if stage["stage"] == "operation")
    operation_route = next(
        decision for decision in result["manager_decisions"] if decision["after_stage"] == "operation"
    )
    assert operation_stage["status"] == "failed"
    assert operation_route["next_node"] == "review"
    assert "review" in adapter.calls
    assert result["status"] == "failed"


def test_real_domain_workflow_is_blocked_without_explicit_live_llm_permission(tmp_path: Path) -> None:
    store = YieldMindStore(tmp_path / "yieldmind.sqlite3")
    result = YieldDomainWorkflow(store=store).run(
        DomainWorkflowRequest(run_dir=str(tmp_path / "blocked"), allow_live_llm=False)
    )

    assert result["status"] == "failed"
    assert [item["stage"] for item in result["stages"]] == ["start", "finish"]
    assert result["real_llm_calls"] == 0
    assert result["model_call_mode"] == "live_blocked"
    assert "allow_live_llm=true" in result["errors"][0]


def test_domain_request_auto_enables_synthetic_data_when_default_data_is_missing(tmp_path: Path) -> None:
    adapter = FakeDomainAdapter()
    store = YieldMindStore(tmp_path / "yieldmind.sqlite3")
    missing_data = tmp_path / "missing.csv"

    result = YieldDomainWorkflow(store=store, adapter=adapter).run(
        DomainWorkflowRequest(
            data_path=str(missing_data),
            run_dir=str(tmp_path / "auto-synthetic"),
            synthetic_data=None,
            external_search=False,
            require_search_results=False,
        )
    )

    assert result["status"] == "passed"
    assert result["manager_args"]["synthetic_data"] is True


def test_domain_workflow_enqueue_uses_distinct_task_type_and_is_idempotent(
    tmp_path: Path, monkeypatch
) -> None:
    store = YieldMindStore(tmp_path / "yieldmind.sqlite3")
    published: list[dict[str, Any]] = []
    monkeypatch.setattr("yieldmind.task_queue.redis_health", lambda: {"ok": True})
    monkeypatch.setattr(
        "yieldmind.task_queue.run_domain_workflow_task.apply_async",
        lambda **kwargs: published.append(kwargs),
    )
    request = EnqueueDomainWorkflowRequest(
        workflow=DomainWorkflowRequest(
            run_dir=str(tmp_path / "queued-run"),
            allow_live_llm=False,
        ),
        idempotency_key="domain-workflow-test",
    )

    first = enqueue_domain_workflow(request, store=store)
    second = enqueue_domain_workflow(request, store=store)

    assert first["task"]["task_type"] == "domain_workflow"
    assert first["task"]["status"] == "queued"
    assert first["idempotent"] is False
    assert second["idempotent"] is True
    assert second["task"]["task_id"] == first["task"]["task_id"]
    assert len(published) == 1
    assert published[0]["queue"] == "yieldmind"
    assert published[0]["args"][1]["allow_live_llm"] is False


def test_domain_stream_exposes_agent_actions_and_manager_decisions(monkeypatch) -> None:
    def fake_run_domain_workflow(request, *, store=None, on_stage_progress=None, **kwargs):
        on_stage_progress(
            {
                "type": "agent_progress",
                "agent": "CandidateAgent",
                "stage": "candidate",
                "action": "propose_candidate_strategy",
                "status": "running",
                "message": "candidate running",
                "ts": 1.0,
            }
        )
        on_stage_progress(
            {
                "type": "manager_decision",
                "agent": "Agent Manager",
                "stage": "candidate",
                "action": "route_next_agent",
                "status": "passed",
                "message": "candidate -> model",
                "decision": {
                    "next_node": "model",
                    "next_agent": "ModelAgent",
                    "reason_code": "phase_completed",
                },
                "ts": 2.0,
            }
        )
        return {"run_id": "run_domain_stream", "status": "passed"}

    monkeypatch.setattr(yieldmind_api, "run_domain_workflow", fake_run_domain_workflow)
    response = TestClient(yieldmind_api.app).post(
        "/api/workflows/domain/stream",
        json={"allow_live_llm": False},
    )

    assert response.status_code == 200
    events = [json.loads(line) for line in response.text.splitlines() if line]
    assert [event["type"] for event in events] == ["agent_progress", "manager_decision", "result"]
    assert events[0]["agent"] == "CandidateAgent"
    assert events[1]["decision"]["next_agent"] == "ModelAgent"


def test_operation_runtime_log_is_streamed_once_as_agent_detail(tmp_path: Path) -> None:
    emitted: list[dict[str, Any]] = []
    state: dict[str, Any] = {"run_id": "run_detail", "run_dir": str(tmp_path)}
    adapter = RealYieldDomainAdapter(on_detail_event=lambda _state, detail: emitted.append(detail))
    event_path = tmp_path / "operation_events.jsonl"
    event_path.write_text(
        "\n".join(
            [
                json.dumps({"sender": "operation", "label": "OperationAgent:", "content": "Model attempt #0 rejected: invalid schema", "ts": 1.0}),
                json.dumps({"sender": "operation", "label": "OperationAgent:", "content": "Accepted searched model on attempt #1.", "ts": 2.0}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    adapter._drain_operation_events(state, event_path)
    adapter._drain_operation_events(state, event_path)

    assert len(emitted) == 2
    assert emitted[0]["agent"] == "OperationAgent"
    assert emitted[0]["attempt"] == 1
    assert emitted[0]["detail_status"] == "warning"
    assert emitted[1]["attempt"] == 2
    assert emitted[1]["detail_status"] == "success"
    assert "runtime_event" in emitted[1]["output_summary"]


def test_real_operation_timeout_is_failed_not_passed(tmp_path: Path, monkeypatch) -> None:
    emitted: list[dict[str, Any]] = []
    monkeypatch.setattr(
        "yieldmind.domain_workflow.run_managed_process",
        lambda *args, **kwargs: ManagedProcessOutcome(
            status="timed_out",
            pid=123,
            exitcode=-15,
            duration_seconds=3.0,
            terminate_sent=True,
            process_group=True,
        ),
    )
    state: dict[str, Any] = {
        "run_dir": str(tmp_path),
        "runtime_env": {},
        "operation_timeout_seconds": 3.0,
    }

    result = RealYieldDomainAdapter(
        on_detail_event=lambda _state, detail: emitted.append(detail),
    ).operation(state)

    assert result["ok"] is False
    assert result["operation_result"]["timed_out"] is True
    assert result["operation_result"]["rcode"] == 1
    assert "timed out" in result["error"]
    assert [event["output_summary"]["phase"] for event in emitted] == [
        "process_started",
        "result_returned",
    ]
    assert emitted[-1]["detail_status"] == "warning"
    assert "3.00" in emitted[-1]["message"]
