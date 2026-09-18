from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from yieldmind.database import YieldMindStore
from yieldmind.domain_workflow import DomainWorkflowRequest, YieldDomainWorkflow
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


def _run(tmp_path: Path, adapter: FakeDomainAdapter, *, n_revise: int = 1) -> dict[str, Any]:
    store = YieldMindStore(tmp_path / "yieldmind.sqlite3")
    return YieldDomainWorkflow(store=store, adapter=adapter).run(
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
    result = _run(tmp_path, adapter)

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


def test_domain_stategraph_routes_cancelled_operation_directly_to_cancel(tmp_path: Path) -> None:
    adapter = FakeDomainAdapter(cancel_operation=True)
    result = _run(tmp_path, adapter)

    assert result["status"] == "cancelled"
    assert "review" not in adapter.calls
    assert [item["stage"] for item in result["stages"]][-2:] == ["cancel", "finish"]
    assert result["process_control"]["terminate_sent"] is True


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
