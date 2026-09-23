"""Paired controlled-failure evaluation for bounded workflow recovery."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from yieldmind.database import YieldMindStore
from yieldmind.tools import ToolRegistry, ToolResult, registry_for_workspace
from yieldmind.workflow import WorkflowRequest, YieldMindWorkflow


RecoveryRoute = Literal["none", "local_repair", "replan", "fatal"]


class RecoveryEvalCase(BaseModel):
    id: str = Field(min_length=1)
    category: str = Field(min_length=1)
    random_state: int = Field(ge=0)
    fault_tool: str = ""
    fail_times: int = Field(default=1, ge=0, le=3)
    expected_route: RecoveryRoute
    error_message: str = "Injected controlled failure."

    @model_validator(mode="after")
    def validate_fault(self) -> "RecoveryEvalCase":
        if self.expected_route == "none" and (self.fault_tool or self.fail_times):
            raise ValueError("Clean controls cannot declare a fault.")
        if self.expected_route != "none" and (not self.fault_tool or self.fail_times < 1):
            raise ValueError("Fault cases require fault_tool and fail_times >= 1.")
        return self


class RecoveryEvalSuite(BaseModel):
    version: str = Field(min_length=1)
    label_scope: str = Field(min_length=1)
    cases: list[RecoveryEvalCase] = Field(min_length=20)
    limitations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_suite(self) -> "RecoveryEvalSuite":
        ids = [case.id for case in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("Recovery eval case IDs must be unique.")
        routes = {case.expected_route for case in self.cases}
        required = {"none", "local_repair", "replan", "fatal"}
        if not required.issubset(routes):
            raise ValueError(f"Recovery eval suite must cover routes: {sorted(required)}")
        return self


class FaultInjectingRegistry:
    """Delegate to real tools except for a bounded, declared fault."""

    def __init__(self, base: ToolRegistry, case: RecoveryEvalCase) -> None:
        self.base = base
        self.case = case
        self.target_calls = 0

    def execute(
        self,
        name: str,
        args: dict[str, Any],
        *,
        run_id: str | None = None,
        idempotency_key: str = "",
    ) -> ToolResult:
        if name == self.case.fault_tool:
            self.target_calls += 1
            if self.target_calls <= self.case.fail_times:
                return ToolResult(
                    ok=False,
                    mode="controlled_fault_injection",
                    error=self.case.error_message,
                    result={
                        "fault_case_id": self.case.id,
                        "fault_tool": self.case.fault_tool,
                        "injected_attempt": self.target_calls,
                    },
                )
        return self.base.execute(
            name,
            args,
            run_id=run_id,
            idempotency_key=idempotency_key,
        )


def validate_recovery_eval_payload(payload: dict[str, Any]) -> RecoveryEvalSuite:
    return RecoveryEvalSuite.model_validate(payload)


def _completed(result: dict[str, Any]) -> bool:
    stages = list(result.get("stages") or [])
    finish = stages[-1] if stages else {}
    artifacts = dict(result.get("artifacts") or {})
    return bool(
        result.get("status") == "passed"
        and finish.get("stage") == "finish"
        and finish.get("status") == "passed"
        and artifacts.get("report_json")
        and not result.get("errors")
    )


def _route_correct(case: RecoveryEvalCase, result: dict[str, Any]) -> bool:
    routes = [str(item.get("route") or "") for item in (result.get("route_history") or [])]
    completed = _completed(result)
    if case.expected_route == "none":
        return completed and not ({"local_repair", "replan"} & set(routes))
    if case.expected_route == "fatal":
        return not completed and not ({"local_repair", "replan"} & set(routes))
    return completed and case.expected_route in routes


def _run_arm(
    case: RecoveryEvalCase,
    *,
    arm: Literal["baseline", "recovery"],
    store: YieldMindStore,
) -> dict[str, Any]:
    registry = FaultInjectingRegistry(registry_for_workspace(store=store), case)
    started = time.perf_counter()
    result = YieldMindWorkflow(store=store, registry=registry).run(
        WorkflowRequest(
            prompt=f"Controlled recovery evaluation case {case.id}",
            n_samples=20,
            n_splits=2,
            random_state=case.random_state,
            max_local_repairs=0 if arm == "baseline" else 1,
            max_replans=0 if arm == "baseline" else 1,
            use_knowledge=case.expected_route == "fatal",
            knowledge_query="controlled unavailable evidence",
        )
    )
    duration = time.perf_counter() - started
    return {
        "arm": arm,
        "completed": _completed(result),
        "status": result.get("status"),
        "run_id": result.get("run_id"),
        "duration_seconds": round(duration, 6),
        "local_repair_count": int(result.get("local_repair_count") or 0),
        "replan_count": int(result.get("replan_count") or 0),
        "routes": [str(item.get("route") or "") for item in (result.get("route_history") or [])],
        "failure_history": list(result.get("failure_history") or []),
        "errors": list(result.get("errors") or []),
        "artifact_names": sorted((result.get("artifacts") or {}).keys()),
        "injected_fault_count": min(registry.target_calls, case.fail_times),
        "route_correct": _route_correct(case, result),
    }


def evaluate_recovery_suite(
    suite: RecoveryEvalSuite,
    *,
    db_path: str | Path | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    temporary_directory: tempfile.TemporaryDirectory[str] | None = None
    if db_path is None:
        temporary_directory = tempfile.TemporaryDirectory(prefix="yieldmind_recovery_eval_")
        resolved_db_path = Path(temporary_directory.name) / "yieldmind.sqlite3"
    else:
        resolved_db_path = Path(db_path)
    try:
        store = YieldMindStore(resolved_db_path)
        cases: list[dict[str, Any]] = []
        for case in suite.cases:
            baseline = _run_arm(case, arm="baseline", store=store)
            recovery = _run_arm(case, arm="recovery", store=store)
            cases.append(
                {
                    "case_id": case.id,
                    "category": case.category,
                    "fault_tool": case.fault_tool,
                    "expected_route": case.expected_route,
                    "baseline": baseline,
                    "recovery": recovery,
                    "recovered": not baseline["completed"] and recovery["completed"],
                }
            )

        total = len(cases)
        baseline_completed = sum(case["baseline"]["completed"] for case in cases)
        recovery_completed = sum(case["recovery"]["completed"] for case in cases)
        recoverable = [case for case in cases if case["expected_route"] in {"local_repair", "replan"}]
        recovered = sum(case["recovered"] for case in recoverable)
        route_correct = sum(case["recovery"]["route_correct"] for case in cases)
        baseline_rate = baseline_completed / total
        recovery_rate = recovery_completed / total
        report = {
            "status": "passed" if route_correct == total and recovered == len(recoverable) else "failed",
            "benchmark_version": suite.version,
            "label_scope": suite.label_scope,
            "case_count": total,
            "metrics": {
                "baseline_completed": baseline_completed,
                "baseline_completion_rate": round(baseline_rate, 6),
                "recovery_completed": recovery_completed,
                "recovery_completion_rate": round(recovery_rate, 6),
                "completion_rate_lift_percentage_points": round((recovery_rate - baseline_rate) * 100, 4),
                "recoverable_failure_count": len(recoverable),
                "recovered_failure_count": recovered,
                "recoverable_failure_recovery_rate": round(recovered / len(recoverable), 6),
                "route_accuracy": round(route_correct / total, 6),
                "baseline_mean_duration_seconds": round(
                    sum(case["baseline"]["duration_seconds"] for case in cases) / total, 6
                ),
                "recovery_mean_duration_seconds": round(
                    sum(case["recovery"]["duration_seconds"] for case in cases) / total, 6
                ),
                "mean_local_repairs": round(
                    sum(case["recovery"]["local_repair_count"] for case in cases) / total, 6
                ),
                "mean_replans": round(
                    sum(case["recovery"]["replan_count"] for case in cases) / total, 6
                ),
            },
            "real_llm_calls": 0,
            "simulated_model_calls": 0,
            "execution_mode": "real_tools_with_controlled_fault_injection",
            "cases": cases,
            "limitations": suite.limitations,
            "duration_seconds": round(time.perf_counter() - started, 6),
        }
        return report
    finally:
        if temporary_directory is not None:
            temporary_directory.cleanup()
