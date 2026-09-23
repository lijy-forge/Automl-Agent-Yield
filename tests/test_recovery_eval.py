from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from yieldmind.recovery_eval import FaultInjectingRegistry, validate_recovery_eval_payload
from yieldmind.tools import ToolResult


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = PROJECT_ROOT / "evals" / "yieldmind_recovery_cases_v1.json"


class _BaseRegistry:
    def __init__(self) -> None:
        self.calls = 0

    def execute(self, name, args, *, run_id=None, idempotency_key="") -> ToolResult:
        self.calls += 1
        return ToolResult(ok=True, result={"name": name})


def test_recovery_eval_v1_has_balanced_controls_and_recovery_routes() -> None:
    suite = validate_recovery_eval_payload(json.loads(CASES_PATH.read_text(encoding="utf-8")))
    route_counts = {
        route: sum(case.expected_route == route for case in suite.cases)
        for route in ("none", "local_repair", "replan", "fatal")
    }

    assert len(suite.cases) == 30
    assert route_counts == {"none": 5, "local_repair": 16, "replan": 4, "fatal": 5}
    assert "not production" in suite.label_scope.lower()


def test_recovery_eval_rejects_fault_on_clean_control() -> None:
    payload = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    payload["cases"][0]["fault_tool"] = "run_fixed_baseline_eval"
    payload["cases"][0]["fail_times"] = 1

    with pytest.raises(ValidationError, match="Clean controls"):
        validate_recovery_eval_payload(payload)


def test_fault_injection_is_bounded_and_then_delegates() -> None:
    suite = validate_recovery_eval_payload(json.loads(CASES_PATH.read_text(encoding="utf-8")))
    case = next(item for item in suite.cases if item.expected_route == "local_repair")
    base = _BaseRegistry()
    registry = FaultInjectingRegistry(base, case)  # type: ignore[arg-type]

    first = registry.execute(case.fault_tool, {})
    second = registry.execute(case.fault_tool, {})

    assert first.ok is False
    assert first.mode == "controlled_fault_injection"
    assert second.ok is True
    assert base.calls == 1
