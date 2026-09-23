from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from yieldmind.repair_memory_eval import (
    evaluate_repair_memory_suite,
    validate_repair_memory_eval_payload,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = PROJECT_ROOT / "evals" / "yieldmind_repair_memory_cases_v1.json"


def test_repair_memory_eval_v1_contract_is_valid_and_explicitly_synthetic() -> None:
    suite = validate_repair_memory_eval_payload(json.loads(CASES_PATH.read_text(encoding="utf-8")))

    assert len(suite.memories) == 8
    assert len(suite.cases) == 13
    assert all(memory.provenance_kind == "synthetic_contract" for memory in suite.memories)
    assert all(case.reuse_outcome == "not_observed" for case in suite.cases)
    assert any(case.expected_memory_ids for case in suite.cases)
    assert any(not case.expected_memory_ids for case in suite.cases)
    assert "not production" in suite.label_scope.lower()


def test_repair_memory_eval_contract_rejects_unknown_labels() -> None:
    payload = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    payload["cases"][0]["expected_memory_ids"] = ["missing_fixture"]

    with pytest.raises(ValidationError, match="unknown fixture IDs"):
        validate_repair_memory_eval_payload(payload)


def test_repair_memory_eval_runs_real_store_path_without_claiming_success_rate() -> None:
    suite = validate_repair_memory_eval_payload(json.loads(CASES_PATH.read_text(encoding="utf-8")))

    report = evaluate_repair_memory_suite(suite)

    assert report["case_count"] == 13
    assert report["positive_case_count"] == 8
    assert report["negative_case_count"] == 5
    assert report["real_llm_calls"] == 0
    assert report["embedding_execution_mode"] == "offline_deterministic_hashing"
    assert report["repair_success"] == {
        "observed_case_count": 0,
        "success_count": 0,
        "rate": None,
        "claimable": False,
        "note": "No observed reuse outcomes; repair-success rate is intentionally null.",
    }
    assert len(report["case_diagnostics"]) == 13
    assert report["status"] == "passed"
    assert report["metrics"] == {
        "top1_accuracy": 1.0,
        "recall_at_k": 1.0,
        "mrr": 1.0,
        "safe_abstention_rate": 1.0,
        "misleading_hit_rate": 0.0,
        "guardrail_rejection_accuracy": 1.0,
    }
    assert all(report["threshold_checks"].values())
