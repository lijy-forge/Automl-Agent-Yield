"""Deterministic evaluation protocol for verified Repair Memory retrieval."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from yieldmind.database import YieldMindStore, connect, json_dumps
from yieldmind.repair_memory import RepairMemoryStore


class RepairMemoryFixture(BaseModel):
    id: str = Field(min_length=1)
    workspace_id: str = "repair_eval"
    execution_mode: str = "llm_freeform"
    error: str = Field(min_length=1)
    repair_summary: str = Field(min_length=1)
    applicability: dict[str, Any] = Field(default_factory=dict)
    verified_age_days: float = Field(default=0.0, ge=0.0)
    provenance_kind: Literal["synthetic_contract", "simulated_workflow", "real_run"] = "synthetic_contract"


class RepairMemoryEvalCase(BaseModel):
    id: str = Field(min_length=1)
    category: str = Field(min_length=1)
    query: str = Field(min_length=1)
    workspace_id: str = "repair_eval"
    execution_mode: str = "llm_freeform"
    applicability: dict[str, Any] = Field(default_factory=dict)
    expected_memory_ids: list[str] = Field(default_factory=list)
    forbidden_memory_ids: list[str] = Field(default_factory=list)
    expected_rejection_reasons: list[
        Literal[
            "verification_expired",
            "applicability_mismatch",
            "error_category_mismatch",
            "dependency_family_mismatch",
        ]
    ] = Field(default_factory=list)
    top_k: int = Field(default=3, ge=1, le=20)
    min_relevance: float = Field(default=0.08, ge=0.0, le=1.0)
    reuse_outcome: Literal["success", "failure", "not_observed"] = "not_observed"

    @model_validator(mode="after")
    def validate_labels(self) -> "RepairMemoryEvalCase":
        overlap = set(self.expected_memory_ids) & set(self.forbidden_memory_ids)
        if overlap:
            raise ValueError(f"expected and forbidden memory IDs overlap: {sorted(overlap)}")
        return self


class RepairMemoryEvalThresholds(BaseModel):
    min_top1_accuracy: float = Field(default=0.8, ge=0.0, le=1.0)
    min_recall_at_k: float = Field(default=1.0, ge=0.0, le=1.0)
    min_safe_abstention_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    min_guardrail_rejection_accuracy: float = Field(default=1.0, ge=0.0, le=1.0)
    max_misleading_hit_rate: float = Field(default=0.0, ge=0.0, le=1.0)


class RepairMemoryEvalSuite(BaseModel):
    version: str = Field(min_length=1)
    label_scope: str = Field(min_length=1)
    memories: list[RepairMemoryFixture] = Field(min_length=3)
    cases: list[RepairMemoryEvalCase] = Field(min_length=10)
    thresholds: RepairMemoryEvalThresholds = Field(default_factory=RepairMemoryEvalThresholds)
    limitations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_references(self) -> "RepairMemoryEvalSuite":
        memory_ids = [item.id for item in self.memories]
        case_ids = [item.id for item in self.cases]
        if len(memory_ids) != len(set(memory_ids)):
            raise ValueError("Repair Memory fixture IDs must be unique.")
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("Repair Memory eval case IDs must be unique.")
        known = set(memory_ids)
        referenced = {
            memory_id
            for case in self.cases
            for memory_id in [*case.expected_memory_ids, *case.forbidden_memory_ids]
        }
        unknown = sorted(referenced - known)
        if unknown:
            raise ValueError(f"Repair Memory eval cases reference unknown fixture IDs: {unknown}")
        if not any(case.expected_memory_ids for case in self.cases):
            raise ValueError("Repair Memory eval suite requires positive retrieval cases.")
        if not any(not case.expected_memory_ids for case in self.cases):
            raise ValueError("Repair Memory eval suite requires negative/abstention cases.")
        return self


def validate_repair_memory_eval_payload(payload: dict[str, Any]) -> RepairMemoryEvalSuite:
    return RepairMemoryEvalSuite.model_validate(payload)


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def evaluate_repair_memory_suite(
    suite: RepairMemoryEvalSuite,
    *,
    db_path: str | Path | None = None,
    now: float = 2_000_000_000.0,
) -> dict[str, Any]:
    """Evaluate ranking and safety using the real RepairMemoryStore path.

    The fixtures are inserted through candidate -> confirmed transitions rather
    than directly materialized as confirmed database rows. The fixed clock makes
    expiry and ranking results reproducible. This benchmark intentionally does
    not claim a repair-success rate when no case has an observed reuse outcome.
    """

    started = time.perf_counter()
    temporary_directory: tempfile.TemporaryDirectory[str] | None = None
    if db_path is None:
        temporary_directory = tempfile.TemporaryDirectory(prefix="yieldmind_repair_memory_eval_")
        resolved_db_path = Path(temporary_directory.name) / "yieldmind.sqlite3"
    else:
        resolved_db_path = Path(db_path)

    try:
        store = YieldMindStore(resolved_db_path)
        repair_store = RepairMemoryStore(store, clock=lambda: now)
        logical_to_database: dict[str, str] = {}
        database_to_logical: dict[str, str] = {}

        for fixture in suite.memories:
            run_id = store.create_run(
                mode="live_llm",
                source="repair_memory_eval",
                status="running",
                metadata={"workspace_id": fixture.workspace_id, "fixture_id": fixture.id},
            )
            candidate = repair_store.record_candidate(
                workspace_id=fixture.workspace_id,
                run_id=run_id,
                execution_mode=fixture.execution_mode,
                operation_result={"rcode": 1, "error_logs": [fixture.error]},
                applicability_context=fixture.applicability,
            )
            if not candidate:
                raise RuntimeError(f"Failed to create Repair Memory fixture: {fixture.id}")
            confirmed = repair_store.confirm(
                str(candidate["memory_id"]),
                successful_operation={"rcode": 0, "action_result": "Synthetic contract repair passed."},
                manager_review={"passed": True, "decision": "eval_fixture_verified"},
                repair_summary=fixture.repair_summary,
            )
            if not confirmed or confirmed.get("validation_status") != "confirmed":
                raise RuntimeError(f"Failed to confirm Repair Memory fixture: {fixture.id}")
            memory_id = str(confirmed["memory_id"])
            logical_to_database[fixture.id] = memory_id
            database_to_logical[memory_id] = fixture.id

            if fixture.verified_age_days:
                verified_at = now - fixture.verified_age_days * 86400.0
                metadata = dict(confirmed.get("metadata") or {})
                metadata["verified_at"] = verified_at
                metadata["last_verified_at"] = verified_at
                with connect(store.db_path) as conn:
                    conn.execute(
                        "UPDATE yieldmind_memories SET updated_at=?, metadata_json=? WHERE memory_id=?",
                        (verified_at, json_dumps(metadata), memory_id),
                    )

        diagnostics: list[dict[str, Any]] = []
        positive_count = 0
        top1_hits = 0
        recall_hits = 0
        reciprocal_rank_total = 0.0
        negative_count = 0
        safe_abstentions = 0
        misleading_cases = 0
        guardrail_count = 0
        guardrail_hits = 0
        observed_reuse_count = 0
        observed_reuse_successes = 0

        for case in suite.cases:
            retrieved = repair_store.retrieve(
                workspace_id=case.workspace_id,
                execution_mode=case.execution_mode,
                limit=case.top_k,
                query_text=case.query,
                min_relevance=case.min_relevance,
                applicability_context=case.applicability,
                strict_error_match=True,
            )
            selected_ids = [
                database_to_logical.get(str(memory.get("memory_id") or ""), "unknown")
                for memory in retrieved
            ]
            expected = set(case.expected_memory_ids)
            forbidden = set(case.forbidden_memory_ids)
            forbidden_hits = [memory_id for memory_id in selected_ids if memory_id in forbidden]
            expected_ranks = [
                index + 1 for index, memory_id in enumerate(selected_ids) if memory_id in expected
            ]
            top1_hit = bool(selected_ids and selected_ids[0] in expected)
            recall_hit = bool(expected_ranks)

            audit = dict(repair_store.last_retrieval_audit)
            rejected = [dict(item) for item in audit.get("rejected", [])]
            rejected_logical = [
                {
                    **item,
                    "memory_id": database_to_logical.get(str(item.get("memory_id") or ""), "unknown"),
                }
                for item in rejected
            ]
            rejection_reasons = {str(item.get("reason") or "") for item in rejected_logical}
            expected_reasons = set(case.expected_rejection_reasons)
            rejection_reason_match = expected_reasons.issubset(rejection_reasons)

            if expected:
                positive_count += 1
                top1_hits += int(top1_hit)
                recall_hits += int(recall_hit)
                if expected_ranks:
                    reciprocal_rank_total += 1.0 / min(expected_ranks)
                case_passed = recall_hit and not forbidden_hits
            else:
                negative_count += 1
                safe_abstention = not selected_ids
                safe_abstentions += int(safe_abstention)
                case_passed = safe_abstention and not forbidden_hits and rejection_reason_match

            if forbidden_hits:
                misleading_cases += 1
            if expected_reasons:
                guardrail_count += 1
                guardrail_hits += int(rejection_reason_match)
            if case.reuse_outcome != "not_observed":
                observed_reuse_count += 1
                observed_reuse_successes += int(case.reuse_outcome == "success")

            diagnostics.append(
                {
                    "case_id": case.id,
                    "category": case.category,
                    "passed": case_passed,
                    "expected_memory_ids": case.expected_memory_ids,
                    "selected_memory_ids": selected_ids,
                    "forbidden_hits": forbidden_hits,
                    "top1_hit": top1_hit if expected else None,
                    "recall_at_k_hit": recall_hit if expected else None,
                    "expected_rejection_reasons": case.expected_rejection_reasons,
                    "observed_rejection_reasons": sorted(rejection_reasons),
                    "rejection_reason_match": rejection_reason_match if expected_reasons else None,
                    "retrieval_audit": {
                        **audit,
                        "rejected": rejected_logical,
                    },
                }
            )

        top1_accuracy = _ratio(top1_hits, positive_count)
        recall_at_k = _ratio(recall_hits, positive_count)
        mrr = round(reciprocal_rank_total / positive_count, 6) if positive_count else None
        safe_abstention_rate = _ratio(safe_abstentions, negative_count)
        misleading_hit_rate = _ratio(misleading_cases, len(suite.cases))
        guardrail_rejection_accuracy = _ratio(guardrail_hits, guardrail_count)
        repair_success_rate = _ratio(observed_reuse_successes, observed_reuse_count)

        threshold_checks = {
            "top1_accuracy": bool(
                top1_accuracy is not None and top1_accuracy >= suite.thresholds.min_top1_accuracy
            ),
            "recall_at_k": bool(
                recall_at_k is not None and recall_at_k >= suite.thresholds.min_recall_at_k
            ),
            "safe_abstention_rate": bool(
                safe_abstention_rate is not None
                and safe_abstention_rate >= suite.thresholds.min_safe_abstention_rate
            ),
            "guardrail_rejection_accuracy": bool(
                guardrail_rejection_accuracy is not None
                and guardrail_rejection_accuracy >= suite.thresholds.min_guardrail_rejection_accuracy
            ),
            "misleading_hit_rate": bool(
                misleading_hit_rate is not None
                and misleading_hit_rate <= suite.thresholds.max_misleading_hit_rate
            ),
        }
        return {
            "status": "passed" if all(threshold_checks.values()) else "failed",
            "benchmark_version": suite.version,
            "label_scope": suite.label_scope,
            "fixture_count": len(suite.memories),
            "case_count": len(suite.cases),
            "positive_case_count": positive_count,
            "negative_case_count": negative_count,
            "metrics": {
                "top1_accuracy": top1_accuracy,
                "recall_at_k": recall_at_k,
                "mrr": mrr,
                "safe_abstention_rate": safe_abstention_rate,
                "misleading_hit_rate": misleading_hit_rate,
                "guardrail_rejection_accuracy": guardrail_rejection_accuracy,
            },
            "repair_success": {
                "observed_case_count": observed_reuse_count,
                "success_count": observed_reuse_successes,
                "rate": repair_success_rate,
                "claimable": observed_reuse_count > 0,
                "note": (
                    "No observed reuse outcomes; repair-success rate is intentionally null."
                    if not observed_reuse_count
                    else "Rate covers only cases with an explicitly observed reuse outcome."
                ),
            },
            "thresholds": suite.thresholds.model_dump(mode="json"),
            "threshold_checks": threshold_checks,
            "embedding_execution_mode": "offline_deterministic_hashing",
            "real_llm_calls": 0,
            "simulated_model_calls": 0,
            "fixed_clock_epoch": now,
            "case_diagnostics": diagnostics,
            "limitations": suite.limitations,
            "duration_seconds": round(time.perf_counter() - started, 6),
        }
    finally:
        if temporary_directory is not None:
            temporary_directory.cleanup()
