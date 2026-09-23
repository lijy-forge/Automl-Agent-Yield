from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import yieldmind.api as yieldmind_api
from yieldmind.database import YieldMindStore
from yieldmind.api import app
from yieldmind.repair_memory import RepairMemoryStore, normalized_error_signature
from scripts.export_yieldmind_repair_memory_outcomes import export_report


def test_error_signature_removes_paths_and_volatile_numbers() -> None:
    left = normalized_error_signature("File /tmp/run_123/train.py line 42: ValueError 17.5")
    right = normalized_error_signature("File /tmp/run_999/train.py line 87: ValueError 22.1")

    assert left == right
    assert "<path>" in left
    assert "<num>" in left


def test_candidate_memory_is_not_retrieved_until_verified(tmp_path) -> None:
    store = YieldMindStore(tmp_path / "yieldmind.sqlite3")
    repair = RepairMemoryStore(store)
    run_id = store.create_run(mode="live_llm", source="test", status="running")
    failure = {
        "rcode": 1,
        "action_result": "Generated script exited with non-zero return code.",
        "error_logs": ["ValueError: input contains NaN in /tmp/run/train.py line 44"],
    }

    candidate = repair.record_candidate(
        workspace_id="workspace_a",
        run_id=run_id,
        execution_mode="llm_freeform",
        operation_result=failure,
    )

    assert candidate is not None
    assert candidate["validation_status"] == "candidate"
    assert repair.retrieve(workspace_id="workspace_a", execution_mode="llm_freeform") == []

    confirmed = repair.confirm(
        candidate["memory_id"],
        successful_operation={"rcode": 0, "action_result": "Training and artifact checks passed."},
        manager_review={"passed": True, "decision": "accepted"},
        repair_summary="Add fold-local imputation before fitting the estimator.",
    )

    assert confirmed is not None
    assert confirmed["validation_status"] == "confirmed"
    retrieved = repair.retrieve(workspace_id="workspace_a", execution_mode="llm_freeform")
    assert [item["memory_id"] for item in retrieved] == [candidate["memory_id"]]
    assert "fold-local imputation" in retrieved[0]["content"]
    assert repair.retrieve(workspace_id="workspace_b", execution_mode="llm_freeform") == []
    assert repair.retrieve(workspace_id="workspace_a", execution_mode="plugin") == []


def test_repair_candidate_is_idempotent_for_same_run_and_signature(tmp_path) -> None:
    store = YieldMindStore(tmp_path / "yieldmind.sqlite3")
    repair = RepairMemoryStore(store)
    run_id = store.create_run(mode="live_llm", source="test", status="running")
    result = {"rcode": 1, "error_logs": ["TypeError: unsupported dtype float128"]}

    first = repair.record_candidate(
        workspace_id="default",
        run_id=run_id,
        execution_mode="free_search",
        operation_result=result,
    )
    second = repair.record_candidate(
        workspace_id="default",
        run_id=run_id,
        execution_mode="free_search",
        operation_result=result,
    )

    assert first is not None and second is not None
    assert first["memory_id"] == second["memory_id"]


def test_confirmed_repair_fingerprint_is_not_duplicated_across_runs(tmp_path) -> None:
    store = YieldMindStore(tmp_path / "yieldmind.sqlite3")
    repair = RepairMemoryStore(store)
    result = {"rcode": 1, "error_logs": ["Missing required artifact metrics.json"]}
    first_run = store.create_run(mode="live_llm", source="test", status="running")
    candidate = repair.record_candidate(
        workspace_id="workspace_a",
        run_id=first_run,
        execution_mode="llm_freeform",
        operation_result=result,
    )
    assert candidate is not None
    repair.confirm(
        candidate["memory_id"],
        successful_operation={"rcode": 0, "action_result": "metrics.json verified"},
        manager_review={"passed": True, "decision": "accepted"},
    )
    second_run = store.create_run(mode="live_llm", source="test", status="running")

    duplicate = repair.record_candidate(
        workspace_id="workspace_a",
        run_id=second_run,
        execution_mode="llm_freeform",
        operation_result=result,
    )

    assert duplicate is None
    assert len(repair.retrieve(workspace_id="workspace_a", execution_mode="llm_freeform")) == 1


def _confirmed_memory(
    repair: RepairMemoryStore,
    store: YieldMindStore,
    *,
    error: str,
    repair_summary: str,
    applicability_context: dict | None = None,
) -> dict:
    run_id = store.create_run(mode="live_llm", source="test", status="running")
    candidate = repair.record_candidate(
        workspace_id="semantic_workspace",
        run_id=run_id,
        execution_mode="llm_freeform",
        operation_result={"rcode": 1, "error_logs": [error]},
        applicability_context=applicability_context,
    )
    assert candidate is not None
    confirmed = repair.confirm(
        candidate["memory_id"],
        successful_operation={"rcode": 0, "action_result": "Guarded rerun passed."},
        manager_review={"passed": True, "decision": "accepted"},
        repair_summary=repair_summary,
    )
    assert confirmed is not None
    return confirmed


def test_repair_memories_rank_by_current_error_similarity(tmp_path) -> None:
    store = YieldMindStore(tmp_path / "yieldmind.sqlite3")
    repair = RepairMemoryStore(store)
    nan_memory = _confirmed_memory(
        repair,
        store,
        error="ValueError: estimator input contains NaN in fold 3",
        repair_summary="Add fold-local imputation before estimator fitting.",
    )
    _confirmed_memory(
        repair,
        store,
        error="TypeError: categorical material_type column has object dtype",
        repair_summary="Encode the categorical column inside each training fold.",
    )

    retrieved = repair.retrieve(
        workspace_id="semantic_workspace",
        execution_mode="llm_freeform",
        query_text="ValueError: model fit failed because input data contains NaN",
    )

    assert retrieved[0]["memory_id"] == nan_memory["memory_id"]
    assert retrieved[0]["retrieval"]["rank"] == 1
    assert retrieved[0]["retrieval"]["lexical_score"] > retrieved[1]["retrieval"]["lexical_score"]
    assert retrieved[0]["retrieval"]["method"] == "deterministic_hashing_embedding"
    assert "relevance=" in repair.prompt_context(retrieved)


def test_repair_retrieval_rejects_unrelated_error_category_and_dependency_family(tmp_path) -> None:
    store = YieldMindStore(tmp_path / "yieldmind.sqlite3")
    repair = RepairMemoryStore(store)
    _confirmed_memory(
        repair,
        store,
        error="ValueError: estimator input contains NaN",
        repair_summary="Apply fold-local imputation.",
    )
    _confirmed_memory(
        repair,
        store,
        error="ImportError: scikit-learn removed estimator API",
        repair_summary="Use the supported scikit-learn estimator API.",
    )

    retrieved = repair.retrieve(
        workspace_id="semantic_workspace",
        execution_mode="llm_freeform",
        query_text="AttributeError: numpy removed a legacy API alias",
        min_relevance=0.08,
        strict_error_match=True,
    )

    assert retrieved == []
    reasons = {item["reason"] for item in repair.last_retrieval_audit["rejected"]}
    assert "error_category_mismatch" in reasons
    assert "dependency_family_mismatch" in reasons


def test_repair_reuse_attempts_are_idempotent_and_only_observed_after_review(
    tmp_path,
    monkeypatch,
) -> None:
    store = YieldMindStore(tmp_path / "yieldmind.sqlite3")
    repair = RepairMemoryStore(store, clock=lambda: 2_000_000_000.0)
    memory = _confirmed_memory(
        repair,
        store,
        error="Missing required artifact metrics.json",
        repair_summary="Write metrics.json atomically before returning success.",
    )
    run_id = store.create_run(
        mode="live_llm",
        source="yield_domain_stategraph",
        status="running",
        metadata={"workspace_id": "semantic_workspace"},
    )

    first = repair.begin_reuse_attempt(
        memory["memory_id"],
        run_id=run_id,
        workspace_id="semantic_workspace",
        execution_mode="llm_freeform",
        matched_round=0,
        applied_round=1,
        retrieval={"rank": 1, "score": 0.91},
        query_signature="missing required artifact metrics.json",
    )
    duplicate = repair.begin_reuse_attempt(
        memory["memory_id"],
        run_id=run_id,
        workspace_id="semantic_workspace",
        execution_mode="llm_freeform",
        matched_round=0,
        applied_round=1,
    )

    assert first["reuse_id"] == duplicate["reuse_id"]
    assert first["status"] == "applied"
    assert repair.reuse_outcome_summary(workspace_id="semantic_workspace")["success_rate"] is None

    failed = repair.complete_reuse_attempt(
        first["reuse_id"],
        operation_result={"rcode": 1, "error_logs": ["metrics.json still missing"]},
        manager_review={"passed": False, "decision": "revise", "feedback": "artifact missing"},
    )
    cannot_overwrite = repair.complete_reuse_attempt(
        first["reuse_id"],
        operation_result={"rcode": 0, "action_result": "passed"},
        manager_review={"passed": True, "decision": "accepted"},
    )
    summary = repair.reuse_outcome_summary(workspace_id="semantic_workspace")

    assert failed is not None and failed["status"] == "failed"
    assert cannot_overwrite is not None and cannot_overwrite["status"] == "failed"
    assert summary["attempt_count"] == 1
    assert summary["observed_count"] == 1
    assert summary["failure_count"] == 1
    assert summary["success_rate"] == 0.0
    assert summary["computable"] is True
    assert summary["claimable"] is False
    assert summary["minimum_claimable_observations"] == 30

    monkeypatch.setattr(yieldmind_api, "store", store)
    api_response = TestClient(app).get(
        "/api/repair-memory/reuse-summary",
        params={"workspace_id": "semantic_workspace", "include_attempts": True},
    )
    report = export_report(store, workspace_id="semantic_workspace")

    assert api_response.status_code == 200
    assert api_response.json()["summary"]["success_rate"] == 0.0
    assert len(api_response.json()["attempts"]) == 1
    assert report["status"] == "preliminary"
    assert report["summary"]["aggregation_unit"] == "run_revision_episode"
    assert report["real_llm_calls"] == 0


def test_repair_memory_embedding_failure_degrades_to_lexical_ranking(tmp_path) -> None:
    store = YieldMindStore(tmp_path / "yieldmind.sqlite3")
    writer = RepairMemoryStore(store)
    target = _confirmed_memory(
        writer,
        store,
        error="KeyError: missing required feature solid_fraction",
        repair_summary="Validate the feature contract before training.",
    )
    _confirmed_memory(
        writer,
        store,
        error="ConnectionError: embedding endpoint unavailable",
        repair_summary="Check the loopback embedding service health.",
    )

    def unavailable_embedding():
        raise RuntimeError("embedding service unavailable")

    reader = RepairMemoryStore(store, embedding_factory=unavailable_embedding)
    retrieved = reader.retrieve(
        workspace_id="semantic_workspace",
        execution_mode="llm_freeform",
        query_text="KeyError missing feature solid_fraction",
    )

    assert retrieved[0]["memory_id"] == target["memory_id"]
    assert retrieved[0]["retrieval"]["method"] == "lexical_fallback"
    assert retrieved[0]["retrieval"]["degraded_error"] == "RuntimeError"


def test_data_schema_repair_is_rejected_when_feature_contract_differs(tmp_path) -> None:
    store = YieldMindStore(tmp_path / "yieldmind.sqlite3")
    repair = RepairMemoryStore(store)
    memory = _confirmed_memory(
        repair,
        store,
        error="ValueError: input feature matrix contains NaN",
        repair_summary="Apply fold-local imputation to the approved feature matrix.",
        applicability_context={
            "dataset_schema": "yield_schema_v1",
            "dataset_role": "augmented_development",
            "target_column": "yield_stress",
            "feature_signature": "features-v1",
        },
    )

    compatible = repair.retrieve(
        workspace_id="semantic_workspace",
        execution_mode="llm_freeform",
        query_text="ValueError input contains NaN",
        applicability_context={
            "dataset_schema": "yield_schema_v1",
            "dataset_role": "augmented_development",
            "target_column": "yield_stress",
            "feature_signature": "features-v1",
        },
    )
    incompatible = repair.retrieve(
        workspace_id="semantic_workspace",
        execution_mode="llm_freeform",
        query_text="ValueError input contains NaN",
        applicability_context={
            "dataset_schema": "yield_schema_v2",
            "dataset_role": "observed_development",
            "target_column": "tau0",
            "feature_signature": "features-v2",
        },
    )

    assert [item["memory_id"] for item in compatible] == [memory["memory_id"]]
    assert compatible[0]["applicability_match"]["compatible"] is True
    assert incompatible == []
    assert repair.last_retrieval_audit["rejected_count"] == 1
    mismatch_fields = {
        mismatch["field"]
        for mismatch in repair.last_retrieval_audit["rejected"][0]["mismatches"]
    }
    assert {"dataset_schema", "dataset_role", "target_column", "feature_signature"} <= mismatch_fields


def test_dependency_repair_requires_compatible_major_minor_versions(tmp_path) -> None:
    store = YieldMindStore(tmp_path / "yieldmind.sqlite3")
    repair = RepairMemoryStore(store)
    _confirmed_memory(
        repair,
        store,
        error="ImportError: scikit-learn version removed estimator API",
        repair_summary="Use the estimator constructor supported by the pinned dependency version.",
        applicability_context={
            "python_version": "3.11.9",
            "dependency_versions": {"scikit-learn": "1.4.2", "numpy": "1.26.4"},
        },
    )

    same_minor = repair.retrieve(
        workspace_id="semantic_workspace",
        execution_mode="llm_freeform",
        query_text="ImportError estimator API version mismatch",
        applicability_context={
            "python_version": "3.11.12",
            "dependency_versions": {"scikit-learn": "1.4.0", "numpy": "1.26.2"},
        },
    )
    different_minor = repair.retrieve(
        workspace_id="semantic_workspace",
        execution_mode="llm_freeform",
        query_text="ImportError estimator API version mismatch",
        applicability_context={
            "python_version": "3.12.1",
            "dependency_versions": {"scikit-learn": "1.5.0", "numpy": "2.0.0"},
        },
    )

    assert len(same_minor) == 1
    assert different_minor == []
    assert repair.last_retrieval_audit["rejected_count"] == 1


def test_verified_repair_expires_from_retrieval_after_policy_window(tmp_path) -> None:
    store = YieldMindStore(tmp_path / "yieldmind.sqlite3")
    now = [1_000_000.0]
    repair = RepairMemoryStore(store, max_age_days=90, clock=lambda: now[0])
    memory = _confirmed_memory(
        repair,
        store,
        error="ValueError: input contains NaN",
        repair_summary="Apply fold-local imputation.",
    )

    fresh = repair.retrieve(
        workspace_id="semantic_workspace",
        execution_mode="llm_freeform",
        query_text="ValueError input contains NaN",
    )
    now[0] += 91 * 86400
    expired = repair.retrieve(
        workspace_id="semantic_workspace",
        execution_mode="llm_freeform",
        query_text="ValueError input contains NaN",
    )

    assert [item["memory_id"] for item in fresh] == [memory["memory_id"]]
    assert fresh[0]["freshness"]["fresh"] is True
    assert expired == []
    rejection = repair.last_retrieval_audit["rejected"][0]
    assert rejection["memory_id"] == memory["memory_id"]
    assert rejection["reason"] == "verification_expired"
    assert rejection["freshness"]["age_days"] == 91.0


def test_expired_repair_can_be_idempotently_reconfirmed_by_passed_run(tmp_path) -> None:
    store = YieldMindStore(tmp_path / "yieldmind.sqlite3")
    now = [1_000_000.0]
    repair = RepairMemoryStore(store, max_age_days=90, clock=lambda: now[0])
    memory = _confirmed_memory(
        repair,
        store,
        error="TimeoutError: managed process timed out",
        repair_summary="Reduce the bounded search budget.",
    )
    now[0] += 91 * 86400
    evidence_run = store.create_run(
        mode="live_llm",
        source="yield_domain_stategraph",
        status="running",
        metadata={"workspace_id": "semantic_workspace"},
    )

    with pytest.raises(ValueError, match="passed evidence run"):
        repair.reconfirm_from_passed_run(
            memory["memory_id"],
            evidence_run_id=evidence_run,
            reviewer="tester",
        )
    unrelated_run = store.create_run(
        mode="live_llm",
        source="yield_domain_stategraph",
        status="passed",
        metadata={"workspace_id": "other_workspace"},
    )
    store.update_run(
        unrelated_run,
        status="passed",
        result={"execution_mode": "llm_freeform"},
        completed=True,
    )
    with pytest.raises(ValueError, match="workspace_id does not match"):
        repair.reconfirm_from_passed_run(
            memory["memory_id"],
            evidence_run_id=unrelated_run,
            reviewer="tester",
        )
    store.update_run(
        evidence_run,
        status="passed",
        result={"ok": True, "execution_mode": "llm_freeform"},
        completed=True,
    )
    reconfirmed = repair.reconfirm_from_passed_run(
        memory["memory_id"],
        evidence_run_id=evidence_run,
        reviewer="tester",
        note="Controlled reproduction passed.",
    )
    duplicate = repair.reconfirm_from_passed_run(
        memory["memory_id"],
        evidence_run_id=evidence_run,
        reviewer="tester",
    )

    assert reconfirmed is not None and duplicate is not None
    assert reconfirmed["metadata"]["verification_count"] == 2
    assert duplicate["metadata"]["verification_count"] == 2
    assert reconfirmed["metadata"]["last_reconfirmation_source"] == "explicit_api_reconfirmation"
    retrieved = repair.retrieve(
        workspace_id="semantic_workspace",
        execution_mode="llm_freeform",
        query_text="managed process timeout",
    )
    assert [item["memory_id"] for item in retrieved] == [memory["memory_id"]]


def test_reconfirm_api_requires_passed_evidence_run(tmp_path, monkeypatch) -> None:
    store = YieldMindStore(tmp_path / "yieldmind.sqlite3")
    repair = RepairMemoryStore(store)
    memory = _confirmed_memory(
        repair,
        store,
        error="ImportError: estimator API changed",
        repair_summary="Use the pinned estimator API.",
    )
    evidence_run = store.create_run(
        mode="live_llm",
        source="yield_domain_stategraph",
        status="running",
        metadata={"workspace_id": "semantic_workspace"},
    )
    monkeypatch.setattr(yieldmind_api, "store", store)
    client = TestClient(app)

    rejected = client.post(
        f"/api/memories/{memory['memory_id']}/reconfirm",
        json={"evidence_run_id": evidence_run, "reviewer": "reviewer_a"},
    )
    store.update_run(
        evidence_run,
        status="passed",
        result={"ok": True, "execution_mode": "llm_freeform"},
        completed=True,
    )
    accepted = client.post(
        f"/api/memories/{memory['memory_id']}/reconfirm",
        json={"evidence_run_id": evidence_run, "reviewer": "reviewer_a", "note": "Reproduced."},
    )

    assert rejected.status_code == 400
    assert accepted.status_code == 200
    assert accepted.json()["memory"]["metadata"]["verification_count"] == 2
