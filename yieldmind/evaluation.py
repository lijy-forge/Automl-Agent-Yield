"""Offline evaluation harness for the YieldMind upgrade layer."""

from __future__ import annotations

import json
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from yieldmind.database import YieldMindStore, json_dumps
from yieldmind.function_calling import ToolPlanRequest, local_rule_plan
from yieldmind.knowledge_base import KnowledgeBase, KnowledgeIngestRequest, KnowledgeSearchRequest, evaluate_retrieval
from yieldmind.memory import AddMessageRequest, CreateSessionRequest, SessionMemoryStore, UpsertMemoryRequest
from yieldmind.tools import OFFLINE_MODE, ToolRegistry, registry_for_workspace
from yieldmind.workflow import WorkflowRequest, run_workflow


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEED_KNOWLEDGE_PATH = PROJECT_ROOT / "knowledge_sources" / "yieldmind_domain_seed.md"
EVAL_DATASET_PATH = PROJECT_ROOT / "evals" / "yieldmind_offline_cases.json"


def _case(name: str, tool: str, args: dict[str, Any], expect_ok: bool = True) -> dict[str, Any]:
    return {"name": name, "tool": tool, "args": args, "expect_ok": expect_ok}


def default_cases(python_executable: str) -> list[dict[str, Any]]:
    demo_path = str(PROJECT_ROOT / "agent_workspace" / "data" / "yield_synthetic" / "synthetic_yield_v1.csv")
    return [
        _case("generate_demo_data", "generate_demo_yield_data", {"n_samples": 60, "random_state": 42, "overwrite": True}),
        _case("profile_demo_data", "profile_yield_data", {"data_path": demo_path}),
        _case("baseline_demo_data", "run_fixed_baseline_eval", {"data_path": demo_path, "n_splits": 3, "random_state": 42}),
        _case("candidate_benchmark_demo_data", "run_candidate_benchmark", {"data_path": demo_path, "n_splits": 3, "random_state": 42}),
        _case(
            "sandbox_python_echo",
            "run_sandboxed_python",
            {"argv": [python_executable, "-c", "print('yieldmind-eval-ok')"], "timeout_seconds": 10},
        ),
        _case(
            "docker_sandbox_requires_authorization",
            "run_docker_sandboxed_python",
            {"argv": ["python", "-c", "print('must-not-run')"]},
            expect_ok=False,
        ),
        _case("missing_data_rejected", "profile_yield_data", {"data_path": "agent_workspace/does_not_exist.csv"}, expect_ok=False),
    ]


def _case_passed(result_ok: bool, expect_ok: bool, result: dict[str, Any]) -> bool:
    if result_ok != expect_ok:
        return False
    if result_ok and result.get("simulated_model_calls", 0) != 0:
        return False
    return True


def default_planner_cases() -> list[dict[str, Any]]:
    return [
        {
            "name": "baseline_prompt_routes_to_baseline",
            "prompt": "请对 demo 数据做 baseline 评测",
            "expected_tools": ["generate_demo_yield_data", "run_fixed_baseline_eval"],
        },
        {
            "name": "profile_prompt_routes_to_profile",
            "prompt": "读取数据并做 schema/profile 画像",
            "expected_tools": ["generate_demo_yield_data", "profile_yield_data"],
        },
        {
            "name": "sandbox_prompt_routes_to_sandbox",
            "prompt": "检查工具执行隔离 sandbox",
            "expected_tools": ["generate_demo_yield_data", "run_sandboxed_python"],
        },
        {
            "name": "knowledge_prompt_routes_to_search",
            "prompt": "检索文献证据并给出引用",
            "expected_tools": ["generate_demo_yield_data", "search_knowledge"],
        },
        {
            "name": "candidate_prompt_routes_to_benchmark",
            "prompt": "对候选机理和模型策略做 benchmark 方案评估",
            "expected_tools": ["generate_demo_yield_data", "run_candidate_benchmark"],
        },
    ]


def load_eval_dataset(path: str | Path | None = None) -> dict[str, Any]:
    dataset_path = Path(path or EVAL_DATASET_PATH)
    if not dataset_path.exists():
        return {
            "version": "builtin-default",
            "planner_cases": default_planner_cases(),
            "retrieval_cases": [
                {"query": "YODEL maximum packing density yield stress", "expected_terms": ["yodel", "packing"]},
                {"query": "mechanism ablation evidence chunk", "expected_terms": ["mechanism", "ablation"]},
            ],
            "redaction_cases": [],
            "budget_cases": [],
        }
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Eval dataset must be a JSON object: {dataset_path}")
    payload.setdefault("version", dataset_path.name)
    payload.setdefault("planner_cases", default_planner_cases())
    payload.setdefault("retrieval_cases", [])
    payload.setdefault("redaction_cases", [])
    payload.setdefault("budget_cases", [])
    return payload


def run_offline_evaluation(
    *,
    registry: ToolRegistry | None = None,
    store: YieldMindStore | None = None,
    out_dir: str | Path | None = None,
    python_executable: str,
) -> dict[str, Any]:
    started = time.time()
    out_root = Path(out_dir or PROJECT_ROOT / "agent_workspace" / "yieldmind" / "evals")
    out_root.mkdir(parents=True, exist_ok=True)
    store = store or YieldMindStore()
    registry = registry or registry_for_workspace(store=store)
    dataset = load_eval_dataset()
    run_id = store.create_run(mode=OFFLINE_MODE, source="yieldmind_offline_eval", status="running")
    store.add_event(
        run_id,
        stage="eval",
        message="Offline YieldMind evaluation started.",
        payload={"eval_dataset_version": dataset.get("version"), "eval_dataset_path": str(EVAL_DATASET_PATH)},
    )

    rows: list[dict[str, Any]] = []
    for item in default_cases(python_executable):
        before = time.time()
        result = registry.execute(item["tool"], item["args"], run_id=run_id)
        row = {
            "case": item["name"],
            "tool": item["tool"],
            "expect_ok": item["expect_ok"],
            "ok": result.ok,
            "passed": _case_passed(result.ok, item["expect_ok"], result.model_dump()),
            "duration_seconds": round(time.time() - before, 4),
            "error": result.error,
            "artifacts": result.artifacts,
            "llm_calls": result.llm_calls,
            "simulated_model_calls": result.simulated_model_calls,
        }
        rows.append(row)
        store.add_event(
            run_id,
            stage="eval_case",
            level="info" if row["passed"] else "error",
            message=f"{item['name']}: {'passed' if row['passed'] else 'failed'}",
            payload=row,
        )

    planner_rows: list[dict[str, Any]] = []
    for item in dataset.get("planner_cases") or default_planner_cases():
        plan = local_rule_plan(ToolPlanRequest(prompt=item["prompt"], max_tool_calls=int(item.get("max_tool_calls", 6))))
        actual = {call.tool_name for call in plan.tool_calls}
        expected = set(item["expected_tools"])
        row = {
            "case": item["name"],
            "prompt": item["prompt"],
            "expected_tools": sorted(expected),
            "actual_tools": sorted(actual),
            "passed": expected.issubset(actual),
            "mode": plan.mode,
            "llm_calls": plan.llm_calls,
            "simulated_model_calls": plan.simulated_model_calls,
        }
        planner_rows.append(row)
        store.add_event(
            run_id,
            stage="planner_case",
            level="info" if row["passed"] else "error",
            message=f"{item['name']}: {'passed' if row['passed'] else 'failed'}",
            payload=row,
        )

    workflow_started = time.time()
    workflow_result = run_workflow(
        WorkflowRequest(prompt="Run deterministic demo baseline workflow.", n_samples=60, n_splits=3),
        store=store,
    )
    workflow_row = {
        "case": "offline_workflow_demo_baseline",
        "passed": workflow_result.get("status") == "passed",
        "status": workflow_result.get("status"),
        "workflow_backend": workflow_result.get("workflow_backend"),
        "langgraph_available": workflow_result.get("langgraph_available"),
        "duration_seconds": round(time.time() - workflow_started, 4),
        "run_id": workflow_result.get("run_id"),
        "artifact_keys": sorted((workflow_result.get("artifacts") or {}).keys()),
    }
    store.add_event(
        run_id,
        stage="workflow_case",
        level="info" if workflow_row["passed"] else "error",
        message=f"offline_workflow_demo_baseline: {'passed' if workflow_row['passed'] else 'failed'}",
        payload=workflow_row,
    )

    knowledge_started = time.time()
    eval_chroma_parent = PROJECT_ROOT / "agent_workspace" / "yieldmind" / "eval_chroma_tmp"
    eval_chroma_parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="run_", dir=eval_chroma_parent) as eval_chroma_dir:
        kb = KnowledgeBase(store=store, chroma_dir=eval_chroma_dir, collection_name="yieldmind_offline_eval")
        ingest_result = kb.ingest(KnowledgeIngestRequest(paths=[str(SEED_KNOWLEDGE_PATH)]))
        search_result = kb.search(KnowledgeSearchRequest(query="YODEL packing yield stress phi_m", top_k=3))
        retrieval_cases = dataset.get("retrieval_cases") or [
            {"query": "YODEL maximum packing density yield stress", "expected_terms": ["yodel", "packing"]},
            {"query": "mechanism ablation evidence chunk", "expected_terms": ["mechanism", "ablation"]},
        ]
        retrieval_eval = evaluate_retrieval(
            kb,
            retrieval_cases,
            top_k=3,
        )
    retrieval_passed = sum(1 for item in retrieval_eval.get("cases", []) if item.get("passed"))
    knowledge_smoke_passed = bool(search_result.get("hits"))
    knowledge_row = {
        "case": "knowledge_ingest_search",
        "passed": knowledge_smoke_passed and retrieval_passed == len(retrieval_eval.get("cases", [])),
        "duration_seconds": round(time.time() - knowledge_started, 4),
        "ingested_documents": len(ingest_result.get("documents", [])),
        "hit_count": len(search_result.get("hits", [])),
        "retrieval_eval": retrieval_eval,
        "retrieval_passed_cases": retrieval_passed,
        "embedding_model": search_result.get("embedding_model"),
    }
    store.add_event(
        run_id,
        stage="knowledge_case",
        level="info" if knowledge_row["passed"] else "error",
        message=f"knowledge_ingest_search: {'passed' if knowledge_row['passed'] else 'failed'}",
        payload=knowledge_row,
    )

    safety_started = time.time()
    redaction_result = registry.execute(
        "redact_sensitive_payload",
        {
            "payload": {
                "Authorization": "Bearer sk-eval1234567890",
                "contact": "owner@example.com",
                "database_url": "postgresql://user:pass@localhost:5432/yieldmind",
            }
        },
        run_id=run_id,
    )
    redaction_rendered = json_dumps(redaction_result.result)
    budget_result = registry.execute(
        "plan_token_budget",
        {
            "max_input_tokens": 140,
            "reserved_output_tokens": 20,
            "sections": [
                {"name": "constraints", "text": "target_column=tau0", "tokens": 50, "required": True, "priority": 100},
                {"name": "evidence", "text": "retrieved evidence chunks", "tokens": 60, "priority": 80},
                {"name": "old_history", "text": "older conversation turns", "tokens": 90, "priority": 5},
            ],
        },
        run_id=run_id,
    )
    first_hit = (search_result.get("hits") or [{}])[0]
    valid_evidence_result = None
    invalid_evidence_result = None
    if first_hit.get("chunk_id"):
        valid_evidence_result = registry.execute(
            "validate_evidence_refs",
            {
                "refs": [
                    {
                        "chunk_id": first_hit["chunk_id"],
                        "text_hash": first_hit.get("text_hash", ""),
                        "index_version": first_hit.get("index_version", ""),
                    }
                ]
            },
            run_id=run_id,
        )
        invalid_evidence_result = registry.execute(
            "validate_evidence_refs",
            {
                "refs": [
                    {
                        "chunk_id": first_hit["chunk_id"],
                        "text_hash": "wrong-hash",
                        "index_version": first_hit.get("index_version", ""),
                    }
                ]
            },
            run_id=run_id,
        )
    selected_budget = {item.get("name") for item in budget_result.result.get("selected_sections", [])}
    dropped_budget = {item.get("name") for item in budget_result.result.get("dropped_sections", [])}
    safety_row = {
        "case": "safety_evidence_budget",
        "passed": (
            redaction_result.ok
            and "sk-eval1234567890" not in redaction_rendered
            and "owner@example.com" not in redaction_rendered
            and "postgresql://user:pass" not in redaction_rendered
            and budget_result.ok
            and {"constraints", "evidence"}.issubset(selected_budget)
            and "old_history" in dropped_budget
            and valid_evidence_result is not None
            and valid_evidence_result.ok
            and invalid_evidence_result is not None
            and not invalid_evidence_result.ok
        ),
        "duration_seconds": round(time.time() - safety_started, 4),
        "redaction_count": redaction_result.result.get("redaction_count"),
        "budget_selected": sorted(str(item) for item in selected_budget),
        "budget_dropped": sorted(str(item) for item in dropped_budget),
        "evidence_valid_count": (valid_evidence_result.result.get("valid_count") if valid_evidence_result else 0),
        "evidence_invalid_count": (invalid_evidence_result.result.get("invalid_count") if invalid_evidence_result else 0),
        "llm_calls": 0,
        "simulated_model_calls": 0,
    }
    store.add_event(
        run_id,
        stage="safety_case",
        level="info" if safety_row["passed"] else "error",
        message=f"safety_evidence_budget: {'passed' if safety_row['passed'] else 'failed'}",
        payload=safety_row,
    )

    redaction_rows: list[dict[str, Any]] = []
    for item in dataset.get("redaction_cases", []):
        before = time.time()
        result = registry.execute("redact_sensitive_payload", {"payload": item.get("payload", {})}, run_id=run_id)
        rendered = json_dumps(result.result)
        forbidden_terms = [str(term) for term in item.get("forbidden_terms", [])]
        missing_forbidden_terms = [term for term in forbidden_terms if term in rendered]
        min_redactions = int(item.get("min_redactions", 0))
        row = {
            "case": item.get("name", "redaction_case"),
            "tool": "redact_sensitive_payload",
            "passed": result.ok
            and not missing_forbidden_terms
            and int(result.result.get("redaction_count") or 0) >= min_redactions,
            "duration_seconds": round(time.time() - before, 4),
            "redaction_count": result.result.get("redaction_count"),
            "min_redactions": min_redactions,
            "forbidden_term_leaks": missing_forbidden_terms,
            "llm_calls": result.llm_calls,
            "simulated_model_calls": result.simulated_model_calls,
        }
        redaction_rows.append(row)
        store.add_event(
            run_id,
            stage="redaction_case",
            level="info" if row["passed"] else "error",
            message=f"{row['case']}: {'passed' if row['passed'] else 'failed'}",
            payload=row,
        )

    budget_rows: list[dict[str, Any]] = []
    for item in dataset.get("budget_cases", []):
        before = time.time()
        result = registry.execute("plan_token_budget", item.get("request", {}), run_id=run_id)
        selected = {section.get("name") for section in result.result.get("selected_sections", [])}
        dropped = {section.get("name") for section in result.result.get("dropped_sections", [])}
        clipped = {section.get("name") for section in result.result.get("selected_sections", []) if section.get("clipped")}
        expected_selected = set(item.get("expected_selected", []))
        expected_dropped = set(item.get("expected_dropped", []))
        expected_clipped = set(item.get("expected_clipped", []))
        expected_over_budget = item.get("expected_over_budget")
        over_budget_ok = True if expected_over_budget is None else bool(result.result.get("over_budget")) is bool(expected_over_budget)
        expect_ok = bool(item.get("expect_ok", True))
        row = {
            "case": item.get("name", "budget_case"),
            "tool": "plan_token_budget",
            "expect_ok": expect_ok,
            "ok": result.ok,
            "passed": (
                result.ok is expect_ok
                and expected_selected.issubset(selected)
                and expected_dropped.issubset(dropped)
                and expected_clipped.issubset(clipped)
                and over_budget_ok
            ),
            "duration_seconds": round(time.time() - before, 4),
            "selected": sorted(str(name) for name in selected),
            "dropped": sorted(str(name) for name in dropped),
            "clipped": sorted(str(name) for name in clipped),
            "over_budget": result.result.get("over_budget"),
            "llm_calls": result.llm_calls,
            "simulated_model_calls": result.simulated_model_calls,
        }
        budget_rows.append(row)
        store.add_event(
            run_id,
            stage="budget_case",
            level="info" if row["passed"] else "error",
            message=f"{row['case']}: {'passed' if row['passed'] else 'failed'}",
            payload=row,
        )

    memory_started = time.time()
    memory_store = SessionMemoryStore(store)
    session = memory_store.create_session(CreateSessionRequest(constraints={"target_column": "yield_stress"}))
    session_id = session["session_id"]
    turn1 = memory_store.add_message(
        AddMessageRequest(session_id=session_id, content="目标列改成 tau0，请不要重新训练", idempotency_key="eval-turn-1")
    )
    turn2 = memory_store.add_message(
        AddMessageRequest(session_id=session_id, content="把搜索预算调小，再运行一次", idempotency_key="eval-turn-2")
    )
    mem = memory_store.upsert_memory(
        UpsertMemoryRequest(
            session_id=session_id,
            content="用户确认偏好小搜索预算",
            source_ref=turn2["turn"]["turn_id"],
            validation_status="confirmed",
        )
    )
    context_before_delete = memory_store.build_context(session_id, max_tokens=600)
    deleted = memory_store.delete_memory(type("Req", (), {"memory_id": mem["memory_id"]})())
    context_after_delete = memory_store.build_context(session_id, max_tokens=600)
    current_constraints = memory_store.get_session(session_id).get("constraints", {})
    memory_row = {
        "case": "session_memory_constraints",
        "passed": (
            turn1.get("action") == "answer_only"
            and turn2.get("action") == "create_run"
            and str(turn2.get("run_id", "")).startswith("run_")
            and current_constraints.get("target_column") == "tau0"
            and current_constraints.get("search_budget") == "small"
            and deleted.get("ok") is True
        ),
        "duration_seconds": round(time.time() - memory_started, 4),
        "session_id": session_id,
        "run_id": turn2.get("run_id"),
        "context_tokens_before_delete": context_before_delete.get("estimated_tokens"),
        "context_tokens_after_delete": context_after_delete.get("estimated_tokens"),
    }
    store.add_event(
        run_id,
        stage="memory_case",
        level="info" if memory_row["passed"] else "error",
        message=f"session_memory_constraints: {'passed' if memory_row['passed'] else 'failed'}",
        payload=memory_row,
    )

    passed = sum(1 for row in rows if row["passed"])
    planner_passed = sum(1 for row in planner_rows if row["passed"])
    workflow_passed = 1 if workflow_row["passed"] else 0
    knowledge_case_count = 1 + len(retrieval_eval.get("cases", []))
    knowledge_passed = (1 if knowledge_smoke_passed else 0) + retrieval_passed
    safety_case_count = 1 + len(redaction_rows) + len(budget_rows)
    safety_passed = (
        (1 if safety_row["passed"] else 0)
        + sum(1 for row in redaction_rows if row["passed"])
        + sum(1 for row in budget_rows if row["passed"])
    )
    memory_passed = 1 if memory_row["passed"] else 0
    total_cases = len(rows) + len(planner_rows) + 1 + knowledge_case_count + safety_case_count + 1
    total_passed = passed + planner_passed + workflow_passed + knowledge_passed + safety_passed + memory_passed
    summary = {
        "mode": OFFLINE_MODE,
        "eval_dataset_version": dataset.get("version"),
        "eval_dataset_path": str(EVAL_DATASET_PATH),
        "total_cases": total_cases,
        "passed_cases": total_passed,
        "failed_cases": total_cases - total_passed,
        "tool_cases": len(rows),
        "planner_cases": len(planner_rows),
        "workflow_cases": 1,
        "knowledge_cases": knowledge_case_count,
        "safety_cases": safety_case_count,
        "memory_cases": 1,
        "pass_rate": total_passed / total_cases if total_cases else 0.0,
        "real_llm_calls": 0,
        "simulated_model_calls": 0,
        "model_call_policy": "No LLM/model API calls are made in this offline evaluation; planner cases use deterministic local rules.",
        "langgraph_available": workflow_result.get("langgraph_available"),
        "workflow_backend": workflow_result.get("workflow_backend"),
        "knowledge_embedding_model": knowledge_row.get("embedding_model"),
        "started_at": started,
        "completed_at": time.time(),
    }
    report = {
        "summary": summary,
        "cases": rows,
        "planner_cases": planner_rows,
        "workflow_case": workflow_row,
        "knowledge_case": knowledge_row,
        "safety_case": safety_row,
        "redaction_cases": redaction_rows,
        "budget_cases": budget_rows,
        "memory_case": memory_row,
    }
    report_path = out_root / f"yieldmind_offline_eval_{time.strftime('%Y%m%d_%H%M%S')}.json"
    report_path.write_text(json_dumps(report), encoding="utf-8")
    eval_id = store.create_eval_run(mode=OFFLINE_MODE, summary=summary, report_path=str(report_path), started_at=started)
    summary["eval_id"] = eval_id
    summary["report_path"] = str(report_path)
    store.update_run(run_id, status="passed" if total_passed == total_cases else "failed", result=summary, completed=True)
    return {"summary": summary, "cases": rows}


def report_to_text(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    lines = [
        "YieldMind offline evaluation",
        f"mode: {summary.get('mode')}",
        f"passed: {summary.get('passed_cases')}/{summary.get('total_cases')}",
        (
            "breakdown: "
            f"tools={summary.get('tool_cases')}, "
            f"planner={summary.get('planner_cases')}, "
            f"workflow={summary.get('workflow_cases')}, "
            f"knowledge={summary.get('knowledge_cases')}, "
            f"safety={summary.get('safety_cases')}, "
            f"memory={summary.get('memory_cases')}"
        ),
        f"real_llm_calls: {summary.get('real_llm_calls')}",
        f"simulated_model_calls: {summary.get('simulated_model_calls')}",
        f"report_path: {summary.get('report_path')}",
    ]
    for row in report.get("cases", []):
        marker = "PASS" if row.get("passed") else "FAIL"
        lines.append(f"- {marker} {row.get('case')} ({row.get('tool')})")
        if row.get("error"):
            lines.append(f"  error: {row.get('error')}")
    failure_groups = [
        ("planner", report.get("planner_cases", [])),
        ("redaction", report.get("redaction_cases", [])),
        ("budget", report.get("budget_cases", [])),
    ]
    for group_name, group_rows in failure_groups:
        for row in group_rows:
            if not row.get("passed"):
                lines.append(f"- FAIL {group_name}:{row.get('case')}")
    knowledge_case = report.get("knowledge_case") or {}
    retrieval_cases = ((knowledge_case.get("retrieval_eval") or {}).get("cases") or [])
    for row in retrieval_cases:
        if not row.get("passed"):
            lines.append(f"- FAIL retrieval:{row.get('query')}")
    return "\n".join(lines)
