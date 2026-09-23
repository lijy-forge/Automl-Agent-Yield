"""LangGraph orchestration for the original multi-agent yield workflow.

The graph state contains only serializable configuration, summaries, and
artifact paths. The original domain methods remain the source of truth and are
rehydrated from their JSON artifacts at each node boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Protocol, TypedDict

from pydantic import BaseModel, Field

from knowledge.yield_schema import DEFAULT_LIAN_DATA_PATH, DEFAULT_LIAN_TEST_PATH
from yieldmind.checkpointing import psycopg_connection_string
from yieldmind.database import YieldMindStore
from yieldmind.domain_retrieval import build_domain_knowledge_queries, retrieve_domain_knowledge
from yieldmind.knowledge_runtime import configured_embedding, configured_knowledge_base
from yieldmind.memory import SessionMemoryStore, render_session_context, session_context_summary
from yieldmind.model_usage import merge_model_usage_records, summarize_model_usage
from yieldmind.process_control import ManagedProcessOutcome, run_managed_process
from yieldmind.repair_memory import RepairMemoryStore, normalized_error_signature, operation_error_text


OPERATION_CONTRACT_VERSION = "yieldmind-operation-contract-v1"


class DomainWorkflowRequest(BaseModel):
    prompt: str = "Build an AutoML model for high-solid-content slurry yield-stress prediction."
    data_path: str = Field(default_factory=lambda: os.environ.get("YIELD_DATA_PATH", DEFAULT_LIAN_DATA_PATH))
    test_path: str = Field(default_factory=lambda: os.environ.get("YIELD_TEST_PATH", DEFAULT_LIAN_TEST_PATH))
    run_dir: str = ""
    llm: str = "openai"
    n_revise: int = Field(default=1, ge=0, le=5)
    operation_attempts: int = Field(default=3, ge=1, le=10)
    random_state: int = Field(default=42, ge=0)
    external_search: bool = True
    require_search_results: bool = True
    synthetic_data: bool | None = None
    synthetic_n: int = Field(default=800, ge=100, le=10000)
    query: list[str] = Field(default_factory=list)
    execution_mode: str = Field(default="free_search", pattern=r"^(free_search|plugin|llm_freeform)$")
    champion_policy: str = Field(default="mixed", pattern=r"^(mixed|llm_only)$")
    operation_timeout_seconds: float = Field(default=900.0, ge=1.0, le=86400.0)
    allow_live_llm: bool = False
    workspace_id: str = Field(default="default", min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    session_id: str = ""
    turn_id: str = ""
    session_context_max_tokens: int = Field(default=1200, ge=200, le=8000)
    use_knowledge_search: bool = True
    knowledge_top_k: int = Field(default=5, ge=1, le=20)
    knowledge_context_budget_chars: int = Field(default=12000, ge=500, le=50000)


DOMAIN_STAGE_AGENT_MAP = {
    "start": "Agent Manager",
    "prepare": "Agent Manager",
    "data": "DataAgent",
    "requirements": "Agent Manager",
    "search": "SearchAgent",
    "candidate": "CandidateAgent",
    "model": "ModelAgent",
    "pre_execution": "Agent Manager",
    "operation": "OperationAgent",
    "review": "Agent Manager",
    "revision_feedback": "Agent Manager",
    "cancel": "Agent Manager",
    "finish": "Agent Manager",
}

DOMAIN_STAGE_ACTION_MAP = {
    "start": "initialize_domain_run",
    "prepare": "prepare_domain_workspace",
    "data": "inspect_and_prepare_data",
    "requirements": "build_requirement_context",
    "search": "retrieve_routed_knowledge_and_external_evidence",
    "candidate": "propose_candidate_strategy",
    "model": "build_model_plan",
    "pre_execution": "approve_operation_contract",
    "operation": "execute_modeling_operation",
    "review": "review_operation_result",
    "revision_feedback": "build_revision_feedback",
    "cancel": "cancel_domain_workflow",
    "finish": "finalize_domain_run",
}

DOMAIN_STAGE_RUNNING_MESSAGES = {
    "start": "Agent Manager is creating the domain run.",
    "prepare": "Agent Manager is preparing the domain workspace.",
    "data": "DataAgent is validating and profiling the modeling data.",
    "requirements": "Agent Manager is building the requirement context.",
    "search": "SearchAgent is collecting domain evidence.",
    "candidate": "CandidateAgent is proposing a candidate strategy.",
    "model": "ModelAgent is building the model plan.",
    "pre_execution": "Agent Manager is reviewing the operation contract.",
    "operation": "OperationAgent is executing the approved modeling plan.",
    "review": "Agent Manager is reviewing the operation result.",
    "revision_feedback": "Agent Manager is preparing bounded revision feedback.",
    "cancel": "Agent Manager is cancelling the domain workflow.",
    "finish": "Agent Manager is finalizing the domain run.",
}


class ManagerRouteDecision(BaseModel):
    """Serializable, auditable routing output produced by the Manager layer."""

    decision_id: str
    after_stage: str
    next_node: str
    next_agent: str
    reason_code: str
    reason: str
    feedback: str = ""
    remaining_revision_budget: int = 0
    allowed_next_nodes: list[str] = Field(default_factory=list)
    validated: bool = True
    ts: float


class DomainManagerRouter:
    """Deterministic control-plane validation around Manager routing decisions."""

    LEGAL_TRANSITIONS = {
        "start": ("prepare", "cancel", "finish"),
        "prepare": ("data", "cancel", "finish"),
        "data": ("requirements", "cancel", "finish"),
        "requirements": ("search", "cancel", "finish"),
        "search": ("candidate", "cancel", "finish"),
        "candidate": ("model", "cancel", "finish"),
        "model": ("pre_execution", "cancel", "finish"),
        "pre_execution": ("operation", "review", "cancel", "finish"),
        "operation": ("review", "cancel", "finish"),
        "review": ("revision_feedback", "cancel", "finish"),
        "revision_feedback": ("candidate", "cancel", "finish"),
        "cancel": ("finish",),
    }

    SUCCESS_TARGETS = {
        "start": "prepare",
        "prepare": "data",
        "data": "requirements",
        "requirements": "search",
        "search": "candidate",
        "candidate": "model",
        "model": "pre_execution",
        "revision_feedback": "candidate",
    }

    def decide(
        self,
        state: "DomainWorkflowState",
        *,
        after_stage: str,
        cancel_requested: bool,
    ) -> ManagerRouteDecision:
        allowed = list(self.LEGAL_TRANSITIONS[after_stage])
        remaining = max(
            0,
            int(state.get("max_manager_revisions", 0)) - int(state.get("manager_revision_round", 0)),
        )
        feedback = str(state.get("manager_feedback") or "")

        if after_stage == "cancel":
            next_node, reason_code, reason = "finish", "cancellation_finalized", "Cancellation was persisted."
        elif cancel_requested or state.get("cancelled"):
            next_node, reason_code, reason = "cancel", "cancel_requested", "A cancellation request is active."
        elif after_stage == "operation":
            next_node, reason_code, reason = "review", "operation_completed", "OperationAgent returned a result."
        elif not state.get("last_node_ok", True):
            next_node, reason_code, reason = "finish", "node_failed", str(
                state.get("last_error") or f"{after_stage} failed."
            )
        elif after_stage == "pre_execution":
            if state.get("pre_execution_passed"):
                next_node, reason_code, reason = (
                    "operation",
                    "pre_execution_approved",
                    "The operation contract passed Manager review.",
                )
            else:
                next_node, reason_code, reason = (
                    "review",
                    "pre_execution_blocked",
                    "The blocked operation result must be reviewed without executing OperationAgent.",
                )
        elif after_stage == "review":
            review = state.get("post_execution_review", {})
            if review.get("passed"):
                next_node, reason_code, reason = "finish", "review_accepted", "The result passed Manager review."
            elif remaining > 0:
                next_node, reason_code, reason = (
                    "revision_feedback",
                    "review_requires_revision",
                    "The result requires a bounded Candidate/Model revision.",
                )
            else:
                next_node, reason_code, reason = (
                    "finish",
                    "revision_budget_exhausted",
                    "The result was not accepted and no revision budget remains.",
                )
        else:
            next_node = self.SUCCESS_TARGETS[after_stage]
            reason_code = "phase_completed"
            reason = f"{after_stage} completed successfully."

        validated = next_node in allowed
        if not validated:
            raise ValueError(f"Illegal Manager route after {after_stage}: {next_node}; allowed={allowed}")
        return ManagerRouteDecision(
            decision_id=f"decision_{uuid.uuid4().hex[:12]}",
            after_stage=after_stage,
            next_node=next_node,
            next_agent=DOMAIN_STAGE_AGENT_MAP[next_node],
            reason_code=reason_code,
            reason=reason,
            feedback=feedback,
            remaining_revision_budget=remaining,
            allowed_next_nodes=allowed,
            validated=validated,
            ts=time.time(),
        )


class DomainWorkflowState(TypedDict, total=False):
    run_id: str
    thread_id: str
    status: str
    workflow_backend: str
    checkpoint_backend: str
    manager_args: dict[str, Any]
    allow_live_llm: bool
    execution_mode: str
    operation_timeout_seconds: float
    run_dir: str
    runtime_env: dict[str, str]
    runtime_contract: dict[str, Any]
    data_contract: dict[str, Any]
    data_lineage: dict[str, Any]
    anchor_audit: dict[str, Any]
    operation_event_log: str
    workspace_id: str
    session_id: str
    turn_id: str
    session_context: dict[str, Any]
    session_context_prompt: str
    session_context_summary: dict[str, Any]
    session_context_max_tokens: int
    repair_memories: list[dict[str, Any]]
    repair_memory_context: str
    repair_memory_retrieval: dict[str, Any]
    pending_repair_memory_ids: list[str]
    verified_repair_memory_ids: list[str]
    repair_memory_reuse_ids: list[str]
    repair_memory_reuse_round: int
    repair_memory_active_reuse_ids: list[str]
    reconfirmed_repair_memory_ids: list[str]
    use_knowledge_search: bool
    knowledge_top_k: int
    knowledge_context_budget_chars: int
    knowledge_search_report: dict[str, Any]
    artifacts: dict[str, str]
    stages: list[dict[str, Any]]
    route_history: list[dict[str, Any]]
    manager_decision: dict[str, Any]
    manager_decisions: list[dict[str, Any]]
    agent_activities: list[dict[str, Any]]
    errors: list[str]
    last_node_ok: bool
    last_error: str
    manager_revision_round: int
    max_manager_revisions: int
    manager_feedback: str
    pre_execution_passed: bool
    operation_result: dict[str, Any]
    post_execution_review: dict[str, Any]
    process_control: dict[str, Any]
    cancelled: bool
    real_llm_calls: int | None
    simulated_model_calls: int
    model_call_mode: str
    model_usage_records: list[dict[str, Any]]
    model_usage_summary: dict[str, Any]


class DomainWorkflowAdapter(Protocol):
    uses_live_llm: bool

    def prepare(self, state: DomainWorkflowState) -> dict[str, Any]: ...
    def data(self, state: DomainWorkflowState) -> dict[str, Any]: ...
    def requirements(self, state: DomainWorkflowState) -> dict[str, Any]: ...
    def search(self, state: DomainWorkflowState) -> dict[str, Any]: ...
    def candidate(self, state: DomainWorkflowState) -> dict[str, Any]: ...
    def model(self, state: DomainWorkflowState) -> dict[str, Any]: ...
    def pre_execution(self, state: DomainWorkflowState) -> dict[str, Any]: ...
    def operation(self, state: DomainWorkflowState) -> dict[str, Any]: ...
    def review(self, state: DomainWorkflowState) -> dict[str, Any]: ...
    def revision_feedback(self, state: DomainWorkflowState) -> dict[str, Any]: ...


_RUNTIME_ENV_KEYS = (
    "YIELD_RUN_DIR",
    "YIELD_DATA_PATH",
    "YIELD_TEST_PATH",
    "YIELD_ANCHOR_PATH",
    "YIELD_SYNTHETIC_REPORT_PATH",
    "YIELD_EXECUTION_MODE",
    "YIELD_CHAMPION_POLICY",
)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _data_contract_from_report(report: dict[str, Any]) -> dict[str, Any]:
    profile = dict(report.get("profile") or {})
    schema = dict(profile.get("schema") or {})
    lineage = dict(report.get("dataset_lineage") or {})
    features = sorted(str(item) for item in (schema.get("feature_columns") or []))
    signature = hashlib.sha256("\n".join(features).encode("utf-8")).hexdigest()[:24] if features else ""
    return {
        "dataset_schema": str(schema.get("source_schema") or lineage.get("source_schema") or ""),
        "dataset_role": str(lineage.get("dataset_role") or ""),
        "target_column": str(schema.get("target_column") or lineage.get("target_column") or ""),
        "feature_signature": signature,
        "feature_count": len(features),
    }


def _runtime_contract() -> dict[str, Any]:
    dependency_versions: dict[str, str] = {}
    for distribution in ("numpy", "pandas", "scikit-learn"):
        try:
            dependency_versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
    return {
        "python_version": platform.python_version(),
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "execution_backend": "managed_subprocess",
        "dependency_versions": dependency_versions,
        "operation_contract_version": OPERATION_CONTRACT_VERSION,
    }


def _artifact_map(run_dir: Path) -> dict[str, str]:
    candidates = {
        "manager_run_config": run_dir / "manager_run_config.json",
        "manager_trace": run_dir / "manager_trace.json",
        "data_agent_report": run_dir / "data_agent_report.json",
        "requirement_analysis": run_dir / "requirement_analysis.json",
        "search_report": run_dir / "search_report.json",
        "candidate_report": run_dir / "candidate_report.json",
        "candidate_benchmark_report": run_dir / "candidate_benchmark_report.json",
        "candidate_selection_audit": run_dir / "candidate_selection_audit.json",
        "model_plan": run_dir / "model_plan.json",
        "operation_instructions": run_dir / "operation_instructions.txt",
        "pre_execution_review": run_dir / "pre_execution_review.json",
        "operation_events": run_dir / "operation_events.jsonl",
        "run_result": run_dir / "run_result.json",
        "post_execution_review": run_dir / "post_execution_review.json",
        "manager_revision_notes": run_dir / "manager_revision_notes.json",
        "final_metrics": run_dir / "metrics" / "free_search_report.json",
        "predictions": run_dir / "metrics" / "predictions.csv",
        "recommendations": run_dir / "metrics" / "recommendations.md",
        "predict_script": run_dir / "predict.py",
        "champion_model": run_dir / "trained_models" / "champion.joblib",
    }
    for path in sorted((run_dir / "logs" / "models").glob("*.py")):
        candidates[f"generated_model_{path.stem}"] = path
    for path in sorted((run_dir / "logs" / "mechanisms").glob("*.py")):
        candidates[f"generated_mechanism_{path.stem}"] = path
    return {name: str(path.resolve()) for name, path in candidates.items() if path.exists()}


def _capture_runtime_env() -> dict[str, str]:
    return {key: os.environ[key] for key in _RUNTIME_ENV_KEYS if os.environ.get(key) is not None}


def _restore_runtime_env(state: DomainWorkflowState) -> None:
    for key in _RUNTIME_ENV_KEYS:
        os.environ.pop(key, None)
    for key, value in state.get("runtime_env", {}).items():
        if key in _RUNTIME_ENV_KEYS:
            os.environ[key] = str(value)
    os.environ["YIELD_EXECUTION_MODE"] = str(state.get("execution_mode", "free_search"))


def _manager_args(state: DomainWorkflowState) -> argparse.Namespace:
    return argparse.Namespace(**dict(state["manager_args"]))


def _hydrate_manager(state: DomainWorkflowState, *, cancel_check: Callable[[], bool] | None = None):
    from run_yield import YieldAgentManager

    _restore_runtime_env(state)
    manager = YieldAgentManager(_manager_args(state), cancel_check=cancel_check)
    manager.manager_revision_round = int(state.get("manager_revision_round", 0))
    run_dir = Path(state["run_dir"])
    manager.run_dir = run_dir

    data_report = _read_json(run_dir / "data_agent_report.json")
    manager.train_profile = data_report.get("profile", {}) if isinstance(data_report, dict) else {}
    manager.synthetic_info = data_report.get("synthetic_info", {}) if isinstance(data_report, dict) else {}
    manager.search_report = _read_json(run_dir / "search_report.json")
    manager.candidate_report = _read_json(run_dir / "candidate_report.json")
    manager.candidate_benchmark_report = _read_json(run_dir / "candidate_benchmark_report.json")
    manager.candidate_selection_audit = _read_json(run_dir / "candidate_selection_audit.json")
    manager.model_plan = _read_json(run_dir / "model_plan.json")
    trace = _read_json(run_dir / "manager_trace.json")
    manager.stage_records = trace.get("stages", []) if isinstance(trace.get("stages"), list) else []
    manager.state = str(trace.get("current_state") or manager.state)
    notes = _read_json(run_dir / "manager_revision_notes.json")
    manager.revision_notes = notes.get("revision_notes", []) if isinstance(notes.get("revision_notes"), list) else []
    instructions_path = run_dir / "operation_instructions.txt"
    if instructions_path.exists():
        manager.operation_instructions = instructions_path.read_text(encoding="utf-8")
    return manager


def _real_operation_worker(state: DomainWorkflowState) -> dict[str, Any]:
    event_log = str(state.get("operation_event_log") or "").strip()
    if event_log:
        os.environ["AMLA_EVENT_LOG"] = event_log
    manager = _hydrate_manager(state)
    result = manager._run_operation_agent(skip_pre_execution_review=True)
    result["manager_revision_round"] = manager.manager_revision_round
    result["operation_attempt_budget"] = max(1, int(manager.args.operation_attempts))
    manager._save_result(
        result,
        f"run_result_manager_round_{manager.manager_revision_round}.json",
        announce=False,
    )
    manager._save_result(result)
    return result


class RealYieldDomainAdapter:
    """Thin adapter over the original manager's domain implementations."""

    uses_live_llm = True

    def __init__(
        self,
        *,
        store: YieldMindStore | None = None,
        knowledge_base_factory: Callable[[], Any] | None = None,
        cancel_check: Callable[[], bool] | None = None,
        on_detail_event: Callable[[DomainWorkflowState, dict[str, Any]], None] | None = None,
    ) -> None:
        self.store = store or YieldMindStore()
        self.knowledge_base_factory = knowledge_base_factory
        self.cancel_check = cancel_check
        self.on_detail_event = on_detail_event
        self._event_offsets: dict[str, int] = {}

    @staticmethod
    def _detail_agent(payload: dict[str, Any]) -> str:
        sender = str(payload.get("sender") or "").lower()
        label = str(payload.get("label") or "").lower()
        if "candidate" in label:
            return "CandidateAgent"
        if "model" in label:
            return "ModelAgent"
        if sender == "manager":
            return "Agent Manager"
        if sender == "search":
            return "SearchAgent"
        if sender == "data":
            return "DataAgent"
        return "OperationAgent"

    def _drain_operation_events(self, state: DomainWorkflowState, event_path: Path) -> None:
        if self.on_detail_event is None or not event_path.exists():
            return
        key = str(event_path)
        offset = self._event_offsets.get(key, 0)
        try:
            with event_path.open("rb") as stream:
                stream.seek(offset)
                data = stream.read()
        except OSError:
            return
        boundary = data.rfind(b"\n")
        if boundary < 0:
            return
        complete = data[: boundary + 1]
        self._event_offsets[key] = offset + len(complete)
        for raw_line in complete.splitlines():
            try:
                payload = json.loads(raw_line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict):
                continue
            content = str(payload.get("content") or "").strip()
            if not content:
                continue
            match = re.search(r"attempt\s*#?(\d+)", content, flags=re.IGNORECASE)
            attempt = int(match.group(1)) + 1 if match else 1
            lowered = content.lower()
            detail_status = (
                "warning"
                if any(token in lowered for token in ("rejected", "failed", "error", "timeout"))
                else "success"
                if any(token in lowered for token in ("accepted", "saved", "completed", "picked champion"))
                else "info"
            )
            summary = " ".join(content.split())
            self.on_detail_event(
                state,
                {
                    "agent": self._detail_agent(payload),
                    "action": "operation_runtime_event",
                    "attempt": attempt,
                    "message": summary if len(summary) <= 360 else summary[:357] + "...",
                    "detail_status": detail_status,
                    "output_summary": {
                        "label": payload.get("label") or "Operation runtime",
                        "runtime_event": content,
                    },
                    "ts": float(payload.get("ts") or time.time()),
                },
            )

    def _observe_operation_events(
        self,
        state: DomainWorkflowState,
        event_path: Path,
        stop_event: threading.Event,
    ) -> None:
        while not stop_event.wait(0.25):
            self._drain_operation_events(state, event_path)
        self._drain_operation_events(state, event_path)

    def _emit_operation_process_detail(
        self,
        state: DomainWorkflowState,
        *,
        message: str,
        detail_status: str = "info",
        output_summary: dict[str, Any] | None = None,
    ) -> None:
        if self.on_detail_event is None:
            return
        self.on_detail_event(
            state,
            {
                "agent": "OperationAgent",
                "action": "operation_process_control",
                "attempt": int(state.get("manager_revision_round", 0)) + 1,
                "message": message,
                "detail_status": detail_status,
                "output_summary": output_summary or {},
                "ts": time.time(),
            },
        )

    def _result(self, state: DomainWorkflowState, **extra: Any) -> dict[str, Any]:
        run_dir = Path(state["run_dir"])
        return {
            "ok": True,
            "runtime_env": _capture_runtime_env(),
            "artifacts": _artifact_map(run_dir),
            **extra,
        }

    def prepare(self, state: DomainWorkflowState) -> dict[str, Any]:
        manager = _hydrate_manager(state, cancel_check=self.cancel_check)
        manager._prepare_run()
        return self._result(state)

    def data(self, state: DomainWorkflowState) -> dict[str, Any]:
        manager = _hydrate_manager(state, cancel_check=self.cancel_check)
        manager._run_data_agent()
        report = _read_json(Path(state["run_dir"]) / "data_agent_report.json")
        return self._result(
            state,
            data_lineage=report.get("dataset_lineage") or {},
            anchor_audit=report.get("anchor_audit") or {},
            data_contract=_data_contract_from_report(report),
        )

    def requirements(self, state: DomainWorkflowState) -> dict[str, Any]:
        manager = _hydrate_manager(state, cancel_check=self.cancel_check)
        manager._emit_requirement_context()
        return self._result(
            state,
            session_context_summary=dict(state.get("session_context_summary") or {}),
        )

    def search(self, state: DomainWorkflowState) -> dict[str, Any]:
        manager = _hydrate_manager(state, cancel_check=self.cancel_check)
        if state.get("use_knowledge_search", True):
            try:
                knowledge_base = (
                    self.knowledge_base_factory()
                    if self.knowledge_base_factory is not None
                    else configured_knowledge_base(self.store)
                )
                queries = build_domain_knowledge_queries(
                    str(manager.args.prompt or ""),
                    list(manager.args.query or []),
                    manager.train_profile,
                )
                knowledge_report = retrieve_domain_knowledge(
                    knowledge_base,
                    self.store,
                    queries=queries,
                    top_k=int(state.get("knowledge_top_k", 5)),
                    context_budget_chars=int(state.get("knowledge_context_budget_chars", 12000)),
                )
            except Exception as exc:
                knowledge_report = {
                    "status": "failed",
                    "snippets": [],
                    "errors": [f"{type(exc).__name__}: {exc}"],
                    "provider_summary": {},
                    "source_type_summary": {},
                    "evidence_validation": {
                        "ok": False,
                        "checked_count": 0,
                        "valid_count": 0,
                        "invalid_count": 0,
                    },
                    "real_llm_calls": 0,
                }
        else:
            knowledge_report = {
                "status": "disabled",
                "snippets": [],
                "errors": [],
                "provider_summary": {},
                "source_type_summary": {},
                "real_llm_calls": 0,
            }
        ok = manager._run_search_agent(knowledge_search_report=knowledge_report)
        return self._result(
            state,
            ok=bool(ok),
            error="" if ok else "Required knowledge/external search returned no validated evidence.",
            knowledge_search_report=knowledge_report,
        )

    def candidate(self, state: DomainWorkflowState) -> dict[str, Any]:
        manager = _hydrate_manager(state, cancel_check=self.cancel_check)
        manager._run_candidate_agent(
            manager_feedback=str(state.get("manager_feedback") or ""),
            repair_memory_context=str(state.get("repair_memory_context") or ""),
            session_context=str(state.get("session_context_prompt") or ""),
        )
        return self._result(state, model_usage_records=list(manager.model_usage_records))

    def model(self, state: DomainWorkflowState) -> dict[str, Any]:
        manager = _hydrate_manager(state, cancel_check=self.cancel_check)
        manager._run_model_agent(
            manager_feedback=str(state.get("manager_feedback") or ""),
            repair_memory_context=str(state.get("repair_memory_context") or ""),
            session_context=str(state.get("session_context_prompt") or ""),
        )
        return self._result(state, model_usage_records=list(manager.model_usage_records))

    def pre_execution(self, state: DomainWorkflowState) -> dict[str, Any]:
        manager = _hydrate_manager(state, cancel_check=self.cancel_check)
        manager._transition("PRE_EXEC", "AgentManager building operation contract and checking readiness.")
        manager.operation_instructions = manager._build_operation_instructions()
        repair_context = str(state.get("repair_memory_context") or "").strip()
        session_context = str(state.get("session_context_prompt") or "").strip()
        if session_context:
            manager.operation_instructions += (
                "\n\n# Bounded session context\n"
                + session_context
                + "\nCurrent explicit constraints override older remembered preferences."
            )
        if repair_context:
            manager.operation_instructions += (
                "\n\n# Verified cross-run repair checks\n"
                + repair_context
                + "\nApply only memories relevant to the current generated code and keep all current guardrails."
            )
        instructions_path = manager.run_dir / "operation_instructions.txt"
        instructions_path.write_text(manager.operation_instructions, encoding="utf-8")
        review = manager._pre_execution_review()
        blocked_result = {
            "rcode": 3,
            "stage": "pre_execution_review",
            "action_result": "AgentManager blocked OperationAgent because pre-execution review failed.",
            "code": "",
            "error_logs": [json.dumps(review, ensure_ascii=False)],
            "pre_execution_review": review,
        }
        return self._result(
            state,
            pre_execution_passed=bool(review.get("passed")),
            pre_execution_review=review,
            operation_result={} if review.get("passed") else blocked_result,
        )

    def operation(self, state: DomainWorkflowState) -> dict[str, Any]:
        event_path = Path(state["run_dir"]) / "operation_events.jsonl"
        event_path.parent.mkdir(parents=True, exist_ok=True)
        if str(event_path) not in self._event_offsets:
            event_path.write_text("", encoding="utf-8")
            self._event_offsets[str(event_path)] = 0
        state["operation_event_log"] = str(event_path)
        stop_event = threading.Event()
        observer = threading.Thread(
            target=self._observe_operation_events,
            args=(state, event_path, stop_event),
            name="yieldmind-operation-event-observer",
            daemon=True,
        )
        observer.start()
        self._emit_operation_process_detail(
            state,
            message="已启动隔离代码执行进程，正在生成、运行并验证模型代码。",
            output_summary={"phase": "process_started"},
        )
        try:
            outcome = run_managed_process(
                _real_operation_worker,
                (dict(state),),
                timeout_seconds=float(state.get("operation_timeout_seconds", 900.0)),
                cancel_check=self.cancel_check,
            )
        finally:
            stop_event.set()
            observer.join(timeout=2.0)
            self._drain_operation_events(state, event_path)
        outcome_label = {
            "completed": "执行结果已回传给 Agent Manager。",
            "timed_out": "隔离执行进程已超时并停止。",
            "cancelled": "隔离执行进程已取消并停止。",
        }.get(outcome.status, "隔离执行进程异常结束。")
        self._emit_operation_process_detail(
            state,
            message=f"{outcome_label} 耗时 {outcome.duration_seconds:.2f} 秒。",
            detail_status="success" if outcome.ok else "warning",
            output_summary={"phase": "result_returned", **outcome.metadata()},
        )
        if outcome.status == "cancelled":
            result = {
                "rcode": 130,
                "stage": "operation",
                "action_result": "Domain OperationAgent node was cancelled.",
                "error_logs": ["Managed OperationAgent process group was stopped after cancellation."],
                "cancelled": True,
                "process_control": outcome.metadata(),
            }
        elif outcome.status == "timed_out":
            result = {
                "rcode": 1,
                "stage": "operation",
                "action_result": "Domain OperationAgent node timed out.",
                "error_logs": ["Managed OperationAgent process group was stopped after timeout."],
                "timed_out": True,
                "process_control": outcome.metadata(),
            }
        elif outcome.ok and isinstance(outcome.value, dict):
            result = dict(outcome.value)
            result["process_control"] = outcome.metadata()
        else:
            result = {
                "rcode": 1,
                "stage": "operation",
                "action_result": outcome.error or "Managed OperationAgent process failed.",
                "error_logs": [outcome.error or "Managed OperationAgent process failed."],
                "process_control": outcome.metadata(),
            }
        try:
            result_rcode = int(result.get("rcode", 1))
        except (TypeError, ValueError):
            result_rcode = 1
        operation_ok = result_rcode == 0 and not result.get("cancelled") and not result.get("timed_out")
        return self._result(
            state,
            ok=operation_ok,
            error="" if operation_ok else str(result.get("action_result") or "OperationAgent failed."),
            operation_result=result,
            process_control=outcome.metadata(),
            cancelled=bool(result.get("cancelled")),
            model_usage_records=list(result.get("model_usage_records") or []),
        )

    def review(self, state: DomainWorkflowState) -> dict[str, Any]:
        manager = _hydrate_manager(state, cancel_check=self.cancel_check)
        result = state.get("operation_result", {})
        manager._save_result(result, announce=False)
        remaining = max(0, int(state.get("max_manager_revisions", 0)) - manager.manager_revision_round)
        review = manager._post_execution_review(result, remaining_revisions=remaining)
        return self._result(state, post_execution_review=review)

    def revision_feedback(self, state: DomainWorkflowState) -> dict[str, Any]:
        manager = _hydrate_manager(state, cancel_check=self.cancel_check)
        feedback = manager._build_manager_revision_feedback(
            state.get("operation_result", {}),
            state.get("post_execution_review", {}),
        )
        return self._result(state, manager_feedback=feedback)


class YieldDomainWorkflow:
    def __init__(
        self,
        *,
        store: YieldMindStore | None = None,
        adapter: DomainWorkflowAdapter | None = None,
        cancel_check: Callable[[], bool] | None = None,
        on_run_started: Callable[[str], None] | None = None,
        on_stage_progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.store = store or YieldMindStore()
        self.cancel_check = cancel_check
        self.on_run_started = on_run_started
        self.on_stage_progress = on_stage_progress
        self.manager_router = DomainManagerRouter()
        self.repair_memory_store = RepairMemoryStore(
            self.store,
            embedding_factory=configured_embedding,
        )
        self._activity_sequence = 0
        self._active_activities: dict[str, dict[str, Any]] = {}
        self._stage_attempts: dict[str, int] = {}
        self.adapter = adapter or RealYieldDomainAdapter(
            store=self.store,
            cancel_check=cancel_check,
            on_detail_event=self._record_operation_detail,
        )

    def _record_operation_detail(
        self,
        state: DomainWorkflowState,
        detail: dict[str, Any],
    ) -> None:
        now = float(detail.get("ts") or time.time())
        self._activity_sequence += 1
        event = {
            "type": "agent_progress",
            "schema_version": "agent-activity-detail-v1",
            "activity_id": f"detail_{uuid.uuid4().hex[:12]}",
            "sequence": self._activity_sequence,
            "workflow_kind": "domain",
            "agent": detail.get("agent") or "OperationAgent",
            "stage": "operation",
            "action": detail.get("action") or "operation_runtime_event",
            "attempt": int(detail.get("attempt") or 1),
            "status": "running",
            "detail": True,
            "detail_status": detail.get("detail_status") or "info",
            "message": str(detail.get("message") or "Operation runtime update."),
            "input_summary": {},
            "output_summary": detail.get("output_summary") or {},
            "run_id": state.get("run_id", ""),
            "thread_id": state.get("thread_id", ""),
            "started_at": now,
            "completed_at": now,
            "duration_ms": 0.0,
            "ts": now,
        }
        state.setdefault("agent_activities", []).append(event)
        self._emit(event)

    def _cancel_requested(self, state: DomainWorkflowState) -> bool:
        if state.get("cancelled"):
            return True
        return bool(self.cancel_check and self.cancel_check())

    def _refresh_repair_memories(
        self,
        state: DomainWorkflowState,
        *,
        query_text: str,
        phase: str,
        min_relevance: float = 0.0,
        strict_error_match: bool = False,
    ) -> None:
        memories = self.repair_memory_store.retrieve(
            workspace_id=str(state.get("workspace_id") or "default"),
            execution_mode=str(state.get("execution_mode") or "free_search"),
            query_text=query_text,
            min_relevance=min_relevance,
            applicability_context=self._repair_applicability_context(state),
            strict_error_match=strict_error_match,
        )
        state["repair_memories"] = memories
        state["repair_memory_context"] = self.repair_memory_store.prompt_context(memories)
        top_retrieval = dict((memories[0].get("retrieval") if memories else {}) or {})
        applicability_audit = dict(self.repair_memory_store.last_retrieval_audit or {})
        state["repair_memory_retrieval"] = {
            "phase": phase,
            "query_signature": normalized_error_signature(query_text),
            "selected_count": len(memories),
            "selected_memory_ids": [str(memory.get("memory_id") or "") for memory in memories],
            "ranking_method": top_retrieval.get("method", "recency_fallback"),
            "top_score": float(top_retrieval.get("score") or 0.0),
            "degraded_error": str(top_retrieval.get("degraded_error") or ""),
            "applicability_candidate_count": int(applicability_audit.get("candidate_count") or 0),
            "applicability_rejected_count": int(applicability_audit.get("rejected_count") or 0),
            "applicability_rejections": list(applicability_audit.get("rejected") or []),
        }

    @staticmethod
    def _repair_applicability_context(state: DomainWorkflowState) -> dict[str, Any]:
        return {
            **dict(state.get("runtime_contract") or {}),
            **dict(state.get("data_contract") or {}),
        }

    def _emit(self, event: dict[str, Any]) -> None:
        if self.on_stage_progress is None:
            return
        try:
            self.on_stage_progress(event)
        except Exception:
            # Observability callbacks must never change workflow behavior.
            pass

    def _begin_activity(self, state: DomainWorkflowState, stage: str) -> None:
        now = time.time()
        attempt = self._stage_attempts.get(stage, 0) + 1
        self._stage_attempts[stage] = attempt
        input_summary = {
            "manager_revision_round": int(state.get("manager_revision_round", 0)),
            "remaining_revision_budget": max(
                0,
                int(state.get("max_manager_revisions", 0))
                - int(state.get("manager_revision_round", 0)),
            ),
            "artifact_names": sorted(state.get("artifacts", {})),
            "has_manager_feedback": bool(state.get("manager_feedback")),
            "execution_mode": state.get("execution_mode", ""),
        }
        context_summary = dict(state.get("session_context_summary") or {})
        if context_summary:
            input_summary["session_context"] = context_summary
        if stage == "search":
            input_summary.update(
                {
                    "knowledge_search_enabled": bool(state.get("use_knowledge_search", True)),
                    "knowledge_top_k": int(state.get("knowledge_top_k", 5)),
                    "knowledge_context_budget_chars": int(
                        state.get("knowledge_context_budget_chars", 12000)
                    ),
                    "external_search_enabled": bool(
                        (state.get("manager_args") or {}).get("external_search", False)
                    ),
                }
            )
        if stage in {"candidate", "model", "pre_execution"}:
            repair_retrieval = dict(state.get("repair_memory_retrieval") or {})
            if repair_retrieval:
                input_summary["repair_memory_retrieval"] = repair_retrieval
        activity = {
            "activity_id": f"activity_{uuid.uuid4().hex[:12]}",
            "attempt": attempt,
            "started_at": now,
            "input_summary": input_summary,
        }
        self._active_activities[stage] = activity
        self._activity_sequence += 1
        event = {
            "type": "agent_progress",
            "schema_version": "agent-activity-v1",
            "activity_id": activity["activity_id"],
            "sequence": self._activity_sequence,
            "workflow_kind": "domain",
            "agent": DOMAIN_STAGE_AGENT_MAP[stage],
            "stage": stage,
            "action": DOMAIN_STAGE_ACTION_MAP[stage],
            "attempt": attempt,
            "status": "running",
            "message": DOMAIN_STAGE_RUNNING_MESSAGES[stage],
            "input_summary": input_summary,
            "run_id": state.get("run_id", ""),
            "thread_id": state.get("thread_id", ""),
            "started_at": now,
            "completed_at": None,
            "duration_ms": None,
            "ts": now,
        }
        state.setdefault("agent_activities", []).append(event)
        self._emit(event)

    def _finish_activity(
        self,
        state: DomainWorkflowState,
        stage: str,
        status: str,
        message: str,
        *,
        output_summary: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        now = time.time()
        activity = self._active_activities.pop(
            stage,
            {
                "activity_id": f"activity_{uuid.uuid4().hex[:12]}",
                "attempt": self._stage_attempts.get(stage, 1),
                "started_at": now,
            },
        )
        self._activity_sequence += 1
        event = {
            "type": "agent_progress",
            "schema_version": "agent-activity-v1",
            "activity_id": activity["activity_id"],
            "sequence": self._activity_sequence,
            "workflow_kind": "domain",
            "agent": DOMAIN_STAGE_AGENT_MAP[stage],
            "stage": stage,
            "action": DOMAIN_STAGE_ACTION_MAP[stage],
            "attempt": activity["attempt"],
            "status": status,
            "message": message,
            "input_summary": activity.get("input_summary", {}),
            "output_summary": output_summary or {},
            "run_id": state.get("run_id", ""),
            "thread_id": state.get("thread_id", ""),
            "started_at": activity["started_at"],
            "completed_at": now,
            "duration_ms": round((now - float(activity["started_at"])) * 1000.0, 3),
            "ts": now,
        }
        state.setdefault("agent_activities", []).append(event)
        self._emit(event)
        return event

    def _record_manager_decision(self, state: DomainWorkflowState, after_stage: str) -> None:
        decision = self.manager_router.decide(
            state,
            after_stage=after_stage,
            cancel_requested=self._cancel_requested(state),
        ).model_dump()
        state["manager_decision"] = decision
        state.setdefault("manager_decisions", []).append(decision)
        state.setdefault("route_history", []).append(
            {
                "route": decision["next_node"],
                "after_stage": after_stage,
                "reason_code": decision["reason_code"],
                "decision_id": decision["decision_id"],
                "validated": decision["validated"],
                "ts": decision["ts"],
            }
        )
        self._activity_sequence += 1
        event = {
            "type": "manager_decision",
            "schema_version": "manager-route-v1",
            "sequence": self._activity_sequence,
            "workflow_kind": "domain",
            "agent": "Agent Manager",
            "stage": after_stage,
            "action": "route_next_agent",
            "status": "passed",
            "message": f"Route {after_stage} -> {decision['next_node']}: {decision['reason']}",
            "decision": decision,
            "run_id": state.get("run_id", ""),
            "thread_id": state.get("thread_id", ""),
            "ts": decision["ts"],
        }
        state.setdefault("agent_activities", []).append(event)
        if state.get("run_id"):
            self.store.add_event(
                state["run_id"],
                stage="manager_decision",
                message=event["message"],
                payload=event,
            )
        self._emit(event)

    @staticmethod
    def _result_summary(stage: str, result: dict[str, Any]) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "ok": bool(result.get("ok", True)),
            "artifact_names": sorted((result.get("artifacts") or {}).keys()),
        }
        for key in (
            "pre_execution_passed",
            "cancelled",
            "real_llm_calls",
            "simulated_model_calls",
            "model_call_mode",
        ):
            if key in result:
                summary[key] = result[key]
        if result.get("model_usage_records") is not None:
            summary["model_usage"] = summarize_model_usage(result.get("model_usage_records") or [])
        if result.get("manager_feedback"):
            summary["manager_feedback"] = str(result["manager_feedback"])
        operation = result.get("operation_result") or {}
        if operation:
            summary["operation"] = {
                key: operation[key]
                for key in ("rcode", "stage", "action_result", "cancelled", "timed_out")
                if key in operation
            }
        review = result.get("post_execution_review") or result.get("pre_execution_review") or {}
        if review:
            summary["review"] = {
                key: review[key]
                for key in ("passed", "decision", "reason", "issues", "required_actions")
                if key in review
            }
        knowledge = result.get("knowledge_search_report") or {}
        if knowledge:
            validation = knowledge.get("evidence_validation") or {}
            summary["knowledge_search"] = {
                "status": knowledge.get("status"),
                "snippet_count": len(knowledge.get("snippets") or []),
                "retrieval_mode": knowledge.get("retrieval_mode"),
                "routing_mode": knowledge.get("routing_mode"),
                "embedding_model": knowledge.get("embedding_model"),
                "context_chars_used": knowledge.get("context_chars_used"),
                "valid_evidence_refs": knowledge.get("selected_valid_count", validation.get("valid_count")),
                "invalid_evidence_refs": validation.get("invalid_count"),
                "errors": knowledge.get("errors") or [],
                "query_routes": [
                    {
                        "query": item.get("query"),
                        "resolved_corpus": (item.get("routing") or {}).get("resolved_corpus"),
                        "route_reason": (item.get("routing") or {}).get("reason"),
                        "hit_count": item.get("hit_count"),
                        "latency_ms": item.get("latency_ms"),
                    }
                    for item in (knowledge.get("queries") or [])[:3]
                    if isinstance(item, dict)
                ],
                "top_evidence": [
                    {
                        "source_id": item.get("source_id"),
                        "title": item.get("title"),
                        "corpus": item.get("corpus"),
                        "retrieval_channels": item.get("retrieval_channels") or [],
                    }
                    for item in (knowledge.get("snippets") or [])[:5]
                    if isinstance(item, dict)
                ],
            }
        if stage == "data":
            lineage = result.get("data_lineage") or {}
            anchor_audit = result.get("anchor_audit") or {}
            if lineage:
                summary["data_lineage"] = {
                    key: lineage.get(key)
                    for key in (
                        "dataset_role",
                        "dataset_role_label",
                        "row_count",
                        "feature_count",
                        "fidelity_counts",
                        "augmented_count",
                        "evaluation_role",
                        "evaluation_boundary",
                    )
                }
            if anchor_audit:
                summary["anchor_audit"] = {
                    key: anchor_audit.get(key)
                    for key in (
                        "status",
                        "compatible",
                        "anchor_dataset_role",
                        "guardrail_action",
                        "missing_feature_count",
                        "reasons",
                    )
                }
        if result.get("error"):
            summary["error"] = str(result["error"])
        summary["stage"] = stage
        return summary

    def _record(
        self,
        state: DomainWorkflowState,
        stage: str,
        status: str,
        message: str,
        *,
        output_summary: dict[str, Any] | None = None,
    ) -> None:
        stage_summary = output_summary or {
            "artifact_names": sorted(state.get("artifacts", {})),
            "last_error": state.get("last_error", "") if status == "failed" else "",
            "manager_revision_round": int(state.get("manager_revision_round", 0)),
        }
        activity = self._finish_activity(
            state,
            stage,
            status,
            message,
            output_summary=stage_summary,
        )
        record = {
            "stage_execution_id": f"stage_{uuid.uuid4().hex[:12]}",
            "stage": stage,
            "status": status,
            "message": message,
            "schema_version": activity["schema_version"],
            "activity_id": activity["activity_id"],
            "sequence": activity["sequence"],
            "workflow_kind": activity["workflow_kind"],
            "agent": activity["agent"],
            "action": activity["action"],
            "attempt": activity["attempt"],
            "started_at": activity["started_at"],
            "completed_at": activity["completed_at"],
            "duration_ms": activity["duration_ms"],
            "input_summary": activity["input_summary"],
            "output_summary": activity["output_summary"],
            "ts": time.time(),
        }
        state.setdefault("stages", []).append(record)
        if state.get("run_id"):
            self.store.record_stage_execution(
                stage_execution_id=record["stage_execution_id"],
                run_id=state["run_id"],
                thread_id=state["thread_id"],
                stage=stage,
                status=status,
                input_hash=f"domain-{stage}-{state.get('manager_revision_round', 0)}",
                started_at=record["started_at"],
                completed_at=record["completed_at"],
                payload={
                    "message": message,
                    "agent": record["agent"],
                    "action": record["action"],
                    "attempt": record["attempt"],
                    "duration_ms": record["duration_ms"],
                    "input_summary": record["input_summary"],
                    "output_summary": record["output_summary"],
                },
                error=state.get("last_error", "") if status == "failed" else "",
            )

    def _apply(self, state: DomainWorkflowState, stage: str, result: dict[str, Any]) -> None:
        for key in (
            "runtime_env",
            "operation_result",
            "post_execution_review",
            "process_control",
            "manager_feedback",
            "pre_execution_passed",
            "cancelled",
            "real_llm_calls",
            "simulated_model_calls",
            "model_call_mode",
            "knowledge_search_report",
            "data_lineage",
            "anchor_audit",
            "data_contract",
            "session_context_summary",
        ):
            if key in result:
                state[key] = result[key]
        if "model_usage_records" in result:
            state["model_usage_records"] = merge_model_usage_records(
                state.get("model_usage_records"),
                result.get("model_usage_records"),
            )
            summary = summarize_model_usage(state["model_usage_records"])
            state["model_usage_summary"] = summary
            state["real_llm_calls"] = int(summary["calls_total"])
            state["model_call_mode"] = str(summary["mode"])
        if result.get("artifacts"):
            state.setdefault("artifacts", {}).update(result["artifacts"])
        ok = bool(result.get("ok", True))
        state["last_node_ok"] = ok
        state["last_error"] = str(result.get("error") or "")
        if not ok and state["last_error"]:
            state.setdefault("errors", []).append(state["last_error"])
        self._record(
            state,
            stage,
            "passed" if ok else "failed",
            str(result.get("message") or f"{stage} completed."),
            output_summary=self._result_summary(stage, result),
        )

    def _call(self, state: DomainWorkflowState, stage: str) -> DomainWorkflowState:
        self._begin_activity(state, stage)
        try:
            result = getattr(self.adapter, stage)(state)
            self._apply(state, stage, result)
        except Exception as exc:
            state["last_node_ok"] = False
            state["last_error"] = f"{stage} failed: {type(exc).__name__}: {exc}"
            state.setdefault("errors", []).append(state["last_error"])
            self._record(state, stage, "failed", state["last_error"])
        return state

    def _call_and_route(self, state: DomainWorkflowState, stage: str) -> DomainWorkflowState:
        state = self._call(state, stage)
        self._record_manager_decision(state, stage)
        return state

    def _node_start(self, state: DomainWorkflowState) -> DomainWorkflowState:
        self._begin_activity(state, "start")
        run_id = self.store.create_run(
            mode="live_llm" if self.adapter.uses_live_llm else "simulated_test_adapter",
            source="yield_domain_stategraph",
            status="running",
            metadata={
                "run_dir": state["run_dir"],
                "allow_live_llm": state["allow_live_llm"],
                "workspace_id": state.get("workspace_id", "default"),
                "session_id": state.get("session_id", ""),
                "turn_id": state.get("turn_id", ""),
                "use_knowledge_search": bool(state.get("use_knowledge_search", True)),
                "knowledge_top_k": int(state.get("knowledge_top_k", 5)),
                "knowledge_context_budget_chars": int(
                    state.get("knowledge_context_budget_chars", 12000)
                ),
            },
        )
        state["run_id"] = run_id
        session_id = str(state.get("session_id") or "")
        turn_id = str(state.get("turn_id") or "")
        if turn_id and not session_id:
            state["last_node_ok"] = False
            state["last_error"] = "Session context loading failed: turn_id requires session_id."
            state.setdefault("errors", []).append(state["last_error"])
        elif session_id:
            try:
                memory = SessionMemoryStore(self.store)
                if turn_id:
                    memory.validate_turn(session_id, turn_id)
                context = memory.build_context(
                    session_id,
                    max_tokens=int(state.get("session_context_max_tokens", 1200)),
                )
                if str(context.get("workspace_id") or "default") != str(state.get("workspace_id") or "default"):
                    raise ValueError("session workspace_id does not match workflow workspace_id.")
                memory.link_run(session_id, run_id)
                state["session_context"] = context
                state["session_context_prompt"] = render_session_context(context)
                state["session_context_summary"] = session_context_summary(context)
            except Exception as exc:
                state["session_context"] = {}
                state["session_context_prompt"] = ""
                state["session_context_summary"] = {}
                state["last_node_ok"] = False
                state["last_error"] = f"Session context loading failed: {type(exc).__name__}: {exc}"
                state.setdefault("errors", []).append(state["last_error"])
        try:
            manager_args = dict(state.get("manager_args") or {})
            memory_query = "\n".join(
                [
                    str(manager_args.get("prompt") or ""),
                    *[str(item) for item in (manager_args.get("query") or [])],
                ]
            ).strip()
            self._refresh_repair_memories(
                state,
                query_text=memory_query,
                phase="task_preflight",
            )
        except Exception as exc:
            state["repair_memories"] = []
            state["repair_memory_context"] = ""
            state["repair_memory_retrieval"] = {
                "phase": "task_preflight",
                "selected_count": 0,
                "degraded_error": type(exc).__name__,
            }
            state.setdefault("errors", []).append(
                f"Repair memory retrieval degraded: {type(exc).__name__}: {exc}"
            )
        if self.on_run_started:
            self.on_run_started(run_id)
        if not state.get("last_node_ok", True):
            self._record(state, "start", "failed", state["last_error"])
        elif self.adapter.uses_live_llm and not state.get("allow_live_llm"):
            state["last_node_ok"] = False
            state["last_error"] = "Real domain workflow requires allow_live_llm=true."
            state["errors"].append(state["last_error"])
            self._record(state, "start", "failed", state["last_error"])
        else:
            self._record(
                state,
                "start",
                "passed",
                "Domain StateGraph run created.",
                output_summary={
                    "artifact_names": [],
                    "last_error": "",
                    "manager_revision_round": 0,
                    "verified_repair_memories_loaded": len(state.get("repair_memories", [])),
                    "repair_memory_retrieval": dict(state.get("repair_memory_retrieval") or {}),
                    "session_context": dict(state.get("session_context_summary") or {}),
                },
            )
        self._record_manager_decision(state, "start")
        return state

    def _node_prepare(self, state: DomainWorkflowState) -> DomainWorkflowState:
        return self._call_and_route(state, "prepare")

    def _node_data(self, state: DomainWorkflowState) -> DomainWorkflowState:
        state = self._call(state, "data")
        if state.get("last_node_ok", True):
            try:
                manager_args = dict(state.get("manager_args") or {})
                memory_query = "\n".join(
                    [
                        str(manager_args.get("prompt") or ""),
                        *[str(item) for item in (manager_args.get("query") or [])],
                    ]
                ).strip()
                self._refresh_repair_memories(
                    state,
                    query_text=memory_query,
                    phase="data_contract_preflight",
                )
            except Exception as exc:
                state.setdefault("errors", []).append(
                    f"Repair memory data-contract filtering degraded: {type(exc).__name__}: {exc}"
                )
        self._record_manager_decision(state, "data")
        return state

    def _node_requirements(self, state: DomainWorkflowState) -> DomainWorkflowState:
        return self._call_and_route(state, "requirements")

    def _node_search(self, state: DomainWorkflowState) -> DomainWorkflowState:
        return self._call_and_route(state, "search")

    def _node_candidate(self, state: DomainWorkflowState) -> DomainWorkflowState:
        return self._call_and_route(state, "candidate")

    def _node_model(self, state: DomainWorkflowState) -> DomainWorkflowState:
        return self._call_and_route(state, "model")

    def _node_pre_execution(self, state: DomainWorkflowState) -> DomainWorkflowState:
        return self._call_and_route(state, "pre_execution")

    def _node_operation(self, state: DomainWorkflowState) -> DomainWorkflowState:
        state = self._call(state, "operation")
        operation = state.get("operation_result", {})
        try:
            rcode = int(operation.get("rcode", 1))
        except (TypeError, ValueError):
            rcode = 1
        error_logs = operation.get("error_logs", []) or []
        if not operation.get("cancelled") and (rcode != 0 or error_logs):
            try:
                candidate = self.repair_memory_store.record_candidate(
                    workspace_id=str(state.get("workspace_id") or "default"),
                    run_id=str(state.get("run_id") or ""),
                    execution_mode=str(state.get("execution_mode") or "free_search"),
                    operation_result=operation,
                    applicability_context=self._repair_applicability_context(state),
                )
                if candidate:
                    memory_id = str(candidate.get("memory_id") or "")
                    if memory_id and memory_id not in state.setdefault("pending_repair_memory_ids", []):
                        state["pending_repair_memory_ids"].append(memory_id)
            except Exception as exc:
                state.setdefault("errors", []).append(
                    f"Repair memory candidate persistence degraded: {type(exc).__name__}: {exc}"
                )
            try:
                error_query = operation_error_text(operation)
                if error_query:
                    self._refresh_repair_memories(
                        state,
                        query_text=error_query,
                        phase="operation_error",
                        min_relevance=0.08,
                        strict_error_match=True,
                    )
                    retrieval = dict(state.get("repair_memory_retrieval") or {})
                    state["repair_memory_reuse_ids"] = list(retrieval.get("selected_memory_ids") or [])
                    state["repair_memory_reuse_round"] = int(state.get("manager_revision_round", 0))
                    self._record_operation_detail(
                        state,
                        {
                            "agent": "Agent Manager",
                            "action": "retrieve_verified_repair_memory",
                            "attempt": int(state.get("manager_revision_round", 0)) + 1,
                            "detail_status": "warning",
                            "message": (
                                "Manager matched the current Operation error against verified cross-run repairs "
                                f"and selected {retrieval.get('selected_count', 0)} item(s)."
                            ),
                            "output_summary": retrieval,
                        },
                    )
            except Exception as exc:
                state.setdefault("errors", []).append(
                    f"Repair memory semantic retrieval degraded: {type(exc).__name__}: {exc}"
                )
        self._record_manager_decision(state, "operation")
        return state

    def _node_review(self, state: DomainWorkflowState) -> DomainWorkflowState:
        state = self._call(state, "review")
        review = state.get("post_execution_review", {})
        operation = state.get("operation_result", {})
        completed_reuse_attempts: list[dict[str, Any]] = []
        for reuse_id in list(state.get("repair_memory_active_reuse_ids", [])):
            try:
                completed = self.repair_memory_store.complete_reuse_attempt(
                    reuse_id,
                    operation_result=operation,
                    manager_review=review,
                )
                if completed:
                    completed_reuse_attempts.append(completed)
            except Exception as exc:
                state.setdefault("errors", []).append(
                    f"Repair memory reuse outcome persistence degraded: {type(exc).__name__}: {exc}"
                )
        state["repair_memory_active_reuse_ids"] = []
        if completed_reuse_attempts:
            self._record_operation_detail(
                state,
                {
                    "agent": "Agent Manager",
                    "action": "record_repair_memory_reuse_outcome",
                    "attempt": int(state.get("manager_revision_round", 0)) + 1,
                    "detail_status": (
                        "success"
                        if all(item.get("status") == "succeeded" for item in completed_reuse_attempts)
                        else "warning"
                    ),
                    "message": "Manager recorded the reviewed outcome of applied Repair Memory guidance.",
                    "output_summary": {
                        "reuse_attempt_ids": [item.get("reuse_id") for item in completed_reuse_attempts],
                        "memory_ids": [item.get("memory_id") for item in completed_reuse_attempts],
                        "outcomes": [item.get("status") for item in completed_reuse_attempts],
                        "operation_rcode": operation.get("rcode"),
                        "manager_passed": bool(review.get("passed")),
                    },
                },
            )
        if review.get("passed"):
            for memory_id in list(state.get("pending_repair_memory_ids", [])):
                try:
                    confirmed = self.repair_memory_store.confirm(
                        memory_id,
                        successful_operation=operation,
                        manager_review=review,
                        repair_summary=str(state.get("manager_feedback") or ""),
                    )
                    if confirmed and confirmed.get("validation_status") == "confirmed":
                        if memory_id not in state.setdefault("verified_repair_memory_ids", []):
                            state["verified_repair_memory_ids"].append(memory_id)
                except Exception as exc:
                    state.setdefault("errors", []).append(
                        f"Repair memory confirmation degraded: {type(exc).__name__}: {exc}"
                    )
            reuse_round = int(state.get("repair_memory_reuse_round", -1))
            current_round = int(state.get("manager_revision_round", 0))
            if current_round > reuse_round:
                reconfirmed_this_review: list[str] = []
                for memory_id in list(state.get("repair_memory_reuse_ids", [])):
                    try:
                        reconfirmed = self.repair_memory_store.reconfirm(
                            memory_id,
                            evidence_run_id=str(state.get("run_id") or ""),
                            successful_operation=operation,
                            manager_review=review,
                            source="workflow_error_match_then_success",
                            note=str(state.get("manager_feedback") or ""),
                        )
                        if reconfirmed:
                            if memory_id not in state.setdefault("reconfirmed_repair_memory_ids", []):
                                state["reconfirmed_repair_memory_ids"].append(memory_id)
                            reconfirmed_this_review.append(memory_id)
                    except Exception as exc:
                        state.setdefault("errors", []).append(
                            f"Repair memory reconfirmation degraded: {type(exc).__name__}: {exc}"
                        )
                if reconfirmed_this_review:
                    self._record_operation_detail(
                        state,
                        {
                            "agent": "Agent Manager",
                            "action": "reconfirm_verified_repair_memory",
                            "attempt": current_round + 1,
                            "detail_status": "success",
                            "message": (
                                "Manager reconfirmed verified repair memory after a revised Operation "
                                "succeeded and passed review."
                            ),
                            "output_summary": {
                                "memory_ids": reconfirmed_this_review,
                                "evidence_run_id": state.get("run_id", ""),
                                "manager_revision_round": current_round,
                            },
                        },
                    )
        if state.get("last_node_ok") and not review.get("passed"):
            state["last_error"] = f"Post-execution decision: {review.get('decision', 'not accepted')}"
        self._record_manager_decision(state, "review")
        return state

    def _node_revision_feedback(self, state: DomainWorkflowState) -> DomainWorkflowState:
        state["manager_revision_round"] = int(state.get("manager_revision_round", 0)) + 1
        current_round = int(state["manager_revision_round"])
        matched_round = int(state.get("repair_memory_reuse_round", -1))
        retrieval_by_memory = {
            str(memory.get("memory_id") or ""): dict(memory.get("retrieval") or {})
            for memory in state.get("repair_memories", [])
        }
        applied_attempts: list[dict[str, Any]] = []
        if current_round > matched_round:
            for memory_id in list(state.get("repair_memory_reuse_ids", [])):
                try:
                    attempt = self.repair_memory_store.begin_reuse_attempt(
                        memory_id,
                        run_id=str(state.get("run_id") or ""),
                        workspace_id=str(state.get("workspace_id") or "default"),
                        execution_mode=str(state.get("execution_mode") or "free_search"),
                        matched_round=matched_round,
                        applied_round=current_round,
                        retrieval=retrieval_by_memory.get(memory_id, {}),
                        query_signature=str(
                            (state.get("repair_memory_retrieval") or {}).get("query_signature") or ""
                        ),
                    )
                    applied_attempts.append(attempt)
                    reuse_id = str(attempt.get("reuse_id") or "")
                    if reuse_id and reuse_id not in state.setdefault("repair_memory_active_reuse_ids", []):
                        state["repair_memory_active_reuse_ids"].append(reuse_id)
                except Exception as exc:
                    state.setdefault("errors", []).append(
                        f"Repair memory reuse application persistence degraded: {type(exc).__name__}: {exc}"
                    )
        if applied_attempts:
            self._record_operation_detail(
                state,
                {
                    "agent": "Agent Manager",
                    "action": "apply_verified_repair_memory",
                    "attempt": current_round + 1,
                    "detail_status": "info",
                    "message": "Manager applied matched Repair Memory guidance to the next revision round.",
                    "output_summary": {
                        "reuse_attempt_ids": [item.get("reuse_id") for item in applied_attempts],
                        "memory_ids": [item.get("memory_id") for item in applied_attempts],
                        "matched_round": matched_round,
                        "applied_round": current_round,
                    },
                },
            )
        return self._call_and_route(state, "revision_feedback")

    def _node_cancel(self, state: DomainWorkflowState) -> DomainWorkflowState:
        self._begin_activity(state, "cancel")
        state["cancelled"] = True
        state["status"] = "cancelled"
        self._record(state, "cancel", "cancelled", "Domain workflow cancellation completed.")
        self._record_manager_decision(state, "cancel")
        return state

    def _node_finish(self, state: DomainWorkflowState) -> DomainWorkflowState:
        self._begin_activity(state, "finish")
        review = state.get("post_execution_review", {})
        if state.get("cancelled"):
            status = "cancelled"
        elif review.get("passed"):
            status = "passed"
        else:
            status = "failed"
            if state.get("last_error") and state["last_error"] not in state["errors"]:
                state["errors"].append(state["last_error"])
        state["status"] = status
        self._record(state, "finish", status, f"Domain workflow finished with status={status}.")
        self.store.update_run(state["run_id"], status=status, result=dict(state), completed=True)
        return state

    def _route_manager_decision(self, state: DomainWorkflowState) -> str:
        decision = state.get("manager_decision", {})
        next_node = str(decision.get("next_node") or "finish")
        return next_node

    def _build_graph(self):
        try:
            from langgraph.graph import END, StateGraph
        except ImportError as exc:
            raise RuntimeError("The domain workflow requires LangGraph; install project requirements first.") from exc

        graph = StateGraph(DomainWorkflowState)
        nodes = {
            "start": self._node_start,
            "prepare": self._node_prepare,
            "data": self._node_data,
            "requirements": self._node_requirements,
            "search": self._node_search,
            "candidate": self._node_candidate,
            "model": self._node_model,
            "pre_execution": self._node_pre_execution,
            "operation": self._node_operation,
            "review": self._node_review,
            "revision_feedback": self._node_revision_feedback,
            "cancel": self._node_cancel,
            "finish": self._node_finish,
        }
        for name, node in nodes.items():
            graph.add_node(name, node)
        graph.set_entry_point("start")
        graph.add_conditional_edges(
            "start",
            self._route_manager_decision,
            {"prepare": "prepare", "cancel": "cancel", "finish": "finish"},
        )
        for current, target in (
            ("prepare", "data"),
            ("data", "requirements"),
            ("requirements", "search"),
            ("search", "candidate"),
            ("candidate", "model"),
            ("model", "pre_execution"),
            ("revision_feedback", "candidate"),
        ):
            graph.add_conditional_edges(
                current,
                self._route_manager_decision,
                {target: target, "cancel": "cancel", "finish": "finish"},
            )
        graph.add_conditional_edges(
            "pre_execution",
            self._route_manager_decision,
            {"operation": "operation", "review": "review", "cancel": "cancel", "finish": "finish"},
        )
        graph.add_conditional_edges(
            "operation",
            self._route_manager_decision,
            {"review": "review", "cancel": "cancel", "finish": "finish"},
        )
        graph.add_conditional_edges(
            "review",
            self._route_manager_decision,
            {"revision_feedback": "revision_feedback", "cancel": "cancel", "finish": "finish"},
        )
        graph.add_conditional_edges("cancel", self._route_manager_decision, {"finish": "finish"})
        graph.add_edge("finish", END)
        return graph

    def run(self, request: DomainWorkflowRequest) -> dict[str, Any]:
        from langgraph.checkpoint.memory import MemorySaver

        run_dir = Path(request.run_dir or f"agent_workspace/runs/yield_domain_{time.strftime('%Y%m%d_%H%M%S')}").resolve()
        synthetic_data = request.synthetic_data
        if synthetic_data is None:
            synthetic_data = not Path(request.data_path).expanduser().exists()
        manager_args = {
            "prompt": request.prompt,
            "data_path": request.data_path,
            "test_path": request.test_path,
            "run_dir": str(run_dir),
            "llm": request.llm,
            "n_revise": request.n_revise,
            "operation_attempts": request.operation_attempts,
            "random_state": request.random_state,
            "external_search": request.external_search,
            "require_search_results": request.require_search_results,
            "synthetic_data": synthetic_data,
            "synthetic_n": request.synthetic_n,
            "query": request.query or None,
            "use_knowledge_search": request.use_knowledge_search,
            "knowledge_top_k": request.knowledge_top_k,
            "knowledge_context_budget_chars": request.knowledge_context_budget_chars,
            "session_id": request.session_id,
            "turn_id": request.turn_id,
            "session_context_max_tokens": request.session_context_max_tokens,
            "champion_policy": request.champion_policy,
        }
        state: DomainWorkflowState = {
            "thread_id": f"yield_domain_{uuid.uuid4().hex[:12]}",
            "status": "created",
            "workflow_backend": "langgraph_stategraph",
            "checkpoint_backend": "langgraph_postgres" if self.store.backend == "postgresql" else "langgraph_memory_saver",
            "manager_args": manager_args,
            "allow_live_llm": request.allow_live_llm,
            "execution_mode": request.execution_mode,
            "operation_timeout_seconds": request.operation_timeout_seconds,
            "workspace_id": request.workspace_id,
            "session_id": request.session_id,
            "turn_id": request.turn_id,
            "session_context_max_tokens": request.session_context_max_tokens,
            "session_context": {},
            "session_context_prompt": "",
            "session_context_summary": {},
            "use_knowledge_search": request.use_knowledge_search,
            "knowledge_top_k": request.knowledge_top_k,
            "knowledge_context_budget_chars": request.knowledge_context_budget_chars,
            "knowledge_search_report": {},
            "run_dir": str(run_dir),
            "runtime_env": {
                "YIELD_RUN_DIR": str(run_dir),
                "YIELD_EXECUTION_MODE": request.execution_mode,
                "YIELD_CHAMPION_POLICY": request.champion_policy,
            },
            "runtime_contract": _runtime_contract(),
            "data_contract": {},
            "data_lineage": {},
            "anchor_audit": {},
            "artifacts": {},
            "stages": [],
            "route_history": [],
            "manager_decision": {},
            "manager_decisions": [],
            "agent_activities": [],
            "errors": [],
            "last_node_ok": True,
            "last_error": "",
            "manager_revision_round": 0,
            "max_manager_revisions": request.n_revise,
            "manager_feedback": "",
            "pre_execution_passed": False,
            "operation_result": {},
            "repair_memories": [],
            "repair_memory_context": "",
            "repair_memory_retrieval": {},
            "pending_repair_memory_ids": [],
            "verified_repair_memory_ids": [],
            "repair_memory_reuse_ids": [],
            "repair_memory_reuse_round": -1,
            "repair_memory_active_reuse_ids": [],
            "reconfirmed_repair_memory_ids": [],
            "post_execution_review": {},
            "process_control": {},
            "cancelled": False,
            "real_llm_calls": (
                None if self.adapter.uses_live_llm and request.allow_live_llm else 0
            ),
            "simulated_model_calls": 0,
            "model_call_mode": (
                "live_unmetered"
                if self.adapter.uses_live_llm and request.allow_live_llm
                else "live_blocked"
                if self.adapter.uses_live_llm
                else "simulated_test_adapter"
            ),
            "model_usage_records": [],
            "model_usage_summary": summarize_model_usage([]),
        }
        graph = self._build_graph()
        config = {"configurable": {"thread_id": state["thread_id"]}}
        if self.store.backend == "postgresql":
            from langgraph.checkpoint.postgres import PostgresSaver

            with PostgresSaver.from_conn_string(psycopg_connection_string(self.store.database_url)) as checkpointer:
                final = graph.compile(checkpointer=checkpointer).invoke(state, config=config)
        else:
            final = graph.compile(checkpointer=MemorySaver()).invoke(state, config=config)
        return dict(final)


def run_domain_workflow(
    request: DomainWorkflowRequest,
    *,
    store: YieldMindStore | None = None,
    adapter: DomainWorkflowAdapter | None = None,
    cancel_check: Callable[[], bool] | None = None,
    on_run_started: Callable[[str], None] | None = None,
    on_stage_progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    return YieldDomainWorkflow(
        store=store,
        adapter=adapter,
        cancel_check=cancel_check,
        on_run_started=on_run_started,
        on_stage_progress=on_stage_progress,
    ).run(request)
