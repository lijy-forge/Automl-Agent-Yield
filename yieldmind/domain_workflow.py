"""LangGraph orchestration for the original multi-agent yield workflow.

The graph state contains only serializable configuration, summaries, and
artifact paths. The original domain methods remain the source of truth and are
rehydrated from their JSON artifacts at each node boundary.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Protocol, TypedDict

from pydantic import BaseModel, Field

from knowledge.yield_schema import DEFAULT_LIAN_DATA_PATH, DEFAULT_LIAN_TEST_PATH
from yieldmind.checkpointing import psycopg_connection_string
from yieldmind.database import YieldMindStore
from yieldmind.process_control import ManagedProcessOutcome, run_managed_process


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
    operation_timeout_seconds: float = Field(default=900.0, ge=1.0, le=86400.0)
    allow_live_llm: bool = False


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
    artifacts: dict[str, str]
    stages: list[dict[str, Any]]
    route_history: list[dict[str, Any]]
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
)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


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
        "run_result": run_dir / "run_result.json",
        "post_execution_review": run_dir / "post_execution_review.json",
        "manager_revision_notes": run_dir / "manager_revision_notes.json",
    }
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

    def __init__(self, *, cancel_check: Callable[[], bool] | None = None) -> None:
        self.cancel_check = cancel_check

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
        return self._result(state)

    def requirements(self, state: DomainWorkflowState) -> dict[str, Any]:
        manager = _hydrate_manager(state, cancel_check=self.cancel_check)
        manager._emit_requirement_context()
        return self._result(state)

    def search(self, state: DomainWorkflowState) -> dict[str, Any]:
        manager = _hydrate_manager(state, cancel_check=self.cancel_check)
        ok = manager._run_search_agent()
        return self._result(state, ok=bool(ok), error="" if ok else "Required external search returned no evidence.")

    def candidate(self, state: DomainWorkflowState) -> dict[str, Any]:
        manager = _hydrate_manager(state, cancel_check=self.cancel_check)
        manager._run_candidate_agent(manager_feedback=state.get("manager_feedback", ""))
        return self._result(state)

    def model(self, state: DomainWorkflowState) -> dict[str, Any]:
        manager = _hydrate_manager(state, cancel_check=self.cancel_check)
        manager._run_model_agent(manager_feedback=state.get("manager_feedback", ""))
        return self._result(state)

    def pre_execution(self, state: DomainWorkflowState) -> dict[str, Any]:
        manager = _hydrate_manager(state, cancel_check=self.cancel_check)
        manager._transition("PRE_EXEC", "AgentManager building operation contract and checking readiness.")
        manager.operation_instructions = manager._build_operation_instructions()
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
        outcome = run_managed_process(
            _real_operation_worker,
            (dict(state),),
            timeout_seconds=float(state.get("operation_timeout_seconds", 900.0)),
            cancel_check=self.cancel_check,
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
        return self._result(
            state,
            operation_result=result,
            process_control=outcome.metadata(),
            cancelled=bool(result.get("cancelled")),
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
    ) -> None:
        self.store = store or YieldMindStore()
        self.cancel_check = cancel_check
        self.adapter = adapter or RealYieldDomainAdapter(cancel_check=cancel_check)
        self.on_run_started = on_run_started

    def _cancel_requested(self, state: DomainWorkflowState) -> bool:
        if state.get("cancelled"):
            return True
        return bool(self.cancel_check and self.cancel_check())

    def _record(self, state: DomainWorkflowState, stage: str, status: str, message: str) -> None:
        record = {
            "stage_execution_id": f"stage_{uuid.uuid4().hex[:12]}",
            "stage": stage,
            "status": status,
            "message": message,
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
                started_at=record["ts"],
                completed_at=time.time(),
                payload={"message": message},
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
        ):
            if key in result:
                state[key] = result[key]
        if result.get("artifacts"):
            state.setdefault("artifacts", {}).update(result["artifacts"])
        ok = bool(result.get("ok", True))
        state["last_node_ok"] = ok
        state["last_error"] = str(result.get("error") or "")
        if not ok and state["last_error"]:
            state.setdefault("errors", []).append(state["last_error"])
        self._record(state, stage, "passed" if ok else "failed", str(result.get("message") or f"{stage} completed."))

    def _call(self, state: DomainWorkflowState, stage: str) -> DomainWorkflowState:
        try:
            result = getattr(self.adapter, stage)(state)
            self._apply(state, stage, result)
        except Exception as exc:
            state["last_node_ok"] = False
            state["last_error"] = f"{stage} failed: {type(exc).__name__}: {exc}"
            state.setdefault("errors", []).append(state["last_error"])
            self._record(state, stage, "failed", state["last_error"])
        return state

    def _node_start(self, state: DomainWorkflowState) -> DomainWorkflowState:
        run_id = self.store.create_run(
            mode="live_llm" if self.adapter.uses_live_llm else "simulated_test_adapter",
            source="yield_domain_stategraph",
            status="running",
            metadata={"run_dir": state["run_dir"], "allow_live_llm": state["allow_live_llm"]},
        )
        state["run_id"] = run_id
        if self.on_run_started:
            self.on_run_started(run_id)
        if self.adapter.uses_live_llm and not state.get("allow_live_llm"):
            state["last_node_ok"] = False
            state["last_error"] = "Real domain workflow requires allow_live_llm=true."
            state["errors"].append(state["last_error"])
            self._record(state, "start", "failed", state["last_error"])
        else:
            self._record(state, "start", "passed", "Domain StateGraph run created.")
        return state

    def _node_prepare(self, state: DomainWorkflowState) -> DomainWorkflowState:
        return self._call(state, "prepare")

    def _node_data(self, state: DomainWorkflowState) -> DomainWorkflowState:
        return self._call(state, "data")

    def _node_requirements(self, state: DomainWorkflowState) -> DomainWorkflowState:
        return self._call(state, "requirements")

    def _node_search(self, state: DomainWorkflowState) -> DomainWorkflowState:
        return self._call(state, "search")

    def _node_candidate(self, state: DomainWorkflowState) -> DomainWorkflowState:
        return self._call(state, "candidate")

    def _node_model(self, state: DomainWorkflowState) -> DomainWorkflowState:
        return self._call(state, "model")

    def _node_pre_execution(self, state: DomainWorkflowState) -> DomainWorkflowState:
        return self._call(state, "pre_execution")

    def _node_operation(self, state: DomainWorkflowState) -> DomainWorkflowState:
        return self._call(state, "operation")

    def _node_review(self, state: DomainWorkflowState) -> DomainWorkflowState:
        state = self._call(state, "review")
        review = state.get("post_execution_review", {})
        if state.get("last_node_ok") and not review.get("passed"):
            state["last_error"] = f"Post-execution decision: {review.get('decision', 'not accepted')}"
        return state

    def _node_revision_feedback(self, state: DomainWorkflowState) -> DomainWorkflowState:
        state["manager_revision_round"] = int(state.get("manager_revision_round", 0)) + 1
        return self._call(state, "revision_feedback")

    def _node_cancel(self, state: DomainWorkflowState) -> DomainWorkflowState:
        state["cancelled"] = True
        state["status"] = "cancelled"
        self._record(state, "cancel", "cancelled", "Domain workflow cancellation completed.")
        return state

    def _node_finish(self, state: DomainWorkflowState) -> DomainWorkflowState:
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

    def _route_linear(self, state: DomainWorkflowState, target: str) -> str:
        if self._cancel_requested(state):
            return "cancel"
        return target if state.get("last_node_ok") else "finish"

    def _route_start(self, state: DomainWorkflowState) -> str:
        return self._route_linear(state, "prepare")

    def _route_pre_execution(self, state: DomainWorkflowState) -> str:
        if self._cancel_requested(state):
            return "cancel"
        if not state.get("last_node_ok"):
            return "finish"
        return "operation" if state.get("pre_execution_passed") else "review"

    def _route_operation(self, state: DomainWorkflowState) -> str:
        if state.get("cancelled") or self._cancel_requested(state):
            return "cancel"
        return "review" if state.get("last_node_ok") else "finish"

    def _route_review(self, state: DomainWorkflowState) -> str:
        if self._cancel_requested(state):
            return "cancel"
        if not state.get("last_node_ok"):
            return "finish"
        if state.get("post_execution_review", {}).get("passed"):
            return "finish"
        if int(state.get("manager_revision_round", 0)) < int(state.get("max_manager_revisions", 0)):
            return "revision_feedback"
        return "finish"

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
        graph.add_conditional_edges("start", self._route_start, {"prepare": "prepare", "cancel": "cancel", "finish": "finish"})
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
                lambda current_state, next_node=target: self._route_linear(current_state, next_node),
                {target: target, "cancel": "cancel", "finish": "finish"},
            )
        graph.add_conditional_edges(
            "pre_execution",
            self._route_pre_execution,
            {"operation": "operation", "review": "review", "cancel": "cancel", "finish": "finish"},
        )
        graph.add_conditional_edges(
            "operation",
            self._route_operation,
            {"review": "review", "cancel": "cancel", "finish": "finish"},
        )
        graph.add_conditional_edges(
            "review",
            self._route_review,
            {"revision_feedback": "revision_feedback", "cancel": "cancel", "finish": "finish"},
        )
        graph.add_edge("cancel", "finish")
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
            "run_dir": str(run_dir),
            "runtime_env": {"YIELD_RUN_DIR": str(run_dir), "YIELD_EXECUTION_MODE": request.execution_mode},
            "artifacts": {},
            "stages": [],
            "route_history": [],
            "errors": [],
            "last_node_ok": True,
            "last_error": "",
            "manager_revision_round": 0,
            "max_manager_revisions": request.n_revise,
            "manager_feedback": "",
            "pre_execution_passed": False,
            "operation_result": {},
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
) -> dict[str, Any]:
    return YieldDomainWorkflow(
        store=store,
        adapter=adapter,
        cancel_check=cancel_check,
        on_run_started=on_run_started,
    ).run(request)
