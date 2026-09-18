"""YieldMind workflow runner with optional LangGraph StateGraph support.

When `langgraph` is installed this module compiles a real StateGraph. In the
current lightweight environment it falls back to the same explicit node order
and records `langgraph_available=false` instead of pretending a migration has
already happened.
"""

from __future__ import annotations

import importlib.util
import hashlib
import json
import time
import uuid
from pathlib import Path
from typing import Any, Callable, TypedDict

from pydantic import BaseModel, Field

from yieldmind.checkpointing import psycopg_connection_string
from yieldmind.database import YieldMindStore
from yieldmind.tools import OFFLINE_MODE, ToolRegistry, registry_for_workspace


class WorkflowRequest(BaseModel):
    prompt: str = "Build and evaluate a deterministic yield-stress baseline."
    data_path: str = ""
    n_samples: int = Field(default=80, ge=20, le=5000)
    n_splits: int = Field(default=3, ge=2, le=10)
    random_state: int = Field(default=42, ge=0)
    mode: str = OFFLINE_MODE
    max_local_repairs: int = Field(default=1, ge=0, le=3)
    max_replans: int = Field(default=1, ge=0, le=3)


class WorkflowState(TypedDict, total=False):
    prompt: str
    data_path: str
    n_samples: int
    n_splits: int
    random_state: int
    run_id: str
    status: str
    stages: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]
    artifacts: dict[str, str]
    errors: list[str]
    langgraph_available: bool
    workflow_backend: str
    checkpoint_backend: str
    thread_id: str
    data_source: str
    last_node_ok: bool
    failed_stage: str
    failure_kind: str
    last_error: str
    local_repair_count: int
    replan_count: int
    max_local_repairs: int
    max_replans: int
    repair_target: str
    failure_history: list[dict[str, Any]]
    route_history: list[dict[str, Any]]
    cancelled: bool


def langgraph_available() -> bool:
    return bool(importlib.util.find_spec("langgraph"))


def _record(state: WorkflowState, stage: str, status: str, message: str, payload: dict[str, Any] | None = None) -> None:
    stage_payload = payload or {}
    input_hash = hashlib.sha256(
        json.dumps(stage_payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]
    state.setdefault("stages", []).append(
        {
            "stage_execution_id": f"stage_{uuid.uuid4().hex[:12]}",
            "stage": stage,
            "status": status,
            "message": message,
            "payload": stage_payload,
            "input_hash": input_hash,
            "ts": time.time(),
        }
    )


class YieldMindWorkflow:
    """Small workflow around the real ToolRegistry and existing yield modules."""

    def __init__(
        self,
        *,
        store: YieldMindStore | None = None,
        registry: ToolRegistry | None = None,
        cancel_check: Callable[[], bool] | None = None,
        on_run_started: Callable[[str], None] | None = None,
    ) -> None:
        self.store = store or YieldMindStore()
        self.registry = registry or registry_for_workspace(store=self.store)
        self.cancel_check = cancel_check
        self.on_run_started = on_run_started

    def _cancellation_requested(self, state: WorkflowState) -> bool:
        if state.get("cancelled"):
            return True
        return self.cancel_check is not None and self.cancel_check()

    def _node_cancel(self, state: WorkflowState) -> WorkflowState:
        stages = state.get("stages", [])
        after_stage = stages[-1]["stage"] if stages else "start"
        state["cancelled"] = True
        state["status"] = "cancelled"
        state.setdefault("route_history", []).append(
            {
                "route": "cancel",
                "after_stage": after_stage,
                "ts": time.time(),
            }
        )
        self._record_stage(
            state,
            "cancel",
            "cancelled",
            "Workflow stopped cooperatively at a node boundary.",
            {"after_stage": after_stage},
        )
        return state

    def _tool(self, state: WorkflowState, name: str, args: dict[str, Any]) -> dict[str, Any]:
        attempt = {
            "local_repair_count": int(state.get("local_repair_count", 0)),
            "replan_count": int(state.get("replan_count", 0)),
        }
        key_payload = {
            "thread_id": state.get("thread_id", ""),
            "tool": name,
            "args": args,
            "attempt": attempt,
        }
        idempotency_key = hashlib.sha256(
            json.dumps(key_payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        result = self.registry.execute(
            name,
            args,
            run_id=state.get("run_id"),
            idempotency_key=idempotency_key,
        )
        payload = result.model_dump()
        state.setdefault("tool_results", []).append({"tool": name, "args": args, "result": payload})
        if result.artifacts:
            state.setdefault("artifacts", {}).update(result.artifacts)
        return payload

    def _record_stage(
        self,
        state: WorkflowState,
        stage: str,
        status: str,
        message: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        _record(state, stage, status, message, payload)
        record = state["stages"][-1]
        run_id = state.get("run_id", "")
        if run_id:
            self.store.record_stage_execution(
                stage_execution_id=record["stage_execution_id"],
                run_id=run_id,
                thread_id=state.get("thread_id", ""),
                stage=stage,
                status=status,
                input_hash=record["input_hash"],
                started_at=record["ts"],
                completed_at=time.time(),
                payload={"message": message, **(payload or {})},
                error=state.get("last_error", "") if status == "failed" else "",
            )

    def _stage_result(
        self,
        state: WorkflowState,
        *,
        stage: str,
        result: dict[str, Any],
        message: str,
        payload: dict[str, Any] | None = None,
        failure_kind: str = "local_repair",
    ) -> None:
        ok = bool(result.get("ok"))
        error = str(result.get("error") or "")
        state["last_node_ok"] = ok
        state["failed_stage"] = "" if ok else stage
        state["failure_kind"] = "none" if ok else failure_kind
        state["last_error"] = "" if ok else error
        if not ok:
            state.setdefault("failure_history", []).append(
                {
                    "stage": stage,
                    "kind": failure_kind,
                    "error": error,
                    "local_repair_count": int(state.get("local_repair_count", 0)),
                    "replan_count": int(state.get("replan_count", 0)),
                    "ts": time.time(),
                }
            )
        self._record_stage(state, stage, "passed" if ok else "failed", message, payload)

    def _terminal_failure(self, state: WorkflowState, reason: str) -> None:
        if reason and reason not in state.setdefault("errors", []):
            state["errors"].append(reason)

    def _node_start(self, state: WorkflowState) -> WorkflowState:
        run_id = self.store.create_run(
            mode=state.get("workflow_backend", OFFLINE_MODE),
            source="yieldmind_workflow",
            status="running",
            metadata={"prompt": state.get("prompt", ""), "langgraph_available": state.get("langgraph_available", False)},
        )
        state["run_id"] = run_id
        if self.on_run_started is not None:
            self.on_run_started(run_id)
        state["status"] = "running"
        self.store.add_event(run_id, stage="workflow", message="YieldMind workflow started.", payload=dict(state))
        self._record_stage(state, "start", "passed", "Workflow run created.", {"run_id": run_id})
        return state

    def _node_prepare_data(self, state: WorkflowState) -> WorkflowState:
        data_path = str(state.get("data_path") or "").strip()
        if data_path:
            path = Path(data_path).expanduser()
            if path.exists():
                state["data_source"] = "provided"
                state["last_node_ok"] = True
                state["failed_stage"] = ""
                state["failure_kind"] = "none"
                state["last_error"] = ""
                self._record_stage(state, "prepare_data", "passed", "Using provided data path.", {"data_path": data_path})
                return state
            error = f"Provided data path does not exist: {path}"
            state["last_node_ok"] = False
            state["failed_stage"] = "prepare_data"
            state["failure_kind"] = "fatal"
            state["last_error"] = error
            state.setdefault("failure_history", []).append(
                {"stage": "prepare_data", "kind": "fatal", "error": error, "ts": time.time()}
            )
            self._terminal_failure(state, error)
            self._record_stage(state, "prepare_data", "failed", "Provided data path was rejected.", {"data_path": data_path})
            return state
        result = self._tool(
            state,
            "generate_demo_yield_data",
            {
                "n_samples": int(state.get("n_samples", 80)),
                "random_state": int(state.get("random_state", 42)),
                "overwrite": True,
            },
        )
        paths = ((result.get("result") or {}).get("paths") or {}) if isinstance(result, dict) else {}
        generated_path = paths.get("synthetic_path") or paths.get("data_path") or ""
        state["data_path"] = generated_path
        state["data_source"] = "generated"
        normalized = dict(result)
        normalized["ok"] = bool(result.get("ok") and generated_path)
        if not normalized["ok"] and not normalized.get("error"):
            normalized["error"] = "Demo data tool did not return a usable data path."
        self._stage_result(
            state,
            stage="prepare_data",
            result=normalized,
            message="Demo data prepared.",
            payload={"data_path": generated_path},
            failure_kind="replan",
        )
        return state

    def _node_profile(self, state: WorkflowState) -> WorkflowState:
        result = self._tool(state, "profile_yield_data", {"data_path": state.get("data_path", "")})
        self._stage_result(
            state,
            stage="profile",
            result=result,
            message="Yield data profiled.",
            payload={"row_count": ((result.get("result") or {}).get("row_count"))},
            failure_kind="replan" if state.get("data_source") == "generated" else "fatal",
        )
        return state

    def _node_evaluate(self, state: WorkflowState) -> WorkflowState:
        result = self._tool(
            state,
            "run_fixed_baseline_eval",
            {
                "data_path": state.get("data_path", ""),
                "n_splits": int(state.get("n_splits", 3)),
                "random_state": int(state.get("random_state", 42)),
            },
        )
        best = ((result.get("result") or {}).get("best_baseline") or {}) if isinstance(result, dict) else {}
        self._stage_result(
            state,
            stage="evaluate",
            result=result,
            message="Fixed baselines evaluated.",
            payload={"best_baseline": best},
        )
        return state

    def _node_candidate_benchmark(self, state: WorkflowState) -> WorkflowState:
        result = self._tool(
            state,
            "run_candidate_benchmark",
            {
                "data_path": state.get("data_path", ""),
                "n_splits": int(state.get("n_splits", 3)),
                "random_state": int(state.get("random_state", 42)),
            },
        )
        selected = ((result.get("result") or {}).get("benchmark_report") or {}).get("selected_strategy_id")
        self._stage_result(
            state,
            stage="candidate_benchmark",
            result=result,
            message="Candidate benchmark completed.",
            payload={"selected_strategy_id": selected},
        )
        return state

    def _node_verify_artifacts(self, state: WorkflowState) -> WorkflowState:
        artifact_paths = {key: value for key, value in state.get("artifacts", {}).items() if value}
        result = self._tool(
            state,
            "verify_artifacts",
            {
                "artifact_paths": artifact_paths,
                "require_json": [
                    key
                    for key in artifact_paths
                    if key.endswith("_report") or key.endswith("_audit") or key in {"candidate_report"}
                ],
            },
        )
        verification = result.get("result") or {}
        self._stage_result(
            state,
            stage="verify_artifacts",
            result=result,
            message="Workflow artifacts verified.",
            payload={"reasons": verification.get("reasons", []), "warnings": verification.get("warnings", [])},
        )
        return state

    def _node_report(self, state: WorkflowState) -> WorkflowState:
        result = self._tool(
            state,
            "build_run_report",
            {
                "title": "YieldMind Deterministic Workflow Report",
                "run_id": state.get("run_id", ""),
                "artifact_paths": {key: value for key, value in state.get("artifacts", {}).items() if value},
                "notes": [
                    "Generated by the offline deterministic YieldMind workflow.",
                    "No live LLM/model API call was made in this workflow.",
                ],
            },
        )
        self._stage_result(
            state,
            stage="report",
            result=result,
            message="Workflow report generated.",
            payload={"report_json": (result.get("artifacts") or {}).get("report_json")},
        )
        return state

    def _node_local_repair(self, state: WorkflowState) -> WorkflowState:
        failed_stage = state.get("failed_stage", "")
        state["local_repair_count"] = int(state.get("local_repair_count", 0)) + 1
        if failed_stage in {"evaluate", "candidate_benchmark"}:
            state["n_splits"] = max(2, int(state.get("n_splits", 3)) - 1)
            target = failed_stage
        elif failed_stage == "verify_artifacts":
            target = "candidate_benchmark"
        elif failed_stage == "report":
            target = "report"
        else:
            target = "finish"
            self._terminal_failure(state, state.get("last_error", "Unrecoverable workflow failure."))
        state["repair_target"] = target
        state.setdefault("route_history", []).append(
            {
                "route": "local_repair",
                "failed_stage": failed_stage,
                "target": target,
                "attempt": state["local_repair_count"],
                "ts": time.time(),
            }
        )
        self._record_stage(
            state,
            "local_repair",
            "passed" if target != "finish" else "failed",
            "Applied deterministic bounded local repair.",
            {"failed_stage": failed_stage, "target": target, "n_splits": state.get("n_splits")},
        )
        return state

    def _node_replan(self, state: WorkflowState) -> WorkflowState:
        state["replan_count"] = int(state.get("replan_count", 0)) + 1
        state["local_repair_count"] = 0
        state["random_state"] = int(state.get("random_state", 42)) + state["replan_count"]
        if state.get("data_source") == "generated":
            state["data_path"] = ""
        state.setdefault("route_history", []).append(
            {
                "route": "replan",
                "failed_stage": state.get("failed_stage", ""),
                "attempt": state["replan_count"],
                "ts": time.time(),
            }
        )
        self._record_stage(
            state,
            "replan",
            "passed",
            "Replanned deterministic workflow inputs within the configured budget.",
            {"random_state": state["random_state"], "data_source": state.get("data_source", "")},
        )
        return state

    def _route_after_stage(self, state: WorkflowState, success_target: str) -> str:
        if self._cancellation_requested(state):
            return "cancel"
        if state.get("last_node_ok"):
            return success_target
        failure_kind = state.get("failure_kind", "fatal")
        if failure_kind == "local_repair" and int(state.get("local_repair_count", 0)) < int(
            state.get("max_local_repairs", 0)
        ):
            return "local_repair"
        if failure_kind in {"local_repair", "replan"} and int(state.get("replan_count", 0)) < int(
            state.get("max_replans", 0)
        ):
            return "replan"
        self._terminal_failure(state, state.get("last_error", "Workflow recovery budget exhausted."))
        return "finish"

    def _route_after_repair(self, state: WorkflowState) -> str:
        if self._cancellation_requested(state):
            return "cancel"
        return state.get("repair_target", "finish")

    def _route_after_start(self, state: WorkflowState) -> str:
        return "cancel" if self._cancellation_requested(state) else "prepare_data"

    def _route_after_replan(self, state: WorkflowState) -> str:
        return "cancel" if self._cancellation_requested(state) else "prepare_data"

    def _node_finish(self, state: WorkflowState) -> WorkflowState:
        errors = state.get("errors", [])
        status = "cancelled" if state.get("cancelled") else "failed" if errors else "passed"
        state["status"] = status
        self._record_stage(state, "finish", status, "Workflow finished.", {"errors": errors})
        run_id = state.get("run_id")
        if run_id:
            self.store.add_event(run_id, stage="workflow", message=f"YieldMind workflow {status}.", payload=dict(state))
            self.store.update_run(run_id, status=status, result=dict(state), completed=True)
        return state

    def _run_fallback(self, state: WorkflowState) -> WorkflowState:
        state = self._node_start(state)
        current = self._route_after_start(state)
        nodes = {
            "prepare_data": (self._node_prepare_data, "profile"),
            "profile": (self._node_profile, "evaluate"),
            "evaluate": (self._node_evaluate, "candidate_benchmark"),
            "candidate_benchmark": (self._node_candidate_benchmark, "verify_artifacts"),
            "verify_artifacts": (self._node_verify_artifacts, "report"),
            "report": (self._node_report, "finish"),
        }
        while current != "finish":
            if current == "cancel":
                state = self._node_cancel(state)
                current = "finish"
                continue
            if current == "local_repair":
                state = self._node_local_repair(state)
                current = self._route_after_repair(state)
                continue
            if current == "replan":
                state = self._node_replan(state)
                current = self._route_after_replan(state)
                continue
            node, success_target = nodes[current]
            state = node(state)
            current = self._route_after_stage(state, success_target)
        return self._node_finish(state)

    def _run_langgraph(self, state: WorkflowState) -> WorkflowState:
        from langgraph.graph import END, StateGraph
        from langgraph.checkpoint.memory import MemorySaver

        graph = StateGraph(WorkflowState)
        graph.add_node("start", self._node_start)
        graph.add_node("prepare_data", self._node_prepare_data)
        graph.add_node("profile", self._node_profile)
        graph.add_node("evaluate", self._node_evaluate)
        graph.add_node("candidate_benchmark", self._node_candidate_benchmark)
        graph.add_node("verify_artifacts", self._node_verify_artifacts)
        graph.add_node("report", self._node_report)
        graph.add_node("local_repair", self._node_local_repair)
        graph.add_node("replan", self._node_replan)
        graph.add_node("cancel", self._node_cancel)
        graph.add_node("finish", self._node_finish)
        graph.set_entry_point("start")
        graph.add_conditional_edges(
            "start",
            self._route_after_start,
            {"prepare_data": "prepare_data", "cancel": "cancel"},
        )
        for stage, success_target in (
            ("prepare_data", "profile"),
            ("profile", "evaluate"),
            ("evaluate", "candidate_benchmark"),
            ("candidate_benchmark", "verify_artifacts"),
            ("verify_artifacts", "report"),
            ("report", "finish"),
        ):
            graph.add_conditional_edges(
                stage,
                lambda current_state, target=success_target: self._route_after_stage(current_state, target),
                {
                    success_target: success_target,
                    "local_repair": "local_repair",
                    "replan": "replan",
                    "cancel": "cancel",
                    "finish": "finish",
                },
            )
        graph.add_conditional_edges(
            "local_repair",
            self._route_after_repair,
            {
                "evaluate": "evaluate",
                "candidate_benchmark": "candidate_benchmark",
                "report": "report",
                "cancel": "cancel",
                "finish": "finish",
            },
        )
        graph.add_conditional_edges(
            "replan",
            self._route_after_replan,
            {"prepare_data": "prepare_data", "cancel": "cancel"},
        )
        graph.add_edge("cancel", "finish")
        graph.add_edge("finish", END)
        config = {"configurable": {"thread_id": state["thread_id"]}}
        if self.store.backend == "postgresql":
            from langgraph.checkpoint.postgres import PostgresSaver

            with PostgresSaver.from_conn_string(
                psycopg_connection_string(self.store.database_url)
            ) as checkpointer:
                app = graph.compile(checkpointer=checkpointer)
                return app.invoke(state, config=config)
        app = graph.compile(checkpointer=MemorySaver())
        return app.invoke(state, config=config)

    def run(self, request: WorkflowRequest) -> dict[str, Any]:
        has_langgraph = langgraph_available()
        state: WorkflowState = {
            "prompt": request.prompt,
            "data_path": request.data_path,
            "n_samples": request.n_samples,
            "n_splits": request.n_splits,
            "random_state": request.random_state,
            "stages": [],
            "tool_results": [],
            "artifacts": {},
            "errors": [],
            "failure_history": [],
            "route_history": [],
            "langgraph_available": has_langgraph,
            "workflow_backend": "langgraph_stategraph" if has_langgraph else "local_workflow_fallback",
            "checkpoint_backend": (
                "langgraph_postgres"
                if has_langgraph and self.store.backend == "postgresql"
                else "langgraph_memory_saver"
                if has_langgraph
                else "none"
            ),
            "thread_id": f"workflow_{uuid.uuid4().hex[:12]}",
            "data_source": "",
            "last_node_ok": True,
            "failed_stage": "",
            "failure_kind": "none",
            "last_error": "",
            "local_repair_count": 0,
            "replan_count": 0,
            "max_local_repairs": request.max_local_repairs,
            "max_replans": request.max_replans,
            "repair_target": "",
            "cancelled": False,
        }
        if has_langgraph:
            final_state = self._run_langgraph(state)
        else:
            final_state = self._run_fallback(state)
        return dict(final_state)


def run_workflow(
    request: WorkflowRequest,
    *,
    store: YieldMindStore | None = None,
    cancel_check: Callable[[], bool] | None = None,
    on_run_started: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    return YieldMindWorkflow(
        store=store,
        cancel_check=cancel_check,
        on_run_started=on_run_started,
    ).run(request)
