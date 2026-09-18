"""Function-calling adapters over the YieldMind ToolRegistry.

This module supports two explicitly separated modes:

- `offline_rule_planner`: deterministic local tool selection for regression
  tests. It is not presented as an LLM result.
- `live_llm_function_calling`: OpenAI-compatible Chat Completions tool calling
  through the existing project LLM client. It only runs when explicitly enabled.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from yieldmind.database import YieldMindStore, connect
from yieldmind.safety import RedactRequest, redact_payload
from yieldmind.tools import ToolRegistry, registry_for_workspace


OFFLINE_RULE_PLANNER = "offline_rule_planner"
LIVE_LLM_FUNCTION_CALLING = "live_llm_function_calling"
SIMULATED_TEST_ADAPTER = "simulated_test_adapter"


class ToolCallItem(BaseModel):
    tool_call_id: str = ""
    tool_name: str
    args: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""


class ToolPlanRequest(BaseModel):
    prompt: str
    data_path: str = ""
    session_id: str = ""
    turn_id: str = ""
    allow_live_llm: bool = False
    llm: str = "ark"
    max_tool_calls: int = Field(default=4, ge=1, le=12)
    max_steps: int = Field(default=4, ge=1, le=8)
    max_repeated_tool_calls: int = Field(default=1, ge=0, le=3)


class ToolPlanResult(BaseModel):
    mode: str
    tool_calls: list[ToolCallItem]
    llm_calls: int = 0
    simulated_model_calls: int = 0
    note: str = ""
    usage: dict[str, int] = Field(default_factory=dict)
    assistant_message: dict[str, Any] = Field(default_factory=dict, exclude=True)


class ToolExecutionResult(BaseModel):
    run_id: str = ""
    plan: ToolPlanResult
    results: list[dict[str, Any]]
    passed: bool
    mode: str
    final_answer: str = ""
    llm_calls: int = 0
    simulated_model_calls: int = 0
    steps: list[dict[str, Any]] = Field(default_factory=list)
    stop_reason: str = ""
    requested_tool_calls: int = 0
    executed_tool_calls: int = 0
    repeated_tool_calls: int = 0
    token_usage: dict[str, int] = Field(default_factory=dict)


class ToolCallProtocolError(ValueError):
    """Raised when a model emits a malformed or unauthorized tool call."""


def local_rule_plan(request: ToolPlanRequest) -> ToolPlanResult:
    text = request.prompt.lower()
    calls: list[ToolCallItem] = []
    data_path = request.data_path.strip()

    if not data_path or any(key in text for key in ("demo", "synthetic", "示例", "合成")):
        calls.append(
            ToolCallItem(
                tool_name="generate_demo_yield_data",
                args={"n_samples": 80, "random_state": 42, "overwrite": True},
                reason="No explicit data path or demo/synthetic data was requested.",
            )
        )
    if data_path:
        calls.append(
            ToolCallItem(
                tool_name="profile_yield_data",
                args={"data_path": data_path},
                reason="A data path was provided and should be profiled before evaluation.",
            )
        )
    elif any(key in text for key in ("profile", "画像", "schema", "数据")):
        calls.append(
            ToolCallItem(
                tool_name="profile_yield_data",
                args={"data_path": "agent_workspace/data/yield_synthetic/synthetic_yield_v1.csv"},
                reason="Prompt asks for data/profile; use the deterministic demo dataset path after generation.",
            )
        )
    if any(key in text for key in ("baseline", "evaluate", "评测", "评价", "基线", "run", "运行")):
        calls.append(
            ToolCallItem(
                tool_name="run_fixed_baseline_eval",
                args={"data_path": data_path or "agent_workspace/data/yield_synthetic/synthetic_yield_v1.csv", "n_splits": 3},
                reason="Prompt asks for evaluation/baseline.",
            )
        )
    if any(key in text for key in ("candidate", "候选", "benchmark", "策略评估", "方案评估")):
        calls.append(
            ToolCallItem(
                tool_name="run_candidate_benchmark",
                args={"data_path": data_path or "agent_workspace/data/yield_synthetic/synthetic_yield_v1.csv", "n_splits": 3},
                reason="Prompt asks for candidate strategy benchmark.",
            )
        )
    if any(key in text for key in ("report", "报告", "summary", "汇总")):
        calls.append(
            ToolCallItem(
                tool_name="build_run_report",
                args={"title": "YieldMind Requested Report", "notes": ["Planned from user request."]},
                reason="Prompt asks for a report/summary.",
            )
        )
    if any(key in text for key in ("knowledge", "rag", "文献", "证据", "引用", "检索")):
        calls.append(
            ToolCallItem(
                tool_name="search_knowledge",
                args={"query": request.prompt, "top_k": 5},
                reason="Prompt asks for evidence/knowledge retrieval.",
            )
        )
    if any(key in text for key in ("redact", "脱敏", "privacy", "secret", "api_key", "authorization", "敏感")):
        calls.append(
            ToolCallItem(
                tool_name="redact_sensitive_payload",
                args={"payload": {"input": request.prompt}},
                reason="Prompt asks for sensitive-data redaction or logging safety.",
            )
        )
    if any(key in text for key in ("budget", "token", "上下文", "预算", "压缩")):
        calls.append(
            ToolCallItem(
                tool_name="plan_token_budget",
                args={
                    "max_input_tokens": 1200,
                    "reserved_output_tokens": 300,
                    "sections": [
                        {"name": "constraints", "text": request.prompt, "required": True, "priority": 100},
                        {"name": "evidence", "text": "", "priority": 80},
                    ],
                },
                reason="Prompt asks for context/token-budget planning.",
            )
        )
    if any(key in text for key in ("validate", "校验", "验证")) and any(
        key in text for key in ("evidence", "证据", "引用", "chunk", "chunk_id", "text_hash", "index_version")
    ):
        calls.append(
            ToolCallItem(
                tool_name="validate_evidence_refs",
                args={"refs": [{"chunk_id": "placeholder_chunk_id", "text_hash": "placeholder_hash"}]},
                reason="Prompt asks to validate evidence references; concrete refs are filled after retrieval.",
            )
        )
    if any(key in text for key in ("sandbox", "隔离", "execute", "执行")):
        calls.append(
            ToolCallItem(
                tool_name="run_sandboxed_python",
                args={"argv": ["python", "-c", "print('yieldmind sandbox check')"], "timeout_seconds": 10},
                reason="Prompt asks for execution isolation.",
            )
        )

    deduped: list[ToolCallItem] = []
    seen = set()
    for call in calls:
        key = (call.tool_name, json.dumps(call.args, sort_keys=True, ensure_ascii=False))
        if key not in seen:
            deduped.append(call)
            seen.add(key)
    return ToolPlanResult(
        mode=OFFLINE_RULE_PLANNER,
        tool_calls=deduped[: request.max_tool_calls],
        llm_calls=0,
        simulated_model_calls=0,
        note="Deterministic local planner; this is not an LLM function-calling result.",
    )


def _tool_definitions(registry: ToolRegistry) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": item.name,
                "description": item.description,
                "parameters": item.input_schema,
            },
        }
        for item in registry.definitions()
    ]


def _tool_call_payload(raw: Any) -> dict[str, Any]:
    function = raw.function
    return {
        "id": str(getattr(raw, "id", "")),
        "type": "function",
        "function": {
            "name": str(function.name),
            "arguments": str(function.arguments or "{}"),
        },
    }


def _assistant_message_payload(message: Any) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": getattr(message, "content", None),
        "tool_calls": [_tool_call_payload(raw) for raw in (getattr(message, "tool_calls", None) or [])],
    }


def _initial_live_messages(request: ToolPlanRequest) -> list[dict[str, Any]]:
    return [
        {
            "role": "system",
            "content": (
                "Use YieldMind tools when they are needed to satisfy the user request. "
                "Do not invent tool names or repeat an identical tool call. Base the final answer on tool results, "
                "distinguish measured values from assumptions, and cite chunk_id and source_path for knowledge hits. "
                "Do not claim success for a failed tool or invent missing evidence."
            ),
        },
        {"role": "user", "content": request.prompt},
    ]


def _tool_call_fingerprint(call: ToolCallItem) -> str:
    canonical = json.dumps(
        {"tool_name": call.tool_name, "args": call.args},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _response_usage(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        raw = usage.model_dump()
    elif isinstance(usage, dict):
        raw = usage
    else:
        raw = {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0),
            "completion_tokens": getattr(usage, "completion_tokens", 0),
            "total_tokens": getattr(usage, "total_tokens", 0),
        }
    return {
        key: max(0, int(raw.get(key) or 0))
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    }


def _validate_session_turn(store: YieldMindStore, request: ToolPlanRequest) -> None:
    if request.turn_id and not request.session_id:
        raise ToolCallProtocolError("turn_id requires session_id.")
    if not request.session_id:
        return
    with connect(store.db_path) as conn:
        session = conn.execute(
            "SELECT session_id FROM yieldmind_sessions WHERE session_id=?",
            (request.session_id,),
        ).fetchone()
        if not session:
            raise ToolCallProtocolError(f"Unknown session_id: {request.session_id}")
        if request.turn_id:
            turn = conn.execute(
                "SELECT turn_id FROM yieldmind_turns WHERE turn_id=? AND session_id=?",
                (request.turn_id, request.session_id),
            ).fetchone()
            if not turn:
                raise ToolCallProtocolError("turn_id does not belong to session_id.")


def _parse_live_tool_calls(message: Any, registry: ToolRegistry, *, max_tool_calls: int) -> list[ToolCallItem]:
    raw_calls = list(getattr(message, "tool_calls", None) or [])
    if len(raw_calls) > max_tool_calls:
        raise ToolCallProtocolError(
            f"Model requested {len(raw_calls)} tools, exceeding max_tool_calls={max_tool_calls}."
        )
    calls: list[ToolCallItem] = []
    for position, raw in enumerate(raw_calls, start=1):
        function = raw.function
        name = str(function.name)
        call_id = str(getattr(raw, "id", "") or f"tool_call_{position}")
        try:
            args = json.loads(function.arguments or "{}")
        except json.JSONDecodeError as exc:
            raise ToolCallProtocolError(f"Tool {name!r} returned malformed JSON arguments: {exc.msg}.") from exc
        if not isinstance(args, dict):
            raise ToolCallProtocolError(f"Tool {name!r} arguments must decode to a JSON object.")
        try:
            parsed = registry.validate_call(name, args)
        except KeyError as exc:
            raise ToolCallProtocolError(str(exc)) from exc
        except ValidationError as exc:
            raise ToolCallProtocolError(f"Tool {name!r} arguments failed Pydantic validation: {exc}") from exc
        calls.append(
            ToolCallItem(
                tool_call_id=call_id,
                tool_name=name,
                args=parsed.model_dump(mode="json"),
                reason="Selected by live LLM function calling and validated by the Tool Registry.",
            )
        )
    return calls


def _live_llm_step(
    request: ToolPlanRequest,
    registry: ToolRegistry,
    *,
    messages: list[dict[str, Any]],
    max_tool_calls: int,
    tool_choice: str,
    client: Any | None = None,
) -> ToolPlanResult:
    if not request.allow_live_llm:
        raise PermissionError("Live function calling requires allow_live_llm=true.")

    from configs import AVAILABLE_LLMs
    from utils import get_client

    tools = _tool_definitions(registry)
    active_client = client or get_client(request.llm)
    response = active_client.chat.completions.create(
        model=AVAILABLE_LLMs[request.llm]["model"],
        messages=messages,
        tools=tools,
        tool_choice=tool_choice,
        temperature=0,
    )
    message = response.choices[0].message
    calls = _parse_live_tool_calls(message, registry, max_tool_calls=max_tool_calls)
    assistant_message = _assistant_message_payload(message)
    for payload, call in zip(assistant_message.get("tool_calls", []), calls):
        payload["id"] = call.tool_call_id
    return ToolPlanResult(
        mode=LIVE_LLM_FUNCTION_CALLING,
        tool_calls=calls,
        llm_calls=1,
        simulated_model_calls=0,
        note="Real model call; tool names and Pydantic arguments were validated before execution.",
        usage=_response_usage(response),
        assistant_message=assistant_message,
    )


def _live_llm_plan(request: ToolPlanRequest, registry: ToolRegistry, *, client: Any | None = None) -> ToolPlanResult:
    return _live_llm_step(
        request,
        registry,
        messages=_initial_live_messages(request),
        max_tool_calls=request.max_tool_calls,
        tool_choice="auto",
        client=client,
    )


def _as_simulated_step(result: ToolPlanResult) -> ToolPlanResult:
    result.mode = SIMULATED_TEST_ADAPTER
    result.llm_calls = 0
    result.simulated_model_calls = 1
    result.note = "Injected simulated test adapter; no real model API call."
    return result


def plan_tools(
    request: ToolPlanRequest,
    *,
    registry: ToolRegistry | None = None,
    client: Any | None = None,
    simulated_test_adapter: bool = False,
) -> ToolPlanResult:
    registry = registry or registry_for_workspace()
    if request.allow_live_llm:
        if simulated_test_adapter and client is None:
            raise ValueError("simulated_test_adapter requires an injected client.")
        result = _live_llm_plan(request, registry, client=client)
        if simulated_test_adapter:
            _as_simulated_step(result)
        return result
    return local_rule_plan(request)


def execute_plan(
    request: ToolPlanRequest,
    *,
    registry: ToolRegistry | None = None,
    store: YieldMindStore | None = None,
    run_id: str | None = None,
    client: Any | None = None,
    simulated_test_adapter: bool = False,
) -> ToolExecutionResult:
    store = store or YieldMindStore()
    registry = registry or registry_for_workspace(store=store)
    _validate_session_turn(store, request)
    active_run_id = run_id or store.create_run(
        mode=(
            SIMULATED_TEST_ADAPTER
            if simulated_test_adapter
            else LIVE_LLM_FUNCTION_CALLING
            if request.allow_live_llm
            else OFFLINE_RULE_PLANNER
        ),
        source="function_calling_adapter",
        status="running",
        metadata={
            "prompt": request.prompt,
            "allow_live_llm": request.allow_live_llm,
            "simulated_test_adapter": simulated_test_adapter,
            "max_steps": request.max_steps,
            "max_tool_calls": request.max_tool_calls,
            "max_repeated_tool_calls": request.max_repeated_tool_calls,
            "session_id": request.session_id,
            "turn_id": request.turn_id,
        },
    )
    if not request.allow_live_llm:
        plan = plan_tools(request, registry=registry, client=client, simulated_test_adapter=simulated_test_adapter)
        results: list[dict[str, Any]] = []
        for call in plan.tool_calls:
            started = time.time()
            result = registry.execute(
                call.tool_name,
                call.args,
                run_id=active_run_id,
                session_id=request.session_id,
                turn_id=request.turn_id,
            )
            payload = result.model_dump()
            payload.update(
                {
                    "tool_name": call.tool_name,
                    "reason": call.reason,
                    "duration_seconds": round(time.time() - started, 4),
                }
            )
            results.append(payload)
            store.add_event(
                active_run_id,
                stage="function_calling_tool",
                level="info" if result.ok else "error",
                message=f"{call.tool_name} {'passed' if result.ok else 'failed'} via {plan.mode}.",
                payload=redact_payload(RedactRequest(payload=payload)).payload,
            )
        passed = all(item.get("ok") for item in results) if results else True
        execution = ToolExecutionResult(
            run_id=active_run_id,
            plan=plan,
            results=results,
            passed=passed,
            mode=plan.mode,
            llm_calls=0,
            simulated_model_calls=0,
            steps=[
                {
                    "step_index": 1,
                    "outcome": "offline_tools_executed" if plan.tool_calls else "offline_no_tools",
                    "tool_names": [call.tool_name for call in plan.tool_calls],
                }
            ],
            stop_reason="offline_plan_complete",
            requested_tool_calls=len(plan.tool_calls),
            executed_tool_calls=len(plan.tool_calls),
        )
        store.update_run(
            active_run_id,
            status="passed" if passed else "failed",
            result=execution.model_dump(),
            completed=True,
        )
        return execution

    messages = _initial_live_messages(request)
    steps: list[dict[str, Any]] = []
    results = []
    result_by_fingerprint: dict[str, dict[str, Any]] = {}
    requested_tool_calls = 0
    executed_tool_calls = 0
    repeated_tool_calls = 0
    token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "reported_steps": 0}
    llm_calls = 0
    simulated_model_calls = 0
    final_answer = ""
    stop_reason = ""
    initial_plan: ToolPlanResult | None = None

    for step_index in range(1, request.max_steps + 1):
        remaining_tool_calls = request.max_tool_calls - requested_tool_calls
        tool_choice = "auto" if remaining_tool_calls > 0 else "none"
        if step_index == 1:
            step_plan = plan_tools(
                request,
                registry=registry,
                client=client,
                simulated_test_adapter=simulated_test_adapter,
            )
            initial_plan = step_plan
        else:
            step_plan = _live_llm_step(
                request,
                registry,
                messages=messages,
                max_tool_calls=max(0, remaining_tool_calls),
                tool_choice=tool_choice,
                client=client,
            )
            if simulated_test_adapter:
                _as_simulated_step(step_plan)
        llm_calls += step_plan.llm_calls
        simulated_model_calls += step_plan.simulated_model_calls
        if step_plan.usage:
            token_usage["reported_steps"] += 1
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                token_usage[key] += int(step_plan.usage.get(key, 0))
        messages.append(step_plan.assistant_message)

        assistant_content = str(step_plan.assistant_message.get("content") or "").strip()
        if not step_plan.tool_calls:
            if not assistant_content:
                raise ToolCallProtocolError("Live model returned neither tool calls nor a final answer.")
            final_answer = assistant_content
            stop_reason = "final_answer"
            step_trace = {
                "step_index": step_index,
                "outcome": "final_answer",
                "tool_choice": tool_choice,
                "tool_calls": [],
                "assistant_content": assistant_content,
                "usage": step_plan.usage,
            }
            steps.append(step_trace)
            store.add_event(
                active_run_id,
                stage="agent_loop_step",
                message=f"Agent loop step {step_index} produced the final answer.",
                payload=redact_payload(RedactRequest(payload=step_trace)).payload,
            )
            break

        requested_tool_calls += len(step_plan.tool_calls)
        call_traces: list[dict[str, Any]] = []
        duplicate_limit_exceeded = False
        for call in step_plan.tool_calls:
            fingerprint = _tool_call_fingerprint(call)
            duplicate = fingerprint in result_by_fingerprint
            if duplicate:
                repeated_tool_calls += 1
                payload = dict(result_by_fingerprint[fingerprint])
                payload["duplicate_reused"] = True
                payload["reason"] = "Identical tool call detected; reused the prior result without executing again."
                duplicate_limit_exceeded = repeated_tool_calls > request.max_repeated_tool_calls
            else:
                started = time.time()
                result = registry.execute(
                    call.tool_name,
                    call.args,
                    run_id=active_run_id,
                    session_id=request.session_id,
                    turn_id=request.turn_id,
                    idempotency_key=f"agent-loop:{active_run_id}:{fingerprint}",
                )
                executed_tool_calls += 1
                payload = result.model_dump()
                payload.update(
                    {
                        "tool_name": call.tool_name,
                        "reason": call.reason,
                        "duration_seconds": round(time.time() - started, 4),
                        "duplicate_reused": False,
                    }
                )
                result_by_fingerprint[fingerprint] = dict(payload)
                store.add_event(
                    active_run_id,
                    stage="function_calling_tool",
                    level="info" if result.ok else "error",
                    message=f"{call.tool_name} {'passed' if result.ok else 'failed'} at agent step {step_index}.",
                    payload=redact_payload(
                        RedactRequest(payload={**payload, "step_index": step_index, "fingerprint": fingerprint})
                    ).payload,
                )
            results.append(payload)
            call_traces.append(
                {
                    "tool_call_id": call.tool_call_id,
                    "tool_name": call.tool_name,
                    "fingerprint": fingerprint,
                    "duplicate_reused": duplicate,
                    "ok": bool(payload.get("ok")),
                }
            )
            safe_result = redact_payload(RedactRequest(payload=payload)).payload
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.tool_call_id,
                    "name": call.tool_name,
                    "content": json.dumps(safe_result, ensure_ascii=False),
                }
            )

        step_trace = {
            "step_index": step_index,
            "outcome": "duplicate_limit_exceeded" if duplicate_limit_exceeded else "tools_returned",
            "tool_choice": tool_choice,
            "tool_calls": call_traces,
            "requested_tool_calls_total": requested_tool_calls,
            "executed_tool_calls_total": executed_tool_calls,
            "repeated_tool_calls_total": repeated_tool_calls,
            "usage": step_plan.usage,
        }
        steps.append(step_trace)
        store.add_event(
            active_run_id,
            stage="agent_loop_step",
            level="error" if duplicate_limit_exceeded else "info",
            message=(
                f"Agent loop stopped at step {step_index}: repeated tool-call limit exceeded."
                if duplicate_limit_exceeded
                else f"Agent loop step {step_index} returned {len(step_plan.tool_calls)} tool result(s)."
            ),
            payload=step_trace,
        )
        if duplicate_limit_exceeded:
            stop_reason = "repeated_tool_call_limit"
            break
    else:
        stop_reason = "max_steps_exhausted"

    if initial_plan is None:
        raise RuntimeError("Agent loop did not execute an initial planning step.")
    passed = bool(final_answer) and all(item.get("ok") for item in results)
    execution = ToolExecutionResult(
        run_id=active_run_id,
        plan=initial_plan,
        results=results,
        passed=passed,
        mode=initial_plan.mode,
        final_answer=final_answer,
        llm_calls=llm_calls,
        simulated_model_calls=simulated_model_calls,
        steps=steps,
        stop_reason=stop_reason,
        requested_tool_calls=requested_tool_calls,
        executed_tool_calls=executed_tool_calls,
        repeated_tool_calls=repeated_tool_calls,
        token_usage=token_usage,
    )
    persisted_execution = redact_payload(RedactRequest(payload=execution.model_dump())).payload
    store.update_run(
        active_run_id,
        status="passed" if passed else "failed",
        result=persisted_execution,
        completed=True,
    )
    return execution
