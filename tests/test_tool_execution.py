from __future__ import annotations

import json
import sys
from types import SimpleNamespace

from pydantic import BaseModel

from yieldmind.database import YieldMindStore, connect
from yieldmind.function_calling import ToolPlanRequest, execute_plan
from yieldmind.tool_execution import ExecutionContext, ToolExecutor, compact_tool_result_for_model
from yieldmind.tools import ToolRegistry, ToolResult, ToolSpec


class _NoArgs(BaseModel):
    pass


def _tool_message(name: str, arguments: str, *, call_id: str = "call_1") -> SimpleNamespace:
    call = SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=arguments))
    return SimpleNamespace(content=None, tool_calls=[call])


def _multi_tool_message(calls: list[tuple[str, str, str]]) -> SimpleNamespace:
    return SimpleNamespace(
        content=None,
        tool_calls=[
            SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=arguments))
            for call_id, name, arguments in calls
        ],
    )


class _FakeCompletions:
    def __init__(self, messages: list[SimpleNamespace]) -> None:
        self.messages = list(messages)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=self.messages.pop(0))],
            usage=None,
        )


def _client(messages: list[SimpleNamespace]) -> SimpleNamespace:
    return SimpleNamespace(chat=SimpleNamespace(completions=_FakeCompletions(messages)))


def test_executor_returns_structured_validation_error_without_running_handler(tmp_path) -> None:
    store = YieldMindStore(tmp_path / "yieldmind.sqlite3")
    executor = ToolExecutor(ToolRegistry(store=store), store=store)

    result = executor.execute(
        "run_sandboxed_python",
        {"argv": [sys.executable, "-c", "print('must-not-run')"], "timeout_seconds": 1000},
        context=ExecutionContext(caller="api", grants={"subprocess_execute"}),
    )

    assert result.ok is False
    assert result.error_detail["code"] == "argument_validation_error"
    assert result.error_detail["repairable"] is True
    assert result.error_detail["field_errors"][0]["location"] == ["timeout_seconds"]
    with connect(store.db_path) as conn:
        row = conn.execute("SELECT status FROM yieldmind_tool_calls").fetchone()
    assert row["status"] == "invalid"


def test_high_risk_tool_requires_trusted_context_grant(tmp_path) -> None:
    store = YieldMindStore(tmp_path / "yieldmind.sqlite3")
    registry = ToolRegistry(store=store)
    executor = ToolExecutor(registry, store=store)
    args = {"argv": [sys.executable, "-c", "print('authorized')"], "timeout_seconds": 5}

    denied = executor.execute(
        "run_sandboxed_python",
        args,
        context=ExecutionContext(caller="agent_loop"),
    )
    allowed = executor.execute(
        "run_sandboxed_python",
        args,
        context=ExecutionContext(caller="agent_loop", grants={"subprocess_execute"}),
    )

    assert denied.ok is False
    assert denied.error_detail["code"] == "policy_denied"
    assert denied.audit["policy"]["missing_grants"] == ["subprocess_execute"]
    assert allowed.ok is True
    assert allowed.audit["execution_backend"] == "subprocess"
    assert "authorized" in allowed.result["stdout"]


def test_model_visible_allow_docker_argument_does_not_authorize_execution(tmp_path) -> None:
    store = YieldMindStore(tmp_path / "yieldmind.sqlite3")
    executor = ToolExecutor(ToolRegistry(store=store), store=store)
    result = executor.execute(
        "run_docker_sandboxed_python",
        {"allow_docker": True},
        context=ExecutionContext(caller="agent_loop"),
    )

    assert result.ok is False
    assert result.error_detail["code"] == "policy_denied"
    assert result.audit["policy"]["missing_grants"] == ["docker_execute"]


def test_agent_loop_repairs_invalid_arguments_once_then_executes(tmp_path) -> None:
    invalid = _tool_message(
        "run_sandboxed_python",
        json.dumps({"argv": [sys.executable, "-c", "print('repaired')"], "timeout_seconds": 1000}),
        call_id="invalid_call",
    )
    corrected = _tool_message(
        "run_sandboxed_python",
        json.dumps({"argv": [sys.executable, "-c", "print('repaired')"], "timeout_seconds": 5}),
        call_id="corrected_call",
    )
    final = SimpleNamespace(content="The corrected tool call completed.", tool_calls=[])
    client = _client([invalid, corrected, final])
    store = YieldMindStore(tmp_path / "yieldmind.sqlite3")

    result = execute_plan(
        ToolPlanRequest(
            prompt="Run the bounded command.",
            allow_live_llm=True,
            allow_subprocess=True,
            max_steps=4,
        ),
        registry=ToolRegistry(store=store),
        store=store,
        client=client,
        simulated_test_adapter=True,
    )

    assert result.passed is True
    assert result.argument_repair_attempts == 1
    assert result.executed_tool_calls == 1
    assert result.requested_tool_calls == 1
    assert [step["outcome"] for step in result.steps] == [
        "argument_repair_requested",
        "tools_returned",
        "final_answer",
    ]
    repair_messages = [
        message for message in client.chat.completions.calls[1]["messages"]
        if message["role"] == "tool" and message["tool_call_id"] == "invalid_call"
    ]
    assert repair_messages
    assert "timeout_seconds" in repair_messages[-1]["content"]


def test_agent_loop_stops_after_second_invalid_argument_batch(tmp_path) -> None:
    invalid_args = json.dumps({"argv": [sys.executable, "-c", "print('never')"], "timeout_seconds": 1000})
    client = _client(
        [
            _tool_message("run_sandboxed_python", invalid_args, call_id="bad_1"),
            _tool_message("run_sandboxed_python", invalid_args, call_id="bad_2"),
        ]
    )
    store = YieldMindStore(tmp_path / "yieldmind.sqlite3")

    result = execute_plan(
        ToolPlanRequest(
            prompt="Keep returning invalid arguments.",
            allow_live_llm=True,
            allow_subprocess=True,
            max_steps=3,
        ),
        registry=ToolRegistry(store=store),
        store=store,
        client=client,
        simulated_test_adapter=True,
    )

    assert result.passed is False
    assert result.stop_reason == "tool_argument_repair_exhausted"
    assert result.argument_repair_attempts == 2
    assert result.executed_tool_calls == 0
    with connect(store.db_path) as conn:
        count = conn.execute("SELECT COUNT(*) AS count FROM yieldmind_tool_calls").fetchone()["count"]
    assert count == 0


def test_invalid_mixed_batch_has_no_partial_execution(tmp_path) -> None:
    mixed = _multi_tool_message(
        [
            ("valid_call", "redact_sensitive_payload", '{"payload":{"value":"safe"}}'),
            ("invalid_call", "search_knowledge", "{}"),
        ]
    )
    final = SimpleNamespace(content="I stopped after the validation feedback.", tool_calls=[])
    client = _client([mixed, final])
    store = YieldMindStore(tmp_path / "yieldmind.sqlite3")

    result = execute_plan(
        ToolPlanRequest(prompt="Test an atomic tool batch.", allow_live_llm=True, max_steps=3),
        registry=ToolRegistry(store=store),
        store=store,
        client=client,
        simulated_test_adapter=True,
    )

    assert result.passed is True
    assert result.executed_tool_calls == 0
    assert result.steps[0]["outcome"] == "argument_repair_requested"
    tool_messages = [message for message in client.chat.completions.calls[1]["messages"] if message["role"] == "tool"]
    assert len(tool_messages) == 2
    assert any("batch_rejected_due_to_invalid_peer" in message["content"] for message in tool_messages)


def test_model_tool_result_is_bounded_but_keeps_artifact_pointer() -> None:
    result = ToolResult(
        ok=True,
        result={"rows": [{"text": "x" * 4000} for _ in range(20)], "api_key": "secret-value"},
        artifacts={"full_result": "agent_workspace/runs/run_1/artifacts/full_result.json"},
    )

    compacted = compact_tool_result_for_model(result, max_chars=1800)
    serialized = json.dumps(compacted, ensure_ascii=False)

    assert len(serialized) <= 1800
    assert compacted["result_truncated"] is True
    assert compacted["original_result_chars"] > 1800
    assert "agent_workspace/runs/run_1/artifacts/full_result.json" in serialized
    assert "secret-value" not in serialized


def test_agent_loop_only_compacts_model_message_not_execution_result(tmp_path) -> None:
    long_value = "bounded-result-marker-" + ("x" * 6000)
    tool_call = _tool_message(
        "redact_sensitive_payload",
        json.dumps({"payload": {"large_text": long_value}}),
        call_id="large_result_call",
    )
    final = SimpleNamespace(content="The bounded result was received.", tool_calls=[])
    client = _client([tool_call, final])
    store = YieldMindStore(tmp_path / "yieldmind.sqlite3")

    result = execute_plan(
        ToolPlanRequest(
            prompt="Return a large tool result.",
            allow_live_llm=True,
            max_steps=3,
            max_tool_result_chars=1400,
        ),
        registry=ToolRegistry(store=store),
        store=store,
        client=client,
        simulated_test_adapter=True,
    )

    tool_messages = [
        message
        for message in client.chat.completions.calls[1]["messages"]
        if message["role"] == "tool" and message["tool_call_id"] == "large_result_call"
    ]
    model_payload = json.loads(tool_messages[-1]["content"])
    assert len(tool_messages[-1]["content"]) <= 1400
    assert model_payload["result_truncated"] is True
    assert len(json.dumps(result.results[0], ensure_ascii=False)) > 5000
    assert result.steps[0]["tool_calls"][0]["model_result_truncated"] is True


def test_retry_safe_tool_retries_only_with_idempotency_key(tmp_path) -> None:
    store = YieldMindStore(tmp_path / "yieldmind.sqlite3")
    registry = ToolRegistry(store=store)
    attempts = {"count": 0}

    def flaky_handler(_args: BaseModel) -> ToolResult:
        attempts["count"] += 1
        if attempts["count"] < 3:
            return ToolResult(
                ok=False,
                error="temporary dependency failure",
                error_detail={"code": "temporary_dependency", "retryable": True},
            )
        return ToolResult(ok=True, result={"attempt": attempts["count"]})

    registry._register(
        ToolSpec(
            name="test_retry_safe",
            description="Synthetic retry-safe test tool.",
            args_model=_NoArgs,
            handler=flaky_handler,
            retry_safe=True,
            max_transient_retries=2,
            retry_backoff_seconds=0,
        )
    )
    executor = ToolExecutor(registry, store=store)

    result = executor.execute(
        "test_retry_safe",
        {},
        context=ExecutionContext(caller="agent_loop"),
        idempotency_key="retry-safe-call-1",
    )

    assert result.ok is True
    assert attempts["count"] == 3
    assert result.audit["execution_attempts"] == 3
    assert result.audit["transient_retries"] == 2
    with connect(store.db_path) as conn:
        rows = conn.execute("SELECT status FROM yieldmind_tool_calls").fetchall()
    assert [row["status"] for row in rows] == ["passed"]


def test_retry_is_suppressed_for_non_idempotent_or_non_transient_failure(tmp_path) -> None:
    store = YieldMindStore(tmp_path / "yieldmind.sqlite3")
    registry = ToolRegistry(store=store)
    transient_attempts = {"count": 0}
    permanent_attempts = {"count": 0}

    def transient_handler(_args: BaseModel) -> ToolResult:
        transient_attempts["count"] += 1
        return ToolResult(ok=False, error="temporary", error_detail={"retryable": True})

    def permanent_handler(_args: BaseModel) -> ToolResult:
        permanent_attempts["count"] += 1
        raise RuntimeError("programming error")

    for name, handler in (("test_no_key", transient_handler), ("test_permanent", permanent_handler)):
        registry._register(
            ToolSpec(
                name=name,
                description="Synthetic retry boundary test tool.",
                args_model=_NoArgs,
                handler=handler,
                retry_safe=True,
                max_transient_retries=2,
                retry_backoff_seconds=0,
            )
        )
    executor = ToolExecutor(registry, store=store)

    no_key = executor.execute("test_no_key", {}, context=ExecutionContext(caller="agent_loop"))
    permanent = executor.execute(
        "test_permanent",
        {},
        context=ExecutionContext(caller="agent_loop"),
        idempotency_key="permanent-call-1",
    )

    assert transient_attempts["count"] == 1
    assert no_key.audit["retry_suppressed_reason"] == "missing_idempotency_key"
    assert permanent_attempts["count"] == 1
    assert permanent.error_detail["retryable"] is False
    assert permanent.audit["transient_retries"] == 0
