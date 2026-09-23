"""Policy-controlled execution lifecycle for registered YieldMind tools."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from yieldmind.database import YieldMindStore
from yieldmind.safety import RedactRequest, redact_payload
from yieldmind.tools import OFFLINE_MODE, ToolRegistry, ToolResult, ToolSpec


class ExecutionContext(BaseModel):
    actor_id: str = "local_user"
    caller: Literal["api", "agent_loop", "workflow", "evaluation", "system", "legacy_registry"] = "system"
    execution_mode: str = OFFLINE_MODE
    run_id: str = ""
    session_id: str = ""
    turn_id: str = ""
    grants: set[str] = Field(default_factory=set)
    allowed_read_roots: list[str] = Field(default_factory=list)
    allowed_write_roots: list[str] = Field(default_factory=list)


class ToolErrorDetail(BaseModel):
    code: str
    phase: Literal["lookup", "parsing", "validation", "policy", "idempotency", "execution"]
    tool_name: str
    message: str
    repairable: bool = False
    retryable: bool = False
    field_errors: list[dict[str, Any]] = Field(default_factory=list)
    correction_hint: str = ""


class PolicyDecision(BaseModel):
    allowed: bool
    risk_level: str
    execution_backend: str
    required_grants: list[str] = Field(default_factory=list)
    missing_grants: list[str] = Field(default_factory=list)
    reason: str


class PreparedToolCall(BaseModel):
    tool_name: str
    parsed_args: Any

    model_config = {"arbitrary_types_allowed": True}


class ToolPolicy:
    """Small capability policy; intentionally not a full RBAC system."""

    _ALLOWED_CALLERS = {"api", "agent_loop", "workflow", "evaluation", "system", "legacy_registry"}

    def authorize(
        self,
        spec: ToolSpec,
        context: ExecutionContext,
        parsed_args: BaseModel,
    ) -> PolicyDecision:
        missing = sorted(set(spec.required_grants) - set(context.grants))
        if context.caller not in self._ALLOWED_CALLERS:
            return PolicyDecision(
                allowed=False,
                risk_level=spec.risk_level,
                execution_backend=spec.execution_backend,
                required_grants=list(spec.required_grants),
                reason=f"Caller {context.caller!r} is not permitted to execute tools.",
            )
        if spec.risk_level == "high" and spec.execution_backend not in {"subprocess", "docker"}:
            return PolicyDecision(
                allowed=False,
                risk_level=spec.risk_level,
                execution_backend=spec.execution_backend,
                required_grants=list(spec.required_grants),
                reason="High-risk tools must use the subprocess or docker backend.",
            )
        if missing:
            return PolicyDecision(
                allowed=False,
                risk_level=spec.risk_level,
                execution_backend=spec.execution_backend,
                required_grants=list(spec.required_grants),
                missing_grants=missing,
                reason=f"Missing explicit execution grants: {', '.join(missing)}.",
            )
        return PolicyDecision(
            allowed=True,
            risk_level=spec.risk_level,
            execution_backend=spec.execution_backend,
            required_grants=list(spec.required_grants),
            reason="Tool call is authorized by the configured risk policy.",
        )


def _validation_detail(name: str, exc: ValidationError) -> ToolErrorDetail:
    fields: list[dict[str, Any]] = []
    for item in exc.errors(include_url=False):
        fields.append(
            {
                "location": [str(part) for part in item.get("loc", ())],
                "message": str(item.get("msg") or "Invalid value."),
                "type": str(item.get("type") or "validation_error"),
                "input": item.get("input"),
                "constraints": dict(item.get("ctx") or {}),
            }
        )
    first = fields[0] if fields else {}
    location = ".".join(first.get("location") or []) or "arguments"
    hint = f"Correct {location}: {first.get('message', 'invalid value')}. Keep the same tool unless the tool name is invalid."
    return ToolErrorDetail(
        code="argument_validation_error",
        phase="validation",
        tool_name=name,
        message=f"Tool {name!r} arguments failed Pydantic validation.",
        repairable=True,
        field_errors=fields,
        correction_hint=hint,
    )


class ToolExecutor:
    def __init__(
        self,
        registry: ToolRegistry,
        *,
        store: YieldMindStore | None = None,
        policy: ToolPolicy | None = None,
    ) -> None:
        self.registry = registry
        self.store = store if store is not None else registry.store
        self.policy = policy or ToolPolicy()

    def prepare(self, name: str, args: dict[str, Any] | None = None) -> tuple[PreparedToolCall | None, ToolErrorDetail | None]:
        try:
            spec = self.registry.get_spec(name)
        except KeyError:
            allowed = ", ".join(self.registry.names())
            return None, ToolErrorDetail(
                code="unknown_tool",
                phase="lookup",
                tool_name=name,
                message=f"Unknown tool: {name}",
                repairable=True,
                correction_hint=f"Choose one registered tool: {allowed}.",
            )
        try:
            parsed = spec.args_model.model_validate(args or {})
        except ValidationError as exc:
            return None, _validation_detail(name, exc)
        return PreparedToolCall(tool_name=name, parsed_args=parsed), None

    @staticmethod
    def _audit(spec: ToolSpec, context: ExecutionContext, decision: PolicyDecision | None = None) -> dict[str, Any]:
        return {
            "risk_level": spec.risk_level,
            "execution_backend": spec.execution_backend,
            "caller": context.caller,
            "actor_id": context.actor_id,
            "grants": sorted(context.grants),
            "retry_safe": spec.retry_safe,
            "max_transient_retries": spec.max_transient_retries,
            "policy": decision.model_dump(mode="json") if decision is not None else {},
        }

    @staticmethod
    def _exception_is_transient(exc: Exception) -> bool:
        if isinstance(exc, (TimeoutError, ConnectionError)):
            return True
        return type(exc).__name__ in {
            "ConnectError",
            "ConnectTimeout",
            "ReadTimeout",
            "RemoteProtocolError",
            "ServiceUnavailableError",
        }

    def _record_rejected(
        self,
        *,
        name: str,
        args: dict[str, Any],
        context: ExecutionContext,
        detail: ToolErrorDetail,
        status: str,
        audit: dict[str, Any],
    ) -> ToolResult:
        result = ToolResult(
            ok=False,
            mode=context.execution_mode,
            error=detail.message,
            error_detail=detail.model_dump(mode="json"),
            audit=audit,
        )
        if self.store is not None:
            safe_args = redact_payload(RedactRequest(payload=args)).payload
            safe_result = redact_payload(RedactRequest(payload=result.model_dump(mode="json"))).payload
            self.store.record_tool_call(
                tool_name=name,
                mode=context.execution_mode,
                status=status,
                args=safe_args,
                result=safe_result,
                error=detail.message,
                run_id=context.run_id or None,
                session_id=context.session_id or None,
                turn_id=context.turn_id or None,
            )
        return result

    def execute(
        self,
        name: str,
        args: dict[str, Any] | None = None,
        *,
        context: ExecutionContext | None = None,
        idempotency_key: str = "",
    ) -> ToolResult:
        context = context or ExecutionContext()
        raw_args = args or {}
        prepared, error = self.prepare(name, raw_args)
        if error is not None or prepared is None:
            return self._record_rejected(
                name=name,
                args=raw_args,
                context=context,
                detail=error or ToolErrorDetail(
                    code="invalid_tool_call",
                    phase="validation",
                    tool_name=name,
                    message="Tool call could not be prepared.",
                ),
                status="invalid",
                audit={"caller": context.caller, "actor_id": context.actor_id},
            )

        spec = self.registry.get_spec(name)
        decision = self.policy.authorize(spec, context, prepared.parsed_args)
        audit = self._audit(spec, context, decision)
        if not decision.allowed:
            detail = ToolErrorDetail(
                code="policy_denied",
                phase="policy",
                tool_name=name,
                message=decision.reason,
                repairable=False,
                correction_hint="Obtain explicit authorization outside the model-generated tool arguments.",
            )
            return self._record_rejected(
                name=name,
                args=raw_args,
                context=context,
                detail=detail,
                status="denied",
                audit=audit,
            )

        started = time.time()
        payload_args = redact_payload(RedactRequest(payload=prepared.parsed_args.model_dump(mode="json"))).payload
        claimed_call: dict[str, Any] | None = None
        if self.store is not None and idempotency_key:
            claimed_call, existed = self.store.claim_tool_call(
                tool_name=name,
                mode=context.execution_mode,
                args=payload_args,
                idempotency_key=idempotency_key,
                run_id=context.run_id or None,
                session_id=context.session_id or None,
                turn_id=context.turn_id or None,
                started_at=started,
            )
            if existed:
                previous_result = claimed_call.get("result") or {}
                if claimed_call.get("status") in {"passed", "failed"} and previous_result:
                    return ToolResult.model_validate(previous_result)
                detail = ToolErrorDetail(
                    code="idempotency_conflict",
                    phase="idempotency",
                    tool_name=name,
                    message=(
                        "An earlier execution with this idempotency key is still running or has an ambiguous "
                        "outcome; refusing automatic replay."
                    ),
                    retryable=False,
                )
                return ToolResult(
                    ok=False,
                    mode=context.execution_mode,
                    error=detail.message,
                    error_detail=detail.model_dump(mode="json"),
                    audit=audit,
                )
        retry_enabled = bool(idempotency_key and spec.retry_safe and spec.max_transient_retries > 0)
        attempts = 0
        retry_delays: list[float] = []
        while True:
            attempts += 1
            try:
                result = spec.handler(prepared.parsed_args)
            except Exception as exc:
                retryable = self._exception_is_transient(exc)
                detail = ToolErrorDetail(
                    code="transient_handler_exception" if retryable else "handler_exception",
                    phase="execution",
                    tool_name=name,
                    message=f"{type(exc).__name__}: {exc}",
                    retryable=retryable,
                )
                result = ToolResult(
                    ok=False,
                    mode=context.execution_mode,
                    error=detail.message,
                    error_detail=detail.model_dump(mode="json"),
                )
            retryable_result = bool((result.error_detail or {}).get("retryable"))
            retries_used = attempts - 1
            if result.ok or not retryable_result or not retry_enabled or retries_used >= spec.max_transient_retries:
                break
            delay = min(max(0.0, spec.retry_backoff_seconds) * (2**retries_used), 2.0)
            retry_delays.append(delay)
            if delay:
                time.sleep(delay)
        audit["execution_attempts"] = attempts
        audit["transient_retries"] = attempts - 1
        audit["retry_delays_seconds"] = retry_delays
        if not retry_enabled and bool((result.error_detail or {}).get("retryable")):
            audit["retry_suppressed_reason"] = (
                "missing_idempotency_key" if not idempotency_key else "tool_not_declared_retry_safe"
            )
        result.audit = audit
        status = "passed" if result.ok else "failed"
        payload_result = redact_payload(RedactRequest(payload=result.model_dump(mode="json"))).payload
        if self.store is not None:
            completed = time.time()
            if claimed_call is not None:
                self.store.complete_tool_call(
                    str(claimed_call["call_id"]),
                    status=status,
                    mode=result.mode,
                    result=payload_result,
                    error=result.error,
                    completed_at=completed,
                )
            else:
                self.store.record_tool_call(
                    tool_name=name,
                    mode=result.mode,
                    status=status,
                    args=payload_args,
                    result=payload_result,
                    error=result.error,
                    run_id=context.run_id or None,
                    session_id=context.session_id or None,
                    turn_id=context.turn_id or None,
                    started_at=started,
                    completed_at=completed,
                )
        return result


_MODEL_RESULT_PRIORITY_KEYS = (
    "status",
    "stage",
    "summary",
    "message",
    "metrics",
    "best_baseline",
    "champion",
    "selected_strategy_id",
    "final_selection",
    "counts",
    "evidence_validation",
    "selected_chunk_ids",
    "hits",
    "errors",
    "warnings",
)


def _compact_value(value: Any, *, depth: int = 0) -> Any:
    if depth >= 4:
        text = json.dumps(value, ensure_ascii=False, default=str)
        return text if len(text) <= 800 else text[:797] + "..."
    if isinstance(value, str):
        return value if len(value) <= 1200 else value[:1197] + "..."
    if isinstance(value, list):
        limit = 5
        items = [_compact_value(item, depth=depth + 1) for item in value[:limit]]
        if len(value) > limit:
            items.append({"_omitted_items": len(value) - limit})
        return items
    if isinstance(value, dict):
        ordered = [key for key in _MODEL_RESULT_PRIORITY_KEYS if key in value]
        ordered.extend(key for key in value if key not in ordered)
        limit = 30 if depth == 0 else 18
        selected = ordered[:limit]
        compacted = {str(key): _compact_value(value[key], depth=depth + 1) for key in selected}
        if len(ordered) > limit:
            compacted["_omitted_fields"] = [str(key) for key in ordered[limit:]]
        return compacted
    return value


def compact_tool_result_for_model(
    result: ToolResult | dict[str, Any],
    *,
    max_chars: int = 12000,
) -> dict[str, Any]:
    """Return a bounded tool-message payload while leaving persisted results intact."""
    limit = max(1000, int(max_chars))
    payload = result.model_dump(mode="json") if isinstance(result, ToolResult) else dict(result)
    safe_payload = redact_payload(RedactRequest(payload=payload)).payload
    serialized = json.dumps(safe_payload, ensure_ascii=False, default=str)
    if len(serialized) <= limit:
        return safe_payload

    compacted = _compact_value(safe_payload)
    compacted["result_truncated"] = True
    compacted["original_result_chars"] = len(serialized)
    compacted["model_result_budget_chars"] = limit
    compacted["full_result_artifacts"] = dict(safe_payload.get("artifacts") or {})
    compacted_serialized = json.dumps(compacted, ensure_ascii=False, default=str)
    if len(compacted_serialized) <= limit:
        return compacted

    artifact_items = list(dict(safe_payload.get("artifacts") or {}).items())[:5]
    compact_artifacts = {
        str(key): (str(value) if len(str(value)) <= 320 else str(value)[:317] + "...")
        for key, value in artifact_items
    }
    raw_error_detail = dict(safe_payload.get("error_detail") or {})
    fixed = {
        "ok": bool(safe_payload.get("ok")),
        "mode": safe_payload.get("mode"),
        "error": str(safe_payload.get("error") or "")[:400],
        "error_detail": {
            key: _compact_value(raw_error_detail[key])
            for key in ("code", "phase", "message", "correction_hint")
            if key in raw_error_detail
        },
        "artifacts": compact_artifacts,
        "result_truncated": True,
        "original_result_chars": len(serialized),
        "model_result_budget_chars": limit,
    }
    if len(json.dumps(fixed, ensure_ascii=False, default=str)) >= limit:
        fixed = {
            "ok": bool(safe_payload.get("ok")),
            "mode": safe_payload.get("mode"),
            "error": str(safe_payload.get("error") or "")[:160],
            "artifact_paths": [str(value)[:160] for _, value in artifact_items[:2]],
            "result_truncated": True,
            "original_result_chars": len(serialized),
            "model_result_budget_chars": limit,
        }
    preview = json.dumps(compacted.get("result") or {}, ensure_ascii=False, default=str)
    preview_budget = max(0, limit - len(json.dumps(fixed, ensure_ascii=False, default=str)) - 24)
    fixed["result_preview"] = preview[:preview_budget]
    while len(json.dumps(fixed, ensure_ascii=False, default=str)) > limit and fixed["result_preview"]:
        overflow = len(json.dumps(fixed, ensure_ascii=False, default=str)) - limit
        fixed["result_preview"] = fixed["result_preview"][: max(0, len(fixed["result_preview"]) - overflow - 1)]
    return fixed
