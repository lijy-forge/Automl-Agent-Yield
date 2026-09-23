"""Provider-neutral accounting for chat-model calls in the domain workflow."""

from __future__ import annotations

import time
import uuid
from typing import Any


def normalize_token_usage(raw: Any) -> dict[str, int | None]:
    """Normalize OpenAI-compatible and SDK-specific usage payloads."""
    if raw is None:
        payload: dict[str, Any] = {}
    elif isinstance(raw, dict):
        payload = raw
    elif hasattr(raw, "to_dict"):
        try:
            payload = raw.to_dict(mode="json")
        except TypeError:
            payload = raw.to_dict()
    elif hasattr(raw, "model_dump"):
        payload = raw.model_dump()
    else:
        payload = {}

    def _integer(*keys: str) -> int | None:
        for key in keys:
            value = payload.get(key)
            if value is not None:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    pass
        return None

    input_tokens = _integer("input_tokens", "prompt_tokens")
    output_tokens = _integer("output_tokens", "completion_tokens")
    total_tokens = _integer("total_tokens")
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def build_model_usage_record(
    *,
    agent: str,
    stage: str,
    purpose: str,
    provider: str,
    model: str,
    usage: Any = None,
    attempt: int = 1,
    duration_ms: float | None = None,
    status: str = "completed",
    error: str = "",
) -> dict[str, Any]:
    tokens = normalize_token_usage(usage)
    return {
        "call_id": f"llm_{uuid.uuid4().hex[:16]}",
        "agent": agent,
        "stage": stage,
        "purpose": purpose,
        "provider": provider,
        "model": model,
        "attempt": max(1, int(attempt)),
        **tokens,
        "usage_reported": tokens["total_tokens"] is not None,
        "duration_ms": round(float(duration_ms), 3) if duration_ms is not None else None,
        "status": status,
        "error": str(error or ""),
        "ts": time.time(),
    }


def summarize_model_usage(records: list[dict[str, Any]] | None) -> dict[str, Any]:
    rows = [row for row in (records or []) if isinstance(row, dict)]
    reported = [row for row in rows if row.get("usage_reported")]

    def _sum(key: str) -> int:
        return sum(int(row.get(key) or 0) for row in reported)

    per_agent: dict[str, dict[str, int]] = {}
    for row in rows:
        agent = str(row.get("agent") or "UnknownAgent")
        bucket = per_agent.setdefault(
            agent,
            {"calls": 0, "reported_calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        )
        bucket["calls"] += 1
        if row.get("usage_reported"):
            bucket["reported_calls"] += 1
            for key in ("input_tokens", "output_tokens", "total_tokens"):
                bucket[key] += int(row.get(key) or 0)

    if not rows:
        mode = "no_calls"
    elif len(reported) == len(rows):
        mode = "live_metered"
    elif reported:
        mode = "live_partial"
    else:
        mode = "live_unmetered"
    return {
        "mode": mode,
        "calls_total": len(rows),
        "calls_reported": len(reported),
        "calls_unreported": len(rows) - len(reported),
        "input_tokens": _sum("input_tokens"),
        "output_tokens": _sum("output_tokens"),
        "total_tokens": _sum("total_tokens"),
        "per_agent": per_agent,
    }


def merge_model_usage_records(*groups: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for group in groups:
        for row in group or []:
            if not isinstance(row, dict):
                continue
            call_id = str(row.get("call_id") or "")
            if call_id and call_id in seen:
                continue
            if call_id:
                seen.add(call_id)
            merged.append(dict(row))
    return merged
