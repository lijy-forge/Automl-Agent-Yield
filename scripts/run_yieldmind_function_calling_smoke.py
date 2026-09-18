#!/usr/bin/env python3
"""Run an explicit YieldMind function-calling smoke check.

By default this script does not call an LLM. It writes a skipped report with an
offline planner preview. Use --allow-live-llm to make one real model call.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs import AVAILABLE_LLMs
from yieldmind.database import YieldMindStore, json_dumps
from yieldmind.function_calling import LIVE_LLM_FUNCTION_CALLING, ToolPlanRequest, local_rule_plan, plan_tools
from yieldmind.safety import RedactRequest, redact_payload
from yieldmind.tools import registry_for_workspace


DEFAULT_PROMPT = (
    "请根据需求选择 YieldMind 工具：检索屈服应力证据，控制上下文 token budget，"
    "并校验证据引用。只返回需要调用的工具。"
)


def _safe_llm_config(llm: str) -> dict[str, str]:
    cfg = AVAILABLE_LLMs.get(llm, {})
    base_url = str(cfg.get("base_url", ""))
    parsed = urlparse(base_url)
    return {
        "llm": llm,
        "model": str(cfg.get("model", "")),
        "base_url_host": parsed.netloc or "",
        "has_api_key": str(cfg.get("api_key", "")) != "",
    }


def _write_report(out_dir: Path, payload: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"yieldmind_function_calling_smoke_{time.strftime('%Y%m%d_%H%M%S')}.json"
    safe_payload = redact_payload(RedactRequest(payload=payload)).payload
    path.write_text(json_dumps(safe_payload), encoding="utf-8")
    return path


def run_smoke(
    *,
    prompt: str,
    llm: str,
    max_tool_calls: int,
    allow_live_llm: bool,
    out_dir: str | Path,
) -> dict:
    started = time.time()
    out_root = Path(out_dir)
    store = YieldMindStore()
    registry = registry_for_workspace(store=store)
    run_id = store.create_run(
        mode=LIVE_LLM_FUNCTION_CALLING if allow_live_llm else "skipped_live_llm_not_enabled",
        source="yieldmind_function_calling_smoke",
        status="running",
        metadata={"prompt": prompt, "llm": llm, "allow_live_llm": allow_live_llm},
    )
    request = ToolPlanRequest(prompt=prompt, llm=llm, max_tool_calls=max_tool_calls, allow_live_llm=allow_live_llm)

    if llm not in AVAILABLE_LLMs:
        payload = {
            "status": "failed",
            "reason": f"Unknown llm config: {llm}",
            "mode": "not_started",
            "run_id": run_id,
            "real_llm_calls": 0,
            "simulated_model_calls": 0,
            "llm_config": {"llm": llm},
            "duration_seconds": round(time.time() - started, 4),
        }
        report_path = _write_report(out_root, payload)
        payload["report_path"] = str(report_path)
        store.update_run(run_id, status="failed", result=payload, completed=True)
        return payload

    if not allow_live_llm:
        preview = local_rule_plan(ToolPlanRequest(prompt=prompt, max_tool_calls=max_tool_calls))
        payload = {
            "status": "skipped",
            "reason": "Live LLM function calling requires --allow-live-llm; no model API call was made.",
            "mode": "skipped_live_llm_not_enabled",
            "run_id": run_id,
            "prompt": prompt,
            "llm_config": _safe_llm_config(llm),
            "offline_preview_plan": preview.model_dump(),
            "real_llm_calls": 0,
            "simulated_model_calls": 0,
            "duration_seconds": round(time.time() - started, 4),
        }
        report_path = _write_report(out_root, payload)
        payload["report_path"] = str(report_path)
        store.add_event(run_id, stage="function_calling_smoke", message="Live smoke skipped by default.", payload=payload)
        store.update_run(run_id, status="skipped", result=payload, completed=True)
        return payload

    llm_config = _safe_llm_config(llm)
    if not llm_config["has_api_key"]:
        payload = {
            "status": "blocked",
            "reason": f"LLM {llm!r} has no configured API key; no model API call was made.",
            "mode": "blocked_missing_api_key",
            "run_id": run_id,
            "prompt": prompt,
            "llm_config": llm_config,
            "real_llm_calls": 0,
            "simulated_model_calls": 0,
            "duration_seconds": round(time.time() - started, 4),
        }
        report_path = _write_report(out_root, payload)
        payload["report_path"] = str(report_path)
        store.add_event(run_id, stage="function_calling_smoke", level="warning", message=payload["reason"], payload=payload)
        store.update_run(run_id, status="blocked", result=payload, completed=True)
        return payload

    try:
        plan = plan_tools(request, registry=registry)
        tool_names = [call.tool_name for call in plan.tool_calls]
        payload = {
            "status": "passed" if plan.mode == LIVE_LLM_FUNCTION_CALLING and plan.llm_calls == 1 else "failed",
            "mode": plan.mode,
            "run_id": run_id,
            "prompt": prompt,
            "llm_config": llm_config,
            "plan": plan.model_dump(),
            "selected_tool_names": tool_names,
            "real_llm_calls": plan.llm_calls,
            "simulated_model_calls": plan.simulated_model_calls,
            "duration_seconds": round(time.time() - started, 4),
        }
        report_path = _write_report(out_root, payload)
        payload["report_path"] = str(report_path)
        store.add_event(
            run_id,
            stage="function_calling_smoke",
            level="info" if payload["status"] == "passed" else "error",
            message=f"Function-calling smoke {payload['status']}.",
            payload=payload,
        )
        store.update_run(run_id, status=payload["status"], result=payload, completed=True)
        return payload
    except Exception as exc:
        payload = {
            "status": "failed",
            "reason": f"{type(exc).__name__}: {exc}",
            "mode": LIVE_LLM_FUNCTION_CALLING,
            "run_id": run_id,
            "prompt": prompt,
            "llm_config": llm_config,
            "real_llm_calls": 0,
            "simulated_model_calls": 0,
            "duration_seconds": round(time.time() - started, 4),
        }
        report_path = _write_report(out_root, payload)
        payload["report_path"] = str(report_path)
        store.add_event(run_id, stage="function_calling_smoke", level="error", message=payload["reason"], payload=payload)
        store.update_run(run_id, status="failed", result=payload, completed=True)
        return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Run YieldMind function-calling smoke check.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--llm", default="ark")
    parser.add_argument("--max-tool-calls", type=int, default=6)
    parser.add_argument("--allow-live-llm", action="store_true")
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "agent_workspace" / "yieldmind" / "function_calling"))
    args = parser.parse_args()
    report = run_smoke(
        prompt=args.prompt,
        llm=args.llm,
        max_tool_calls=args.max_tool_calls,
        allow_live_llm=args.allow_live_llm,
        out_dir=args.out_dir,
    )
    print(json_dumps(report))
    if report["status"] in {"passed", "skipped"}:
        return 0
    if report["status"] == "blocked":
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
