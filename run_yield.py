#!/usr/bin/env python3
"""Entry point for LLM-generated yield-stress AutoML.

YieldAgentManager keeps model code generation flexible while preserving a
manager loop. It supplies data/schema context, optional external-search
snippets, candidate/model plans, and a strict verification contract; the
OperationAgent writes the final training script inside each managed attempt.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import queue
import signal
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from configs import AVAILABLE_LLMs, DEFAULT_LLM, LOW_TOKEN_MODE, LOW_TOKEN_N_REVISE
from knowledge.yield_domain import (
    DEFAULT_YIELD_SEARCH_QUERIES,
    REFERENCE_MECHANISM_NOTES,
    YIELD_DOMAIN_CONTEXT,
)
from knowledge.yield_candidate_benchmark import (
    apply_benchmark_selection,
    execute_plugin_candidate_artifacts,
    execute_selected_benchmark_proxy_artifacts,
    run_candidate_benchmark,
)
from knowledge.yield_fusion_specs import (
    fusion_spec_readiness,
    merge_with_default_fusion_specs,
    normalize_fusion_specs,
)
from knowledge.yield_executor_specs import (
    attach_executor_specs_to_fusion_specs,
    executor_spec_readiness,
    normalize_executor_specs,
)
from knowledge.yield_schema import (
    DEFAULT_LIAN_DATA_PATH,
    DEFAULT_LIAN_TEST_PATH,
    load_yield_dataframe,
)
from knowledge.yield_synthetic_data import ensure_default_synthetic_yield_data
from knowledge.yield_retriever import build_yield_query_plan, retrieve_yield_sources
from operation_agent import OperationAgent
from operation_agent.yield_guardrails import verify_yield_plugin_harness_run
from utils import _emit_event, get_client


def _now_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _format_counts(counts: dict[str, Any]) -> str:
    if not isinstance(counts, dict) or not counts:
        return "none"
    return ", ".join(f"{key}={value}" for key, value in counts.items())


def _trim_line(text: Any, limit: int = 240) -> str:
    line = " ".join(str(text or "").split())
    return line if len(line) <= limit else line[:limit] + "..."


class _AgentLLMTimeout(TimeoutError):
    pass


def _agent_llm_timeout_seconds() -> float:
    raw = os.environ.get("YIELD_AGENT_LLM_TIMEOUT") or os.environ.get("LLM_REQUEST_TIMEOUT") or "240"
    try:
        return max(1.0, float(raw))
    except Exception:
        return 240.0


def _operation_timeout_seconds() -> float:
    raw = os.environ.get("YIELD_OPERATION_TIMEOUT") or "300"
    try:
        return max(1.0, float(raw))
    except Exception:
        return 300.0


def _call_with_agent_timeout(label: str, func):
    timeout_seconds = _agent_llm_timeout_seconds()
    if not hasattr(signal, "SIGALRM"):
        return func()
    def _raise_timeout(signum, frame):
        raise _AgentLLMTimeout(f"{label} exceeded {timeout_seconds:.0f}s")

    try:
        previous_handler = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, _raise_timeout)
        previous_timer = signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
    except Exception:
        return func()
    try:
        return func()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        if previous_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, previous_timer[0], previous_timer[1])
        signal.signal(signal.SIGALRM, previous_handler)


def _llm_chat_completion_worker(
    result_queue,
    llm: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
) -> None:
    try:
        response = get_client(llm).chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )
        result_queue.put({"ok": True, "content": response.choices[0].message.content or ""})
    except Exception as exc:
        result_queue.put({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


def _chat_completion_content_with_process_timeout(
    label: str,
    llm: str,
    messages: list[dict[str, str]],
    temperature: float,
) -> str:
    timeout_seconds = _agent_llm_timeout_seconds()
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    proc = ctx.Process(
        target=_llm_chat_completion_worker,
        args=(result_queue, llm, AVAILABLE_LLMs[llm]["model"], messages, temperature),
    )
    proc.daemon = True
    proc.start()
    proc.join(timeout_seconds)
    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        if proc.is_alive() and hasattr(proc, "kill"):
            proc.kill()
            proc.join(2)
        raise _AgentLLMTimeout(f"{label} exceeded {timeout_seconds:.0f}s")
    try:
        result = result_queue.get_nowait()
    except queue.Empty:
        raise RuntimeError(f"{label} exited without returning a completion")
    if not isinstance(result, dict) or not result.get("ok"):
        raise RuntimeError(result.get("error", "unknown LLM completion failure") if isinstance(result, dict) else result)
    return str(result.get("content") or "").strip()


def _operation_agent_worker(
    result_queue,
    user_requirements: dict[str, Any],
    llm: str,
    code_path: str,
    task: str,
    instructions: str,
    n_attempts: int,
) -> None:
    try:
        op = OperationAgent(
            user_requirements=user_requirements,
            llm=llm,
            code_path=code_path,
            task=task,
        )
        result = op.implement_solution(
            instructions,
            full_pipeline=False,
            n_attempts=max(1, int(n_attempts)),
        )
        result_queue.put({"ok": True, "result": result})
    except Exception as exc:
        result_queue.put(
            {
                "ok": False,
                "result": {
                    "rcode": 1,
                    "stage": "operation",
                    "action_result": f"OperationAgent failed before producing runnable code: {type(exc).__name__}: {exc}",
                    "code": "",
                    "error_logs": [f"OperationAgent exception: {type(exc).__name__}: {exc}"],
                },
            }
        )


def _run_operation_agent_with_process_timeout(
    user_requirements: dict[str, Any],
    llm: str,
    code_path: str,
    task: str,
    instructions: str,
    n_attempts: int,
) -> dict[str, Any]:
    timeout_seconds = _operation_timeout_seconds()
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    proc = ctx.Process(
        target=_operation_agent_worker,
        args=(result_queue, user_requirements, llm, code_path, task, instructions, n_attempts),
    )
    proc.daemon = True
    proc.start()
    proc.join(timeout_seconds)
    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        if proc.is_alive() and hasattr(proc, "kill"):
            proc.kill()
            proc.join(2)
        return {
            "rcode": 1,
            "stage": "operation",
            "action_result": f"OperationAgent timed out after {timeout_seconds:.0f}s before completing generated-code execution.",
            "code": "",
            "error_logs": [f"OperationAgent timeout after {timeout_seconds:.0f}s."],
        }
    try:
        result = result_queue.get_nowait()
    except queue.Empty:
        return {
            "rcode": 1,
            "stage": "operation",
            "action_result": "OperationAgent process exited without returning a result.",
            "code": "",
            "error_logs": ["OperationAgent process exited without result."],
        }
    if isinstance(result, dict) and isinstance(result.get("result"), dict):
        return result["result"]
    return {
        "rcode": 1,
        "stage": "operation",
        "action_result": "OperationAgent returned an invalid result payload.",
        "code": "",
        "error_logs": ["OperationAgent returned invalid result payload."],
    }


def _compact_search_report_for_prompt(search_report: dict[str, Any], max_snippets: int = 18) -> dict[str, Any]:
    snippets = search_report.get("snippets", []) if isinstance(search_report, dict) else []
    compact_snippets = []
    for item in snippets[:max_snippets]:
        if not isinstance(item, dict):
            continue
        compact_snippets.append(
            {
                "source_id": item.get("source_id") or item.get("id"),
                "provider": item.get("provider") or item.get("source"),
                "title": _trim_line(item.get("title"), 180),
                "link": item.get("link") or item.get("url") or item.get("doi"),
                "snippet": _trim_line(item.get("snippet") or item.get("summary"), 420),
                "relevance_label": item.get("relevance_label"),
                "relevance_score": item.get("relevance_score"),
                "query": _trim_line(item.get("query"), 140),
            }
        )
    return {
        "stage": search_report.get("stage"),
        "enabled": search_report.get("enabled"),
        "queries": search_report.get("queries", [])[:8],
        "provider_summary": search_report.get("provider_summary", {}),
        "source_type_summary": search_report.get("source_type_summary", {}),
        "source_quality": search_report.get("source_quality", {}),
        "snippets": compact_snippets,
        "snippet_count": len(snippets),
        "note": search_report.get("note"),
    }


def _read_json_artifact(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _numeric_metric(payload: Any, names: tuple[str, ...]) -> float | None:
    if not isinstance(payload, dict):
        return None
    for name in names:
        value = payload.get(name)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
    return None


def _nested_dict(payload: dict[str, Any] | None, *names: str) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    for name in names:
        value = payload.get(name)
        if isinstance(value, dict):
            return value
    return None


def _emit_grouped_search_payload(payload: dict[str, Any]) -> None:
    sources = payload.get("sources", []) if isinstance(payload, dict) else []
    queries = payload.get("queries", []) if isinstance(payload, dict) else []
    quality = payload.get("source_quality", {}) if isinstance(payload, dict) else {}
    relevance = quality.get("relevance_summary", {}) if isinstance(quality, dict) else {}
    _emit_event(
        "search",
        "SearchAgent:",
        (
            "YIELD_SEARCH_SUMMARY\n"
            f"TotalSources: {len(sources)}\n"
            f"SourceTypes: {_format_counts(payload.get('source_type_summary', {}))}\n"
            f"Providers: {_format_counts(payload.get('provider_summary', {}))}\n"
            f"Relevance: {_format_counts(relevance)}\n"
            f"QualityRule: {quality.get('quality_rule', '') if isinstance(quality, dict) else ''}"
        ),
        mirror=False,
    )

    query_by_group: dict[str, list[str]] = {}
    for item in queries:
        if not isinstance(item, dict):
            continue
        provider = item.get("provider") or item.get("source") or "Search"
        group = str(item.get("source") or provider)
        query_by_group.setdefault(group, [])
        query = str(item.get("query") or "").strip()
        if query and query not in query_by_group[group]:
            query_by_group[group].append(query)

    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in sources:
        if not isinstance(item, dict):
            continue
        key = f"{item.get('source_type') or 'source'} / {item.get('provider') or item.get('source') or 'Search'}"
        grouped.setdefault(key, []).append(item)

    if not grouped:
        _emit_event("search", "SearchAgent:", "YIELD_SOURCE_GROUP none count=0\nNo auditable external sources returned.", mirror=False)
        return

    for group, items in grouped.items():
        group_queries = []
        for source_name, qs in query_by_group.items():
            if source_name.lower() in group.lower() or any(source_name.lower() in str(x.get("provider", "")).lower() for x in items):
                group_queries.extend(qs)
        if not group_queries:
            group_queries = list(dict.fromkeys(str(x.get("query") or "") for x in items if x.get("query")))
        lines = [f"YIELD_SOURCE_GROUP {group} count={len(items)}"]
        if group_queries:
            lines.append("Queries:")
            lines.extend(f"- {q}" for q in group_queries[:5])
        lines.append("TopSources:")
        for item in sorted(items, key=lambda x: float(x.get("relevance_score", 0) or 0), reverse=True)[:8]:
            link = item.get("link") or item.get("url") or ""
            lines.append(f"[{item.get('source_id')}] {item.get('title', '(untitled)')}")
            lines.append(
                "  "
                f"Relevance: {item.get('relevance_label', 'N/A')} "
                f"({item.get('relevance_score', 'N/A')}/5); "
                f"Role: {item.get('evidence_role', 'unknown')}; "
                f"Category: {item.get('category', '')}"
            )
            if link:
                lines.append(f"  Link: {link}")
            snippet = _trim_line(item.get("snippet"), 260)
            if snippet:
                lines.append(f"  Snippet: {snippet}")
        _emit_event("search", "SearchAgent:", "\n".join(lines), mirror=False)


def _emit_query_plan(user_prompt: str, queries: list[str]) -> None:
    plan = build_yield_query_plan(user_prompt, queries)
    lines = ["YIELD_QUERY_GROUPS"]
    for group in plan:
        source = group.get("source", "Search")
        provider = group.get("provider") or source
        group_queries = group.get("queries", [])
        lines.append(f"{source} ({provider}) count={len(group_queries)}")
        for query in group_queries:
            lines.append(f"- {query}")
    _emit_event("search", "SearchAgent:", "\n".join(lines), mirror=False)


def _try_external_search(enabled: bool, user_prompt: str, queries: list[str], top_k: int = 3) -> dict[str, Any]:
    if not enabled:
        _emit_event("search", "SearchAgent:", "External search disabled by run configuration.", mirror=False)
        return {"source_count": 0, "provider_summary": {}, "queries": [], "sources": []}

    def emit(sender: str, message: str) -> None:
        if str(message or "").startswith("Querying "):
            return
        _emit_event(sender, "SearchAgent:" if sender == "search" else "SYSTEM", message, mirror=False)

    _emit_event(
        "search",
        "SearchAgent:",
        (
            "Retrieving external literature for yield-stress AutoML "
            "(Web Search/SearchAPI or configured provider + arXiv + Semantic Scholar + OpenAlex)."
        ),
        mirror=False,
    )
    _emit_query_plan(user_prompt, queries)
    payload = retrieve_yield_sources(user_prompt=user_prompt, extra_queries=queries, top_k_per_query=top_k, emit=emit)
    _emit_grouped_search_payload(payload)
    _emit_event(
        "search",
        "SearchAgent:",
        (
            "Synthesizing external literature into model primitives and mechanism candidates. "
            f"provider_summary={payload.get('provider_summary', {})}"
        ),
        mirror=False,
    )
    return payload


def _core_yield_context() -> str:
    return """
# Yield-Stress AutoML Core Constraints

Task: predict static yield stress tau0 / yield_stress for high-solid-content
slurries from available formulation and process variables.

The final model is intentionally NOT fixed. The code-generation agent may use
data-driven models, physics-informed neural networks, symbolic/structured
models, multi-fidelity training, ensembles, or other approaches discovered from
external search, as long as the implementation is runnable and passes the
verifier.

Current temporary datasets:
- Lian 2025 layout: Phi, SP_percent -> Tau0_Pa.
- Zhou 1999 layout: phi, d_s_um, optional powder -> tau_Pa.
- Synthetic yield layout: phi, sp_percent, water/binder, fly-ash ratio,
  particle/process/test variables -> yield_stress, generated by fixed project
  code from Lian 2025 Table 6 anchor and documented in synthetic_data_report.
- Future industrial layout may include phi, d50, sigma_d, Emix, temperature,
  and yield_stress.

Core engineering constraints:
- Never use target or auxiliary true-physics columns as model inputs:
  Tau0_Pa, tau_Pa, yield_stress, phi_max, m1_true, m1_lf.
- Fit preprocessing only on train folds/splits.
- Train, validation, and anchor transforms must share one fitted feature schema.
  Do not independently one-hot encode or engineer train and anchor into
  different column spaces.
- Always use 5-fold OOF as the primary generalization estimate. Select exactly
  one secondary evaluation; when a real Table 6 anchor exists, use anchor
  validation as that secondary evaluation. Do not report training-set metrics as
  final performance.
- Save a mechanism report. If any mechanistic equation or physics constraint is
  used, list its paper/source, the exact formula/relationship used, the data
  columns it maps to, and why it is appropriate. If no mechanism is used, state
  that explicitly.
- Preserve physical validity: predicted yield stress must be finite and
  non-negative. If a hidden physical variable is predicted, report validity
  diagnostics such as phi_m > phi or bounded m1_eff.
"""


def _infer_data_capability_audit(profile: dict[str, Any]) -> dict[str, Any]:
    schema = profile.get("schema", {}) if isinstance(profile, dict) else {}
    features = set(str(x) for x in schema.get("feature_columns", []) or [])
    columns = set(str(x) for x in profile.get("columns", []) or [])
    names = {x.lower() for x in features | columns}

    def has_any(*items: str) -> bool:
        lowered = {x.lower() for x in items}
        return bool(names & lowered)

    has_phi = has_any("phi", "Phi".lower(), "volume_fraction", "solid_volume_fraction")
    has_sp = has_any("sp_percent", "SP_percent".lower(), "sp_kg_m3", "plasticizer", "dispersant")
    has_shear_rate = has_any("gamma_dot", "shear_rate", "shear_rate_s-1", "shear_rate_1_s", "dot_gamma")
    has_shear_curve = has_shear_rate and has_any("shear_stress", "tau", "stress_pa", "viscosity")
    has_particle = has_any("d_s_um", "d50", "d50_um", "psd_width", "sigma_d", "specific_surface_m2kg", "powder")
    has_composition = has_any(
        "cement_kg_m3",
        "fly_ash_kg_m3",
        "water_kg_m3",
        "sp_kg_m3",
        "binder_kg_m3",
        "sp_binder_ratio",
        "w_b",
        "fa_ratio",
        "Vp_L",
        "Vw_L",
    )
    has_phi_max_reference = has_any("phi_max", "phi_max_reference", "phi_max_eff_reference")
    has_process = has_any(
        "mixing_energy",
        "mixing_time_min",
        "mixing_speed_rpm",
        "temperature_c",
        "temperature",
        "rest_time_min",
        "curing_agent_ratio",
    )
    has_surface_chemistry = has_any("ph", "zeta", "zeta_potential", "ionic_strength", "salt_concentration")

    support: dict[str, dict[str, Any]] = {}
    if has_shear_curve:
        support["herschel_bulkley_bingham"] = {
            "status": "supported",
            "score_cap": 5,
            "reason": "Shear-rate/flow-curve columns exist, so constitutive residuals can be evaluated directly.",
        }
    else:
        support["herschel_bulkley_bingham"] = {
            "status": "unsupported",
            "score_cap": 1,
            "reason": "No shear-rate/flow-curve columns were found; HB/Bingham can be cited as background but not used as a strict residual loss.",
        }

    if has_phi and (has_particle or has_composition or has_phi_max_reference):
        support["yodel_packing_fmax"] = {
            "status": "supported",
            "score_cap": 5,
            "reason": "Solid fraction exists with particle, formulation, or packing-reference columns, supporting packing/fmax/YODEL-style candidates.",
        }
    elif has_phi:
        support["yodel_packing_fmax"] = {
            "status": "partial",
            "score_cap": 3,
            "reason": "Solid fraction exists, but particle/formulation/packing descriptors are limited; only reduced packing features or baselines are supported.",
        }
    else:
        support["yodel_packing_fmax"] = {
            "status": "unsupported",
            "score_cap": 1,
            "reason": "No solid fraction column was found, so packing/fmax/YODEL-style relations are not supported.",
        }

    # Lian-style formulation-to-packing needs actual mix-proportion columns
    # (cement/fly-ash/water/w_b/fa_ratio). Superplasticizer dosage alone is a
    # dispersant signal, not full formulation, so phi+sp is only partial support.
    if has_phi and has_composition:
        lian_status, lian_cap, lian_reason = (
            "supported",
            5,
            "Phi plus mix-proportion columns (cement/fly-ash/water/w_b/fa_ratio) exist, "
            "so Lian-style formulation-to-packing candidates are fully supported.",
        )
    elif has_phi and has_sp:
        lian_status, lian_cap, lian_reason = (
            "partial",
            3,
            "Phi and superplasticizer dosage exist but full mix-proportion columns are missing; "
            "only reduced/composition-lite Lian-style features are supported, not the full formulation model.",
        )
    elif has_phi:
        lian_status, lian_cap, lian_reason = (
            "partial",
            2,
            "Only phi exists; Lian-style formulation-packing needs superplasticizer or composition columns, "
            "so support is weak.",
        )
    else:
        lian_status, lian_cap, lian_reason = (
            "unsupported",
            1,
            "No phi column was found, so Lian-style formulation-packing is not supported.",
        )
    support["lian_formulation_packing"] = {
        "status": lian_status,
        "score_cap": lian_cap,
        "reason": lian_reason,
    }
    support["thixotropy_process_structure"] = {
        "status": "supported" if has_process else "unsupported",
        "score_cap": 4 if has_process else 1,
        "reason": "Process/rest/temperature variables exist." if has_process else "No process/rest/temperature variables were found.",
    }
    support["dlvo_surface_charge"] = {
        "status": "supported" if has_surface_chemistry else "unsupported",
        "score_cap": 4 if has_surface_chemistry else 1,
        "reason": "Surface chemistry variables exist." if has_surface_chemistry else "No pH/zeta/ionic-strength variables were found.",
    }

    return {
        "available_feature_columns": sorted(features),
        "available_columns": sorted(columns),
        "flags": {
            "has_phi": has_phi,
            "has_additive_or_sp": has_sp,
            "has_shear_rate_or_flow_curve": has_shear_curve,
            "has_particle_or_psd": has_particle,
            "has_formulation_composition": has_composition,
            "has_phi_max_reference": has_phi_max_reference,
            "has_process_or_rest_variables": has_process,
            "has_surface_chemistry": has_surface_chemistry,
        },
        "mechanism_support": support,
        "selection_policy": [
            "Do not select HB/Bingham residual losses unless shear-rate or flow-curve columns exist.",
            "Force at least one YODEL/packing/fmax-style hybrid into the candidate set when phi plus formulation, particle, or packing-reference columns are available.",
            "Prefer supported packing/fmax/YODEL-style candidates over unsupported HB/Bingham/PINN constitutive residuals.",
            "Prefer mechanisms that can be mapped to current columns over mechanisms that only look complex.",
        ],
    }


def _mechanism_kind(text: str) -> str:
    lower = text.lower()
    if any(token in lower for token in ("herschel", "bingham", "casson", "yield surface", "constitutive")):
        return "herschel_bulkley_bingham"
    if any(
        token in lower
        for token in (
            "yodel",
            "flatt",
            "bowen",
            "packing",
            "fmax",
            "phi_m",
            "maximum packing",
            "jamming",
            "solid fraction",
            "volume fraction",
            "phi monotonic",
            "phi-dependent",
            "phi dependent",
        )
    ):
        return "yodel_packing_fmax"
    if any(token in lower for token in ("lian", "formulation", "superplasticizer", "dispersant", "pce", "sp_percent")):
        return "lian_formulation_packing"
    if any(token in lower for token in ("thix", "rest", "aging", "structur", "mixing", "temperature")):
        return "thixotropy_process_structure"
    padded = f" {lower} "
    if any(token in lower for token in ("dlvo", "zeta", "surface charge", "ionic strength", "ionic_strength", "salt concentration", "salt_concentration")) or " ph " in padded:
        return "dlvo_surface_charge"
    return "unknown"


def _mechanism_support_from_audit(text: str, audit: dict[str, Any]) -> tuple[str, str, int, str]:
    kind = _mechanism_kind(text)
    support = audit.get("mechanism_support", {}).get(kind)
    if not isinstance(support, dict):
        return "partial", "Mechanism kind is not recognized; require explicit column mapping before use.", 3, kind
    return (
        str(support.get("status", "partial")),
        str(support.get("reason", "")),
        int(support.get("score_cap", 3)),
        kind,
    )


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _candidate_text(item: dict[str, Any]) -> str:
    if not isinstance(item, dict):
        return ""
    parts = []
    for value in item.values():
        if isinstance(value, (str, int, float, bool)):
            parts.append(str(value))
        elif isinstance(value, (list, tuple)):
            parts.extend(str(x) for x in value)
        elif isinstance(value, dict):
            parts.extend(str(x) for x in value.values())
    return " ".join(parts)


def _slug_id(value: Any, default: str = "item") -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    text = "".join(ch for ch in text if ch.isalnum() or ch == "_").strip("_")
    return text or default


def _required_columns_from_spec(spec: dict[str, Any]) -> list[str]:
    cols = spec.get("required_columns") if isinstance(spec, dict) else []
    if isinstance(cols, str):
        cols = [cols]
    # LLM candidate specs often use intent markers rather than literal schema
    # names. These mean "use the current feature schema after leakage removal",
    # so they must not be treated as missing physical columns.
    all_feature_markers = {
        "all_available_non_leakage",
        "all_available_features",
        "all_non_leakage_features",
        "all_numeric_features",
        "all_features",
        "available_features",
        "all_raw_features",
        "raw_features",
        "all_phys_features",
        "all_physics_features",
        "all_physical_features",
        "phys_features",
        "physics_features",
        "physical_features",
        "all_engineered_features",
        "engineered_features",
        "formulation_features",
        "composition_features",
        "process_features",
        "mixing_features",
        "temperature_features",
        "thermal_features",
        "particle_features",
        "packing_features",
    }

    def is_all_feature_marker(value: str) -> bool:
        lowered = value.strip().lower()
        compact = "".join(ch if ch.isalnum() or ch == "_" else " " for ch in lowered)
        compact = "_".join(compact.split())
        if compact in all_feature_markers:
            return True
        chinese_markers = (
            "所有可用数值特征",
            "所有可用特征",
            "全部可用数值特征",
            "全部安全特征",
            "全部非泄漏特征",
        )
        if any(marker in value for marker in chinese_markers):
            return True
        tokens = set(compact.split("_"))
        has_all_scope = bool(tokens & {"all", "available", "current", "schema", "safe"})
        has_feature_word = bool(tokens & {"feature", "features", "column", "columns", "numeric", "numerical"})
        has_safety_or_scope = bool(tokens & {
            "available", "current", "schema", "safe", "non", "leakage",
            "numeric", "numerical", "raw", "phys", "physics", "physical",
            "engineered", "formulation", "composition", "process", "mixing",
            "temperature", "thermal", "particle", "packing",
        })
        return has_all_scope and has_feature_word and has_safety_or_scope

    out: list[str] = []
    for col in cols or []:
        text = str(col).strip()
        if not text:
            continue
        if is_all_feature_marker(text):
            continue
        out.append(text)
    return out


def _spec_score(spec: dict[str, Any]) -> float:
    base: float | None = None
    for key in ("feasibility_score_1_to_5", "total_score_1_to_10", "score"):
        try:
            value = spec.get(key)
            if value is not None:
                base = float(value)
                break
        except Exception:
            continue
    if base is None:
        base = 0.0
    try:
        # Data-fit is a soft tie-breaker, not a hard filter. It nudges LLM
        # model codegen toward structures that match the current schema while
        # leaving final selection to CV/guardrails.
        fit = float(spec.get("data_fit_score_1_to_5"))
        base += (fit - 3.0) * 0.25
    except Exception:
        pass
    return float(base)


def _model_design_guidance_from_profile(data_profile: dict[str, Any]) -> dict[str, Any]:
    schema = data_profile.get("schema", {}) if isinstance(data_profile, dict) else {}
    features = [str(x) for x in schema.get("feature_columns", []) or []]
    numeric = data_profile.get("numeric_summary", {}) if isinstance(data_profile, dict) else {}
    n_rows = int(data_profile.get("n_rows") or 0) if isinstance(data_profile, dict) else 0
    n_features = len(features)
    phi = numeric.get("phi") if isinstance(numeric, dict) else None
    phi_range = None
    if isinstance(phi, dict):
        try:
            phi_range = float(phi.get("max", 0.0)) - float(phi.get("min", 0.0))
        except Exception:
            phi_range = None

    def count_if(*tokens: str) -> int:
        lowered = [c.lower() for c in features]
        return sum(any(token in col for token in tokens) for col in lowered)

    groups = {
        "physics_or_engineered": count_if("phys_"),
        "thermal": count_if("temp", "thermal", "jacket", "water_tank"),
        "pressure": count_if("pressure", "kpa"),
        "mixing_or_process": count_if("mix", "rpm", "time", "shear", "dose"),
        "composition": count_if("mass", "fraction", "ratio", "ap_", "rdx", "solid"),
    }
    rules: list[str] = []
    cautions: list[str] = []
    if n_rows and n_features and n_rows < max(300, 5 * n_features):
        rules.append(
            "Small-sample/high-dimensional setting: prefer regularized, low-capacity structures "
            "over deep MLPs or unconstrained kernel models."
        )
        rules.append(
            "Use low-order interactions, sparse/regularized regression, or small two-branch voting/stacking "
            "with clear roles instead of a generic estimator wrapper."
        )
        cautions.append("Downweight deep neural networks, Gaussian processes, and wide RBF kernels unless strongly justified.")
    if groups["physics_or_engineered"]:
        rules.append(
            "The schema contains engineered phys_* features; the model idea should explicitly use or preserve "
            "these physics/process features rather than treating the task as generic tabular regression."
        )
    if groups["thermal"] or groups["pressure"] or groups["mixing_or_process"] or groups["composition"]:
        rules.append(
            "Different feature groups exist (composition, thermal/pressure, mixing/process); prefer structures "
            "that handle grouped signals or interactions between these groups."
        )
    if phi_range is not None and abs(phi_range) < 1e-9:
        cautions.append(
            "phi is constant in the current data; pure packing models driven mainly by phi cannot explain variation alone."
        )
    return {
        "n_rows": n_rows,
        "n_features": n_features,
        "feature_group_counts": groups,
        "phi_range": phi_range,
        "model_generation_rules": rules,
        "deprioritize_or_require_extra_justification": cautions,
    }


def _model_design_guidance_from_sample(sample_df: pd.DataFrame | None) -> dict[str, Any]:
    if sample_df is None or not hasattr(sample_df, "columns"):
        return {}
    df = sample_df.copy()
    numeric = df.select_dtypes(include="number")
    summary = {}
    for col in numeric.columns:
        series = pd.to_numeric(numeric[col], errors="coerce")
        summary[col] = {
            "min": float(series.min()) if series.notna().any() else 0.0,
            "max": float(series.max()) if series.notna().any() else 0.0,
        }
    return _model_design_guidance_from_profile({
        "n_rows": int(len(df)),
        "schema": {"feature_columns": list(df.columns)},
        "numeric_summary": summary,
    })


def _score_model_spec_data_fit(spec: dict[str, Any], data_profile: dict[str, Any]) -> tuple[int, list[str]]:
    guidance = _model_design_guidance_from_profile(data_profile)
    text = _candidate_text(spec).lower()
    n_rows = int(guidance.get("n_rows") or 0)
    n_features = int(guidance.get("n_features") or 0)
    phi_range = guidance.get("phi_range")
    score = 3
    reasons: list[str] = []
    small_high_dim = bool(n_rows and n_features and n_rows < max(300, 5 * n_features))

    if small_high_dim:
        if any(token in text for token in ("elastic", "lasso", "ridge", "sparse", "regularized", "bayesian")):
            score += 1
            reasons.append("small/high-dimensional data favors regularized or sparse models")
        if any(token in text for token in ("deep", "wide", "mlp", "neural", "gaussian process", "gpr", "rbf")):
            score -= 1
            reasons.append("small/high-dimensional data makes deep or unconstrained kernel models less stable")
    if any(token in text for token in ("group", "branch", "composition", "thermal", "pressure", "mixing", "process", "phys_")):
        score += 1
        reasons.append("model idea references current feature groups or engineered physics features")
    if phi_range is not None and abs(float(phi_range)) < 1e-9 and any(token in text for token in ("packing", "yodel", "phi")):
        score -= 1
        reasons.append("phi is constant in current data, so phi-only packing variation is limited")
    if not reasons:
        reasons.append("no strong data-fit signal; keep as neutral and let CV decide")
    return max(1, min(5, int(score))), reasons


def _model_spec_category(spec: dict[str, Any]) -> str:
    text = " ".join(
        str(spec.get(k, ""))
        for k in ("id", "name", "family", "modeling_idea")
    ).lower().replace("non-linear", "nonlinear").replace("non linear", "nonlinear")
    if any(token in text for token in ("pinn", "physics-informed", "physics informed", "yodel", "residual", "m1_eff", "latent")):
        return "physics_hybrid"
    if any(token in text for token in ("forest", "randomforest", "random forest", "tree", "boost", "histgbm", "hgb", "xgb", "lightgbm", "catboost")):
        return "tree_ensemble"
    if any(token in text for token in ("kernel", "gaussian process", "gpr", "svr", "svm", "rbf")):
        return "kernel_or_svm"
    if any(token in text for token in ("mlp", "neural", "network", "branch", "branched", "deep")):
        return "neural"
    if any(token in text for token in ("ridge", "linear", "lasso", "elastic", "polynomial", "kernel ridge")):
        return "linear_regularized"
    return "other"


def _model_spec_duplicate_family(spec: dict[str, Any]) -> str:
    text = " ".join(
        str(spec.get(k, ""))
        for k in ("id", "name", "family", "modeling_idea")
    ).lower()
    if any(token in text for token in ("kernel ridge", "kernelridge", "polynomial", "gaussian process", "gpr", "svr", "svm", "extra trees", "extratrees")):
        return ""
    if any(token in text for token in ("random forest", "randomforest", "forest", " rf")):
        return "rf"
    if any(token in text for token in ("histgradient", "histgbm", "hgb", "gradient boost", "boosting", "xgb", "lightgbm", "catboost")):
        return "hgb"
    if any(token in text for token in ("ridge regression", "ridge", "linear model")):
        return "ridge"
    if any(token in text for token in ("mlp", "neural network", "neural")):
        return "mlp"
    return ""


def _model_spec_should_codegen(spec: dict[str, Any]) -> bool:
    if not isinstance(spec, dict):
        return False
    if bool(spec.get("force_llm_codegen")):
        return True
    intent = str(spec.get("codegen_intent") or spec.get("implementation_authority") or "").strip().lower()
    if intent in {"llm_model", "generated_model", "custom_model", "llm_codegen"}:
        return True
    return not bool(_model_spec_duplicate_family(spec))


def _is_complex_model_spec(spec: dict[str, Any]) -> bool:
    if not isinstance(spec, dict):
        return False
    kind = str(
        spec.get("model_kind")
        or spec.get("contract")
        or spec.get("implementation_kind")
        or ""
    ).strip().lower()
    if kind in {
        "latent_physics_architecture",
        "complex_latent_physics",
        "hidden_parameter_physics",
        "latent_m1_eff",
    }:
        return True
    text = _candidate_text(spec).lower()
    return (
        ("latent" in text or "m1_eff" in text or "hidden parameter" in text)
        and ("physics" in text or "physical" in text or "mechanism" in text)
    )


def _supplemental_llm_model_specs(data_profile: dict[str, Any]) -> list[dict[str, Any]]:
    schema = data_profile.get("schema", {}) if isinstance(data_profile, dict) else {}
    target = str(schema.get("target_column") or "yield_stress")
    common = {
        "required_columns": "all_available_numeric_features",
        "expected_artifacts": ["logs/models/<model_id>.py", "metrics/free_search_report.json"],
        "source_ids_or_links": ["MANAGER_MODEL_QUALITY_REPAIR"],
        "source_evidence_map": {
            "manager_policy": (
                "Added because CandidateAgent supplied only fixed-family baselines; "
                "OperationAgent must still test LLM-generated model factories."
            )
        },
        "novelty_filter": "llm_codegen_candidate",
        "force_llm_codegen": True,
        "codegen_intent": "llm_model",
        "manager_quality_repair": True,
    }
    return [
        {
            **common,
            "id": "model_latent_m1_branched_small_data",
            "name": "Latent m1_eff grouped physics architecture",
            "family": "latent_physics_architecture",
            "model_kind": "latent_physics_architecture",
            "complex_model_contract": True,
            "modeling_idea": (
                "Generate a controlled latent-physics estimator using "
                "LatentM1PhysicsLayerRegressor. It should learn a positive "
                f"sample-wise latent m1_eff(X) from grouped composition, thermal/pressure, "
                f"and mixing/process features, then let the harness compute {target} through "
                "the selected fold-local mechanism physics layer. Prefer latent_backend='ridge' "
                "or 'kernel_ridge' for small high-dimensional datasets; include 'mlp' only as "
                "a bounded comparison. Prefer branched_hidden16 or mlp3_hidden16; do not write "
                "a custom training loop."
            ),
            "strengths": "Directly tests whether grouped feature encoders plus a physics layer help small-sample yield prediction.",
            "risks": "Can overfit if the mechanism shape is weak or the latent inversion target is noisy.",
            "feasibility_score_1_to_5": 4,
        },
        {
            **common,
            "id": "model_kernel_ridge_poly_small_data",
            "name": "Small-data polynomial KernelRidge pipeline",
            "family": "kernel_regularized_small_data",
            "modeling_idea": (
                "Generate a structured sklearn estimator that standardizes all numeric non-leakage inputs, "
                "expands low-order interaction features with PolynomialFeatures, and fits KernelRidge or "
                f"Ridge-style regularized regression to {target}. Keep degree/alpha/gamma grids small."
            ),
            "strengths": "Useful when engineered physical features make the response smooth and low-dimensional.",
            "risks": "Polynomial interactions can overfit; CV must decide whether it beats raw Ridge.",
            "feasibility_score_1_to_5": 4,
        },
        {
            **common,
            "id": "model_svr_rbf_small_data",
            "name": "Scaled RBF-SVR small-sample regressor",
            "family": "kernel_or_svm",
            "modeling_idea": (
                "Generate a compact structured estimator, not plain StandardScaler+SVR: combine an RBF-SVR "
                "with a regularized polynomial Ridge/KernelRidge branch using VotingRegressor or StackingRegressor. "
                "Keep C/gamma/epsilon/alpha grids small."
            ),
            "strengths": "Often competitive on small tabular datasets after feature scaling.",
            "risks": "May underperform on noisy high-dimensional features; needs strict hyperparameter caps.",
            "feasibility_score_1_to_5": 4,
        },
        {
            **common,
            "id": "model_gpr_matern_uncertainty",
            "name": "Matern Gaussian-process small-data regressor",
            "family": "kernel_uncertainty",
            "modeling_idea": (
                "Generate a compact structured uncertainty estimator, not plain StandardScaler+GPR: combine "
                "a Matern/RBF GaussianProcessRegressor with a regularized polynomial Ridge/KernelRidge branch "
                "using VotingRegressor or StackingRegressor. Keep kernels and alpha grids bounded."
            ),
            "strengths": "Good diagnostic for small smooth datasets and can expose whether nonlinear kernels help.",
            "risks": "Can be slower and unstable if feature scaling or kernel bounds are poor.",
            "feasibility_score_1_to_5": 3,
        },
    ]


def _ensure_llm_model_candidate_quality(
    payload: dict[str, Any],
    data_profile: dict[str, Any],
    min_codegen_specs: int = 2,
) -> dict[str, Any]:
    models = payload.setdefault("candidate_models", [])
    if not isinstance(models, list):
        models = []
        payload["candidate_models"] = models

    valid_models = [m for m in models if isinstance(m, dict)]
    duplicate_by_id = {
        str(m.get("id") or m.get("name") or ""): _model_spec_duplicate_family(m)
        for m in valid_models
        if _model_spec_duplicate_family(m)
    }
    codegen_ready = [m for m in valid_models if _model_spec_should_codegen(m)]
    has_complex_codegen = any(_is_complex_model_spec(m) and _model_spec_should_codegen(m) for m in valid_models)
    added: list[str] = []
    existing_ids = {str(m.get("id") or "") for m in valid_models}
    for spec in _supplemental_llm_model_specs(data_profile):
        if len(codegen_ready) >= min_codegen_specs and (has_complex_codegen or not _is_complex_model_spec(spec)):
            break
        sid = str(spec.get("id") or "")
        if sid in existing_ids:
            continue
        if _is_complex_model_spec(spec) and has_complex_codegen:
            continue
        models.append(spec)
        valid_models.append(spec)
        codegen_ready.append(spec)
        if _is_complex_model_spec(spec):
            has_complex_codegen = True
        existing_ids.add(sid)
        added.append(sid)

    guidance = _model_design_guidance_from_profile(data_profile)
    for spec in valid_models:
        if not isinstance(spec, dict):
            continue
        fit_score, fit_reasons = _score_model_spec_data_fit(spec, data_profile)
        spec["data_fit_score_1_to_5"] = fit_score
        spec["data_fit_reasons"] = fit_reasons
        spec["data_fit_guidance_summary"] = guidance

    status = "passed" if len(codegen_ready) >= min_codegen_specs and not added else "repaired"
    if len(codegen_ready) < min_codegen_specs:
        status = "insufficient_after_repair"
    audit = {
        "stage": "candidate_model_quality_audit",
        "status": status,
        "rule": (
            "Candidate models must include LLM-codegen-ready ideas beyond fixed RF/HGB/Ridge/MLP baselines; "
            "standard baselines remain fixed-family floor candidates."
        ),
        "min_codegen_specs": int(min_codegen_specs),
        "n_candidate_models_before_repair": len(valid_models) - len(added),
        "n_codegen_ready_after_repair": len([m for m in valid_models if _model_spec_should_codegen(m)]),
        "duplicate_fixed_family_ids": duplicate_by_id,
        "added_model_ids": added,
        "data_fit_guidance": guidance,
        "data_fit_scores": {
            str(m.get("id") or m.get("name") or ""): {
                "score": m.get("data_fit_score_1_to_5"),
                "reasons": m.get("data_fit_reasons"),
            }
            for m in valid_models
        },
        "codegen_ready_model_ids": [
            str(m.get("id") or m.get("name") or "")
            for m in valid_models
            if _model_spec_should_codegen(m)
        ],
        "complex_codegen_model_ids": [
            str(m.get("id") or m.get("name") or "")
            for m in valid_models
            if _is_complex_model_spec(m) and _model_spec_should_codegen(m)
        ],
    }
    payload["candidate_model_quality_audit"] = audit
    if added:
        payload["selection_audit_note"] = (
            str(payload.get("selection_audit_note") or "").strip()
            + (" | " if payload.get("selection_audit_note") else "")
            + "Manager repaired candidate_models with LLM-codegen-ready model briefs because the original list was only fixed baselines."
        )
    return audit


def _select_model_specs_for_codegen(specs: list[dict[str, Any]], cap: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select LLM model specs for codegen by coverage, not first-N order.

    The aim is to avoid spending all quick/normal LLM calls on repeated tree
    baselines when the searched space also contains small-sample linear,
    physics-hybrid, or neural ideas.
    """
    if cap <= 0 or not specs:
        return [], {
            "strategy": "coverage_by_model_category",
            "cap": max(0, int(cap)),
            "categories_considered": [],
            "selected_categories": [],
            "duplicate_fixed_family_ids": {},
        }

    categories = ["physics_hybrid", "linear_regularized", "tree_ensemble", "kernel_or_svm", "neural", "other"]
    buckets: dict[str, list[dict[str, Any]]] = {cat: [] for cat in categories}
    duplicate_by_id: dict[str, str] = {}
    category_by_id: dict[str, str] = {}
    for spec in specs:
        cat = _model_spec_category(spec)
        category_by_id[str(spec.get("id") or spec.get("name") or "")] = cat
        dup = _model_spec_duplicate_family(spec)
        if dup:
            duplicate_by_id[str(spec.get("id") or spec.get("name") or "")] = dup
        buckets.setdefault(cat, []).append(spec)

    def item_key(spec: dict[str, Any]) -> tuple[Any, ...]:
        dup = _model_spec_duplicate_family(spec)
        # Repeated fixed-family wrappers are still allowed, just deprioritized
        # relative to ideas with a different modeling structure.
        return (
            1 if dup in {"rf", "hgb", "ridge", "mlp"} else 0,
            -_spec_score(spec),
            str(spec.get("id") or spec.get("name") or ""),
        )

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    complex_specs = sorted([s for s in specs if _is_complex_model_spec(s)], key=item_key)
    if complex_specs:
        spec = complex_specs[0]
        sid = str(spec.get("id") or spec.get("name") or "")
        selected.append(spec)
        selected_ids.add(sid)
    for cat in categories:
        if len(selected) >= cap:
            break
        bucket = sorted(buckets.get(cat, []), key=item_key)
        if not bucket:
            continue
        spec = bucket[0]
        sid = str(spec.get("id") or spec.get("name") or "")
        if sid in selected_ids:
            continue
        selected.append(spec)
        selected_ids.add(sid)

    if len(selected) < cap:
        for spec in sorted(specs, key=item_key):
            sid = str(spec.get("id") or spec.get("name") or "")
            if sid in selected_ids:
                continue
            selected.append(spec)
            selected_ids.add(sid)
            if len(selected) >= cap:
                break

    selected_ids_list = [str(s.get("id") or s.get("name") or "") for s in selected]
    return selected, {
        "strategy": "coverage_by_model_category",
        "cap": int(cap),
        "category_order": categories,
        "categories_considered": category_by_id,
        "selected_categories": {
            str(s.get("id") or s.get("name") or ""): _model_spec_category(s)
            for s in selected
        },
        "duplicate_fixed_family_ids": duplicate_by_id,
        "selected_ids": selected_ids_list,
        "forced_complex_codegen_id": (
            str(complex_specs[0].get("id") or complex_specs[0].get("name") or "")
            if complex_specs else None
        ),
        "dropped_after_codegen_cap": [
            str(s.get("id") or s.get("name") or "")
            for s in specs
            if str(s.get("id") or s.get("name") or "") not in set(selected_ids_list)
        ],
    }


class _PlannedMechanismSpec:
    """Mechanism shell used only for pre-codegen planned screening."""

    def __init__(self, spec: dict[str, Any]):
        self.spec = dict(spec or {})
        self.id = _slug_id(self.spec.get("id") or self.spec.get("name"), "planned_mechanism")
        self.name = str(self.spec.get("name") or self.id)
        self.paper = str(self.spec.get("paper_or_source") or self.spec.get("paper") or "")
        self.required_columns = tuple(_required_columns_from_spec(self.spec) or ["phi"])
        text = _candidate_text(self.spec).lower()
        self.needs_shear_rate = bool(self.spec.get("needs_shear_rate")) or any(
            token in text for token in ("herschel", "bulkley", "bingham", "casson", "shear-rate", "shear rate")
        )
        self.needs_microstructure = bool(self.spec.get("needs_microstructure")) or any(
            token in text for token in ("psd", "ssa", "d50", "particle size", "specific surface")
        )

    def data_adequacy(self, columns) -> dict[str, Any]:
        from knowledge.yield_mechanisms import _screen_columns

        return _screen_columns(
            self.name,
            self.required_columns,
            self.needs_shear_rate,
            self.needs_microstructure,
            columns,
        )

    def features(self, *_args, **_kwargs):
        raise RuntimeError("planned-only mechanism shell cannot compute features")

    def base_predict(self, *_args, **_kwargs):
        raise RuntimeError("planned-only mechanism shell cannot predict")


def _ensure_yodel_packing_candidate(
    payload: dict[str, Any],
    audit: dict[str, Any],
    data_profile: dict[str, Any],
) -> bool:
    yodel_support = audit.get("mechanism_support", {}).get("yodel_packing_fmax", {})
    support_status = str(yodel_support.get("status", "unsupported"))
    if support_status == "unsupported":
        payload["yodel_packing_audit"] = {
            "candidate_forced": False,
            "support_status": support_status,
            "reason": yodel_support.get("reason", "YODEL/packing is unsupported by current columns."),
        }
        return False

    schema = data_profile.get("schema", {}) if isinstance(data_profile, dict) else {}
    features = [str(x) for x in schema.get("feature_columns", []) or []]
    feature_set = set(features)
    yodel_columns = [
        col
        for col in [
            "phi",
            "sp_percent",
            "w_b",
            "fa_ratio",
            "binder_kg_m3",
            "sp_binder_ratio",
            "d_s_um",
            "d50",
            "d50_um",
            "psd_width",
            "specific_surface_m2kg",
            "mixing_energy",
            "mixing_time_min",
            "mixing_speed_rpm",
            "temperature_c",
            "rest_time_min",
        ]
        if col in feature_set
    ]

    mechanisms = payload.setdefault("candidate_mechanisms", [])
    existing_yodel_mech = next(
        (
            item
            for item in mechanisms
            if isinstance(item, dict)
            and _mechanism_kind(_candidate_text(item)) == "yodel_packing_fmax"
        ),
        None,
    )
    if existing_yodel_mech is None:
        existing_ids = {str(item.get("id")) for item in mechanisms if isinstance(item, dict)}
        mech_id = "P_YODEL_PACKING_FMAX"
        suffix = 1
        while mech_id in existing_ids:
            suffix += 1
            mech_id = f"P_YODEL_PACKING_FMAX_{suffix}"
        existing_yodel_mech = {
            "id": mech_id,
            "name": "YODEL/packing/fmax yield-stress scaling",
            "paper_or_source": "Flatt and Bowen (2006) YODEL plus Lian et al. (2025) virtual maximum packing fraction Table 6 formulation context",
            "formula_or_relationship": (
                "Use a leakage-safe packing index such as max(phi-phi_c,0)^p / max(phi_m_eff-phi, eps), "
                "where phi_m_eff/fmax is a fold-local fitted or fixed global parameter derived only from allowed "
                "formulation/particle/process columns, never from phi_max_eff_reference or other *_reference columns."
            ),
            "required_columns": ["phi"] + [col for col in yodel_columns if col != "phi"],
            "role_options": ["feature", "baseline", "constraint"],
            "applicability_to_current_data": yodel_support.get("reason"),
            "source_ids_or_links": ["LOCAL_YODEL_FLATT_BOWEN_2006", "LOCAL_TABLE6_MATERIALS_18_02983"],
            "source_evidence_map": [
                {
                    "source_id": "LOCAL_YODEL_FLATT_BOWEN_2006",
                    "used_claim": "Yield stress in concentrated suspensions can be related to packing/jamming distance.",
                },
                {
                    "source_id": "LOCAL_TABLE6_MATERIALS_18_02983",
                    "used_claim": "Table 6 formulation variables support virtual maximum packing/fmax style features.",
                },
            ],
            "feasibility_score_1_to_5": 5 if support_status == "supported" else 3,
            "forced_by_manager_audit": True,
        }
        mechanisms.append(existing_yodel_mech)
    mech_id = str(existing_yodel_mech.get("id") or "P_YODEL_PACKING_FMAX")

    models = payload.setdefault("candidate_models", [])
    existing_yodel_model = next(
        (
            item
            for item in models
            if isinstance(item, dict)
            and any(token in _candidate_text(item).lower() for token in ("yodel", "packing", "fmax", "phi_m", "maximum packing"))
        ),
        None,
    )
    if existing_yodel_model is None:
        existing_ids = {str(item.get("id")) for item in models if isinstance(item, dict)}
        model_id = "M_YODEL_RESIDUAL"
        suffix = 1
        while model_id in existing_ids:
            suffix += 1
            model_id = f"M_YODEL_RESIDUAL_{suffix}"
        existing_yodel_model = {
            "id": model_id,
            "name": "YODEL-style packing feature plus ML residual correction",
            "family": "physics_feature_residual_tabular_regression",
            "modeling_idea": (
                "Build fold-local YODEL/packing features from phi and formulation/particle/process columns, "
                "then train a robust sklearn regressor on base plus packing features and compare against the same model without packing features."
            ),
            "required_columns": yodel_columns or ["phi"],
            "strengths": "Matches the available Table 6/synthetic fields better than shear-rate constitutive losses.",
            "risks": "The hidden fmax/phi_m parameter must be fitted inside folds or fixed from literature; reference columns are leakage and cannot be inputs.",
            "expected_artifacts": ["metrics/metrics.json", "logs/mechanism_report.json", "trained_models/*.joblib"],
            "source_ids_or_links": ["LOCAL_YODEL_FLATT_BOWEN_2006", "LOCAL_TABLE6_MATERIALS_18_02983"],
            "source_evidence_map": [],
            "novelty_filter": {
                "accepted": True,
                "reason": "Uses YODEL/packing as an auditable feature layer with residual ML and ablation, not as a fixed local template.",
            },
            "why_not_skill_1_to_5_derivative": "No legacy viscosity Skill1-5 cascade is used.",
            "why_not_fixed_local_template": "The fmax/packing layer must be fitted or ablated inside the current fold protocol.",
            "feasibility_score_1_to_5": 5 if support_status == "supported" else 3,
            "forced_by_manager_audit": True,
        }
        models.append(existing_yodel_model)
    model_id = str(existing_yodel_model.get("id") or "M_YODEL_RESIDUAL")

    strategies = payload.setdefault("candidate_hybrid_strategies", [])
    has_yodel_strategy = any(
        isinstance(item, dict)
        and _mechanism_kind(_candidate_text(item)) == "yodel_packing_fmax"
        for item in strategies
    )
    if not has_yodel_strategy:
        existing_ids = {str(item.get("id")) for item in strategies if isinstance(item, dict)}
        strategy_id = "H_YODEL_PACKING_RESIDUAL"
        suffix = 1
        while strategy_id in existing_ids:
            suffix += 1
            strategy_id = f"H_YODEL_PACKING_RESIDUAL_{suffix}"
        base_score = 9.2 if support_status == "supported" else 7.2
        strategies.append(
            {
                "id": strategy_id,
                "name": "YODEL/packing physical feature layer with ML residual correction",
                "model_id": model_id,
                "mechanism_ids": [mech_id],
                "combination_method": (
                    "Create fold-local packing/fmax/YODEL-style features from allowed columns; train a residual tabular regressor; "
                    "run an ablation against the same regressor without the packing feature layer."
                ),
                "required_columns": yodel_columns or ["phi"],
                "data_support": support_status,
                "data_support_reason": yodel_support.get("reason"),
                "implementation_plan": (
                    "Fit any phi_m/fmax constants inside each CV fold. Do not read phi_max_eff_reference. "
                    "Save mechanism_ablation and explain whether packing features improved OOF/anchor metrics."
                ),
                "expected_benefit": "Uses the mechanism family best aligned with phi/formulation/particle fields.",
                "risks": "Synthetic-data results are low-fidelity; anchor validation must be reported separately.",
                "score_breakdown": {
                    "evidence": 4,
                    "data_support": 5 if support_status == "supported" else 3,
                    "feasibility": 5 if support_status == "supported" else 3,
                    "leakage_safety": 5,
                    "expected_performance": 4,
                    "interpretability": 5,
                    "total_score_1_to_10": base_score,
                },
                "total_score_1_to_10": base_score,
                "selected": False,
                "forced_by_manager_audit": True,
            }
        )

    payload["yodel_packing_audit"] = {
        "candidate_forced": True,
        "support_status": support_status,
        "support_reason": yodel_support.get("reason"),
        "required_decision": "select_if_best_audited_score_else_report_rejection_reason",
        "available_yodel_columns": yodel_columns,
    }
    return True


def _apply_data_capability_audit_to_candidates(
    payload: dict[str, Any],
    data_profile: dict[str, Any],
) -> dict[str, Any]:
    audit = _infer_data_capability_audit(data_profile)
    payload["data_capability_audit"] = audit
    try:
        min_codegen_specs = max(0, int(os.environ.get("YIELD_MIN_LLM_MODEL_SPECS", "2")))
    except Exception:
        min_codegen_specs = 2
    _ensure_llm_model_candidate_quality(payload, data_profile, min_codegen_specs=min_codegen_specs)
    _ensure_yodel_packing_candidate(payload, audit, data_profile)

    schema = data_profile.get("schema", {}) if isinstance(data_profile, dict) else {}
    feature_columns = [str(x) for x in schema.get("feature_columns", []) or []]
    # Readiness checks may require non-feature metadata such as data_fidelity.
    # These columns are still excluded from model inputs by the schema/harness.
    readiness_columns = list(dict.fromkeys(
        feature_columns + [str(x) for x in data_profile.get("columns", []) or []]
    ))

    executor_specs = normalize_executor_specs(payload.get("candidate_executor_specs"), include_defaults=False)
    for spec in executor_specs:
        readiness = executor_spec_readiness(spec, readiness_columns)
        spec["harness_status"] = readiness["status"]
        spec["harness_rejection_reasons"] = readiness["reasons"]
        spec["graph_status"] = readiness.get("graph_status")
        spec["graph_mode"] = readiness.get("graph_mode")
        spec["graph_rejection_reasons"] = readiness.get("graph_reasons", [])
    payload["candidate_executor_specs"] = executor_specs

    fusion_specs = merge_with_default_fusion_specs(payload.get("candidate_fusion_specs"))
    fusion_specs = attach_executor_specs_to_fusion_specs(fusion_specs, executor_specs)
    for spec in fusion_specs:
        readiness = fusion_spec_readiness(spec, readiness_columns)
        reasons = readiness["reasons"]
        spec["harness_status"] = readiness["status"]
        spec["harness_rejection_reasons"] = reasons
        spec["executor_status"] = readiness.get("executor_status")
        spec["executor_rejection_reasons"] = readiness.get("executor_reasons", [])
        if reasons and spec.get("data_support") in (None, "", "supported"):
            spec["data_support"] = "unsupported"
            spec["data_support_reason"] = "; ".join(reasons)
    payload["candidate_fusion_specs"] = fusion_specs

    def infer_fusion_id(strategy: dict[str, Any]) -> str:
        text = _candidate_text(strategy).lower()
        if not strategy.get("mechanism_ids"):
            return "raw_ml"
        if "residual" in text or "y - base" in text or "base + residual" in text:
            return "mechanism_residual"
        if "feature" in text or "append" in text or "augmentation" in text or "derived" in text:
            return "mechanism_features"
        return "mechanism_features"

    for strategy in payload.get("candidate_hybrid_strategies", []) or []:
        if isinstance(strategy, dict) and not strategy.get("fusion_id"):
            strategy["fusion_id"] = infer_fusion_id(strategy)

    mechanism_by_id: dict[str, dict[str, Any]] = {}
    for mechanism in payload.get("candidate_mechanisms", []) or []:
        if not isinstance(mechanism, dict):
            continue
        text = " ".join(
            str(mechanism.get(key, ""))
            for key in ("name", "formula_or_relationship", "applicability_to_current_data", "paper_or_source")
        )
        status, reason, score_cap, kind = _mechanism_support_from_audit(text, audit)
        current_score = mechanism.get("feasibility_score_1_to_5")
        try:
            current_score_int = int(float(current_score))
        except Exception:
            current_score_int = score_cap
        mechanism["data_capability_audit"] = {
            "mechanism_kind": kind,
            "support_status": status,
            "support_reason": reason,
            "score_cap_1_to_5": score_cap,
        }
        mechanism["feasibility_score_1_to_5"] = min(current_score_int, score_cap)
        if status == "unsupported":
            mechanism["role_options"] = ["not_applicable"]
            mechanism["applicability_to_current_data"] = reason
        mechanism_id = str(mechanism.get("id") or "")
        if mechanism_id:
            mechanism_by_id[mechanism_id] = mechanism

    best_supported: dict[str, Any] | None = None
    best_supported_score = -1.0
    selected_was_reset = False
    for strategy in payload.get("candidate_hybrid_strategies", []) or []:
        if not isinstance(strategy, dict):
            continue
        strategy_text = _candidate_text(strategy)
        mech_ids = [str(x) for x in strategy.get("mechanism_ids", []) or []]
        statuses = []
        reasons = []
        kinds = []
        for mech_id in mech_ids:
            mech = mechanism_by_id.get(mech_id)
            audit_info = mech.get("data_capability_audit", {}) if isinstance(mech, dict) else {}
            status = str(audit_info.get("support_status", "partial"))
            statuses.append(status)
            kinds.append(str(audit_info.get("mechanism_kind", "unknown")))
            reason = str(audit_info.get("support_reason", ""))
            if reason:
                reasons.append(f"{mech_id}: {reason}")
        inferred_kind = _mechanism_kind(strategy_text)
        if not statuses and inferred_kind != "unknown":
            status, reason, _score_cap, kind = _mechanism_support_from_audit(strategy_text, audit)
            statuses.append(status)
            kinds.append(kind)
            if reason:
                reasons.append(f"strategy_text: {reason}")
        includes_hb = "herschel_bulkley_bingham" in kinds
        includes_yodel = "yodel_packing_fmax" in kinds
        yodel_status = str(audit.get("mechanism_support", {}).get("yodel_packing_fmax", {}).get("status", "unsupported"))
        yodel_available = yodel_status in {"supported", "partial"}
        if "unsupported" in statuses:
            was_selected = bool(strategy.get("selected"))
            strategy["data_support"] = "unsupported"
            strategy["selected"] = False
            if was_selected:
                selected_was_reset = True
            strategy["rejection_reason"] = (
                "Rejected by data-capability audit. " + " | ".join(reasons)
            ).strip()
            score_cap = 2.0 if includes_hb else 3.5
        elif "partial" in statuses:
            if strategy.get("data_support") == "supported":
                strategy["data_support"] = "partial"
            strategy["data_support_reason"] = (
                str(strategy.get("data_support_reason") or "").strip()
                + (" | " if strategy.get("data_support_reason") and reasons else "")
                + " | ".join(reasons)
            ).strip()
            score_cap = 7.4
        elif not statuses:
            strategy["data_support"] = strategy.get("data_support") or "data_driven_no_mechanism"
            score_cap = 6.8 if yodel_available else 8.0
        else:
            score_cap = 9.6

        strategy["data_capability_audit"] = {
            "mechanism_kinds": kinds,
            "support_statuses": statuses,
            "support_reasons": reasons,
            "deterministic_policy": (
                "HB/Bingham residuals require shear-rate curves; YODEL/packing/fmax is prioritized when supported by phi plus formulation/particle/process columns."
            ),
        }
        breakdown = strategy.get("score_breakdown")
        if not isinstance(breakdown, dict):
            breakdown = {}
            strategy["score_breakdown"] = breakdown
        if statuses:
            breakdown["data_support"] = 1 if "unsupported" in statuses else 3 if "partial" in statuses else 5
        score = _safe_float(strategy.get("total_score_1_to_10", breakdown.get("total_score_1_to_10", 0)), default=0.0)
        if score <= 0:
            score = 5.0
        score = min(score, score_cap)
        if includes_yodel and yodel_status == "supported":
            score = max(score, 9.1)
            breakdown["data_support"] = 5
            breakdown["mechanism_priority_bonus"] = 1.2
            strategy["data_support"] = "supported"
            strategy["data_support_reason"] = audit.get("mechanism_support", {}).get("yodel_packing_fmax", {}).get("reason", "")
        elif includes_yodel and yodel_status == "partial":
            score = max(score, 7.1)
            breakdown["mechanism_priority_bonus"] = 0.6
            strategy["data_support"] = "partial"
        if includes_hb and "unsupported" in statuses:
            score = min(score, 2.0)
            breakdown["unsupported_constitutive_penalty"] = -3.0
        score = round(float(score), 3)
        strategy["deterministic_data_score_1_to_10"] = score
        strategy["total_score_1_to_10"] = score
        breakdown["total_score_1_to_10"] = score
        if strategy.get("data_support") != "unsupported" and score > best_supported_score:
            best_supported = strategy
            best_supported_score = score

    selected = payload.get("selected_combination")
    selected_strategy_id = str(selected.get("strategy_id", "")) if isinstance(selected, dict) else ""
    selected_strategy = next(
        (
            item
            for item in payload.get("candidate_hybrid_strategies", []) or []
            if isinstance(item, dict) and str(item.get("id")) == selected_strategy_id
        ),
        None,
    )
    if selected_strategy is not None and selected_strategy.get("data_support") == "unsupported":
        selected_was_reset = True
    elif selected_strategy is None and best_supported is not None:
        selected_was_reset = True
    elif selected_strategy is not None and best_supported is not None and str(best_supported.get("id")) != selected_strategy_id:
        selected_score = _safe_float(
            selected_strategy.get("deterministic_data_score_1_to_10", selected_strategy.get("total_score_1_to_10")),
            default=0.0,
        )
        if best_supported_score > selected_score + 0.25:
            selected_was_reset = True
    if selected_was_reset and best_supported is not None:
        for item in payload.get("candidate_hybrid_strategies", []) or []:
            if isinstance(item, dict):
                is_selected = str(item.get("id")) == str(best_supported.get("id"))
                item["selected"] = is_selected
                if is_selected:
                    item.pop("rejection_reason", None)
                    item["selected_by_data_capability_audit"] = True
        payload["selected_combination"] = {
            "strategy_id": best_supported.get("id"),
            "model_id": best_supported.get("model_id"),
            "mechanism_ids": best_supported.get("mechanism_ids", []),
            "uses_mechanism": bool(best_supported.get("mechanism_ids")),
            "selection_rationale": (
                "Selected after deterministic data-capability audit. "
                "Mechanisms with missing required columns were down-ranked or rejected; "
                "the remaining strategy has the best audited score."
            ),
            "fallback_if_mechanism_not_applicable": "Use the same model family without the unsupported mechanism and report mechanism_used=false.",
            "evaluation_protocol": "Primary 5-fold OOF plus Table 6 anchor validation when available.",
        }
        payload["selection_audit_note"] = (
            "LLM-selected strategy was reset by deterministic data-capability scoring. "
            "Supported YODEL/packing/fmax candidates are prioritized over unsupported HB/Bingham/PINN residuals."
        )
    elif selected_was_reset and best_supported is None:
        payload["selected_combination"] = {
            "strategy_id": None,
            "model_id": None,
            "mechanism_ids": [],
            "uses_mechanism": False,
            "selection_rationale": "No supported hybrid strategy survived data-capability audit; use pure model fallback.",
            "fallback_if_mechanism_not_applicable": "Train generated data-driven candidates and report mechanisms as rejected.",
            "evaluation_protocol": "Primary 5-fold OOF plus Table 6 anchor validation when available.",
        }
        payload["selection_audit_note"] = "All hybrid strategies were unsupported by current columns."
    else:
        for item in payload.get("candidate_hybrid_strategies", []) or []:
            if isinstance(item, dict):
                item["selected"] = str(item.get("id")) == selected_strategy_id
        payload.setdefault("selection_audit_note", "LLM selection passed deterministic data-capability audit.")

    selected = payload.get("selected_combination")
    selected_strategy_id = str(selected.get("strategy_id", "")) if isinstance(selected, dict) else ""
    selected_strategy = next(
        (
            item
            for item in payload.get("candidate_hybrid_strategies", []) or []
            if isinstance(item, dict) and str(item.get("id")) == selected_strategy_id
        ),
        None,
    )
    selected_kinds = []
    if isinstance(selected_strategy, dict):
        selected_audit = selected_strategy.get("data_capability_audit", {})
        if isinstance(selected_audit, dict):
            selected_kinds.extend(str(x) for x in selected_audit.get("mechanism_kinds", []) or [])
        for mech_id in selected_strategy.get("mechanism_ids", []) or []:
            mech = mechanism_by_id.get(str(mech_id))
            if isinstance(mech, dict):
                mech_audit = mech.get("data_capability_audit", {})
                if isinstance(mech_audit, dict):
                    selected_kinds.append(str(mech_audit.get("mechanism_kind", "")))
    yodel_selected = bool(
        "yodel_packing_fmax" in selected_kinds
        or (
            selected_strategy is not None
            and _mechanism_kind(_candidate_text(selected_strategy)) == "yodel_packing_fmax"
        )
    )
    yodel_audit = payload.setdefault("yodel_packing_audit", {})
    yodel_audit.update(
        {
            "selected": yodel_selected,
            "selected_strategy_id": selected_strategy_id or None,
            "must_report_if_not_selected": str(yodel_audit.get("support_status", "unsupported")) != "unsupported",
            "not_selected_reporting_requirement": (
                "If selected=false while support_status is supported/partial, mechanism_report.json must include a rejected_yodel_packing_candidate entry with a concrete data, performance, or numerical-stability reason."
            ),
        }
    )

    return payload


def run_search_stage(user_prompt: str, data_profile: dict[str, Any], enabled: bool, extra_queries: list[str] | None) -> dict[str, Any]:
    queries = list(DEFAULT_YIELD_SEARCH_QUERIES)
    if extra_queries:
        queries.extend(extra_queries)
    if user_prompt:
        queries.append(f"{user_prompt} yield stress prediction machine learning rheology")
    source_payload = _try_external_search(enabled, user_prompt, queries)
    snippets = source_payload.get("sources", [])
    return {
        "stage": "search",
        "enabled": bool(enabled),
        "user_prompt": user_prompt,
        "queries": queries,
        "snippets": snippets,
        "source_payload": source_payload,
        "provider_summary": source_payload.get("provider_summary", {}),
        "source_type_summary": source_payload.get("source_type_summary", {}),
        "source_quality": source_payload.get("source_quality", {}),
        "data_schema": data_profile.get("schema", {}).get("source_schema"),
        "data_columns": data_profile.get("columns", []),
        "note": (
            "External search returned snippets." if snippets else
            "No external snippets were available; using local design references only."
        ),
    }


def run_model_plan_stage(
    llm: str,
    user_prompt: str,
    data_profile: dict[str, Any],
    search_report: dict[str, Any],
    candidate_report: dict[str, Any] | None = None,
    manager_feedback: str = "",
) -> dict[str, Any]:
    prompt_search_report = _compact_search_report_for_prompt(search_report, max_snippets=18)
    prompt = f"""
You are the ModelAgent for yield-stress AutoML. Create one implementation-ready
modeling plan for OperationAgent. Do not write code.

User prompt:
{user_prompt}

Data profile:
{json.dumps(data_profile, ensure_ascii=False, indent=2)}

Data-aware model design guidance:
{json.dumps(_model_design_guidance_from_profile(data_profile), ensure_ascii=False, indent=2)}

Search report:
{json.dumps(prompt_search_report, ensure_ascii=False, indent=2)}

Candidate report:
{json.dumps(candidate_report or {}, ensure_ascii=False, indent=2)}

Manager revision feedback from previous failed execution, if any:
{manager_feedback or "None. This is the first planning round."}

Requirements:
- Keep the final model flexible and runnable in the current repository.
- Use candidate_hybrid_strategies as the primary decision space when available.
  A hybrid strategy explicitly combines one model candidate with one or more
  mechanism candidates. Select it only when the current columns support it.
- If manager revision feedback is present, address it explicitly by changing
  the selected approach, fallback package choices, artifact plan, or mechanism
  usage instead of repeating the failed plan unchanged.
- Treat the deterministic data-capability audit as higher priority than model
  complexity. HB/Bingham/PINN constitutive residuals are not valid unless
  shear-rate or flow-curve columns exist. Packing/fmax/YODEL-style candidates
  should be preferred when phi plus formulation, particle, process, or
  packing-related columns are available.
- Treat candidate_selection_audit and each strategy's benchmark_audit as
  run-local measured evidence. They are not fixed templates, but they are
  stronger than raw LLM preference. If the benchmark-selected strategy is not
  planned, rejected_hybrid_strategies must give a concrete data, performance,
  package, or numerical-stability reason.
- If candidate_selection_audit.status is no_viable_anchor_candidate or
  selection_blocked_by_anchor_viability=true, do not plan OperationAgent
  implementation of a failed-anchor strategy. Revise candidate direction toward
  anchor-compatible feature subsets, synthetic reweighting, or data-generation
  fixes.
- Preserve alignment between CandidateAgent selection, benchmark proxy, and the
  planned implementation. Include selected_hybrid_strategy and
  benchmark_alignment in the plan. If the planned model differs from the
  benchmark-selected strategy or proxy, state the concrete reason.
- Plan to run the benchmark-selected proxy model and feature mode as an actual
  candidate in OperationAgent. Additional stronger models may be tried, but the
  benchmark-selected proxy must have its own OOF and anchor metrics.
- Plan an implementation_validation report that compares the final implemented
  model against the benchmark-selected strategy using benchmark_oof_rmse,
  actual_oof_rmse, benchmark_anchor_r2 when available, actual_anchor_r2 when
  available, generalization_gap, and validation_status.
- Plan a fit/transform preprocessing pipeline. It must keep train, validation,
  and anchor in the same engineered feature space and save
  preprocessing/preprocessing_audit.json.
- If no hybrid strategy is supported, explain why and then select the best
  pure model fallback.
- You may choose a data-driven, mechanistic, physics-informed, hybrid, ensemble,
  or multi-fidelity approach.
- If recommending a mechanistic model, name the paper/source and exact formula
  that OperationAgent must document in mechanism_report.json.
- When external snippets are available, mechanisms_to_use must cite searched
  source_id values and should include the searched title/link/DOI for each
  mechanistic claim. Local design notes may be background, but they are not a
  substitute for external source evidence when SearchAgent found snippets.
- If recommending no mechanistic model, explain why.
- Explicitly list forbidden feature columns, candidate model families to try,
  selected model rationale, and evaluation protocol.
- The evaluation protocol must be primary 5-fold OOF plus exactly one secondary
  evaluation. Prefer anchor_validation when a real anchor dataset exists.
Return concise JSON with keys: selected_approach, rationale, mechanisms_to_use,
selected_hybrid_strategy, rejected_hybrid_strategies, required_proxy_candidate,
features, target, evaluation, feature_pipeline, benchmark_alignment,
implementation_validation, artifacts.
"""
    try:
        content = _chat_completion_content_with_process_timeout(
            "ModelAgent LLM planning",
            llm,
            [
                {"role": "system", "content": "You are a pragmatic ML research planning agent."},
                {"role": "user", "content": prompt},
            ],
            0.4,
        )
    except Exception as exc:
        content = json.dumps(
            {
                "selected_approach": "LLM-generated flexible regression model",
                "rationale": f"Model planning LLM unavailable: {type(exc).__name__}: {exc}. OperationAgent should choose a runnable approach from local context.",
                "mechanisms_to_use": "optional; document any used mechanism in mechanism_report.json",
                "features": data_profile.get("schema", {}).get("feature_columns", []),
                "target": "yield_stress",
                "evaluation": "primary 5-fold OOF on active training data plus exactly one secondary evaluation; use anchor_validation when a real anchor CSV exists",
                "artifacts": ["metrics/metrics.json", "predictions/yield_predictions.csv", "trained_models/", "preprocessing/preprocessing_audit.json", "preprocessing/", "logs/mechanism_report.json", "logs/synthetic_data_report.json when synthetic data are active"],
            },
            ensure_ascii=False,
            indent=2,
        )
    return {
        "stage": "model_plan",
        "llm": llm,
        "content": content,
    }


def run_candidate_stage(
    llm: str,
    user_prompt: str,
    data_profile: dict[str, Any],
    search_report: dict[str, Any],
    manager_feedback: str = "",
) -> dict[str, Any]:
    prompt_search_report = _compact_search_report_for_prompt(search_report, max_snippets=20)
    prompt = f"""
You are the CandidateAgent for yield-stress AutoML.

Generate candidate model families, candidate mechanisms, candidate fusion specs
(how a model and a mechanism combine), optional executor specs (how the fixed
harness would execute a non-builtin fusion), and explicit model-mechanism hybrid
strategies from external search snippets and the current data schema. Do not
write executable code in this JSON; OperationAgent will turn your
model/mechanism briefs into bounded code under separate safety contracts.

Workflow you must follow:
1. Extract literature model/mechanism primitives from source_id evidence.
2. Generate 5 candidate model families: at least Random Forest and
   HistGradientBoosting as standard baselines, and at least 2 codegen-oriented
   model ideas that are not simple RF/HGB/Ridge/MLP wrappers.
3. Generate 3-5 candidate mechanisms.
4. Generate 2-5 candidate_fusion_specs describing how a model and mechanism are
   combined. Use executable types raw_ml, mechanism_features, or
   mechanism_residual unless you are explicitly proposing a planned-only future
   structure with clear missing executor/data requirements.
5. If a fusion_spec is not one of the executable builtin types, generate a
   matching candidate_executor_spec. It must be declarative only: executor_type,
   executor_graph, required_columns, fit_scope, prediction_rule,
   leakage_safety_notes, and any latent/physics/data fields needed. Do not
   include Python/source code.
6. Generate 3-5 candidate_hybrid_strategies that combine a model, a fusion_spec,
   and one or
   more mechanisms.
   At least one hybrid must be a YODEL/packing/fmax-style candidate when the
   data profile contains phi plus formulation, particle, process, or
   packing-related columns.
7. Score all hybrid strategies and judge whether current data supports them.
   Synthetic low-fidelity data can support model-structure exploration, but a
   real anchor dataset must be used only for secondary validation.
8. Select the best supported hybrid strategy. If none is supported, select the
   best pure model fallback and explain every rejected hybrid.
9. If manager feedback or candidate_selection_audit reports
   no_viable_anchor_candidate, all_candidates_failed_anchor_validation, negative
   anchor R2, or selected_strategy_anchor_generalization_status=failed, do not
   merely rename the previous strategy. Generate targeted recovery candidates:
   an anchor-compatible shared-feature candidate using only Table 6-supported
   fields such as phi, sp_percent, w_b, and fa_ratio when present; a synthetic
   reweighting or anchor-neighborhood candidate; a simple real-anchor sanity
   baseline candidate; and a mechanism no-gain fallback that can reject packing
   features when ablation/anchor evidence is poor.

User prompt:
{user_prompt}

Data profile:
{json.dumps(data_profile, ensure_ascii=False, indent=2)}

Search report:
{json.dumps(prompt_search_report, ensure_ascii=False, indent=2)}

Manager revision feedback from previous failed execution, if any:
{manager_feedback or "None. This is the first candidate-generation round."}

Return strict JSON with:
- model_primitives: 3 to 8 items extracted from literature. Each item needs id,
  primitive, source_ids_or_links, transferable_use_for_yield_stress, constraints.
- candidate_models: 5 items. Each item needs id, name, family,
  modeling_idea, required_columns, strengths, risks, expected_artifacts,
  source_ids_or_links, source_evidence_map, novelty_filter,
  why_not_skill_1_to_5_derivative, why_not_fixed_local_template,
  feasibility_score_1_to_5. Make modeling_idea concrete enough that a bounded
  estimator factory can be generated later (algorithm components + intended
  hyperparameters), not just a generic model name.
  Exactly identify standard baselines with novelty_filter="standard_baseline".
  For at least 2 non-baseline model ideas, set codegen_intent="llm_model" and
  make them concrete enough for OperationAgent to generate MODEL_SPEC +
  make_estimator(params, random_state). Good examples are kernel/SVR/GPR,
  compact feature-group neural models, or physics-input residual estimators.
  These LLM-codegen model ideas must describe a bounded model structure, not
  plain StandardScaler + one regressor. Prefer low-order interaction features,
  regularized kernels, small voting/stacking ensembles, or explicit residual
  model capacity that can be expressed as a sklearn estimator factory.
  Use the Data-aware model design guidance above. For small-sample/high-dimensional
  data, prefer regularized and low-capacity structures; if phi is constant, do
  not make the model depend mainly on phi-only packing variation.
- candidate_mechanisms: 3 to 5 items. Each item needs id, name, paper_or_source,
  formula_or_relationship, required_columns, role_options
  (feature/loss/baseline/constraint/not_applicable), applicability_to_current_data,
  source_ids_or_links, source_evidence_map, feasibility_score_1_to_5.
- candidate_fusion_specs: 2 to 5 items. Each item needs id, name, type,
  description, requires_mechanism true/false, required_columns,
  data_support (supported/partial/unsupported), data_support_reason,
  leakage_safety_notes, expected_benefit, risks, and optionally executor_spec.
  Executable type values today: raw_ml, mechanism_features, mechanism_residual.
  Other types may be proposed only as planned_only and must explain why the
  current harness should not execute them yet.
- candidate_executor_specs: 0 to 5 items. Each item needs id, fusion_id,
  executor_type, description, executor_graph, required_columns, fit_scope,
  prediction_rule, leakage_safety_notes. executor_graph.steps must be a list of
  declarative operations such as select_raw_features, fit_mechanism_params,
  compute_mechanism_features, compute_mechanism_base, fit_model,
  predict_model, fit_residual_model, predict_residual_model, add_predictions,
  infer_latent_targets, fit_latent_model, predict_latent_model,
  compute_physics_output, fit_latent_physics_model,
  predict_latent_physics_model, fit_low_fidelity_base,
  fit_high_fidelity_residual. Planned executor_type values you may propose but
  the current harness will not execute yet: latent_parameter_regressor,
  hidden_parameter_physics_layer, physics_loss_regularizer,
  multi_fidelity_base_residual. These are judged for safety/data support and
  recorded; do not include code/source_code/python_code.
- candidate_hybrid_strategies: 3 to 5 items. Each item needs id, name, model_id,
  fusion_id, mechanism_ids, combination_method, required_columns, data_support
  (supported/partial/unsupported), data_support_reason, implementation_plan,
  expected_benefit, risks, score_breakdown with keys evidence, data_support,
  feasibility, leakage_safety, expected_performance, interpretability,
  total_score_1_to_10, selected true/false, rejection_reason if not selected.
- selected_combination: strategy_id, model_id, mechanism_ids,
  uses_mechanism true/false, selection_rationale,
  fallback_if_mechanism_not_applicable, evaluation_protocol.

Important:
- Models and mechanisms are separate inputs, but the final decision should be a
  model-mechanism hybrid strategy whenever data support allows it.
- If manager revision feedback is present, update the candidate list and
  selected_combination to avoid the previous execution/guardrail failure.
- If manager revision feedback says all candidates failed anchor validation or
  no viable anchor candidate exists, set the prior failed strategy to rejected
  and create candidates designed around synthetic-anchor distribution shift.
  Prefer anchor-compatible feature subsets and reweighting over adding more
  complex model families.
- Do not select Herschel-Bulkley, Bingham, Casson, or PINN constitutive-loss
  strategies as the top hybrid when gamma_dot/shear-rate/flow-curve columns are
  absent. They can be background or rejected candidates only.
- If YODEL/packing/fmax is not selected despite phi plus formulation/particle
  support, selected_combination.selection_rationale must state the concrete
  reason.
- Use data_capability_audit before scoring. Mechanisms that lack required
  columns must be partial/unsupported even if the paper is strong or the model
  sounds complex.
- Do not over-prefer PINN + constitutive residuals only because they look
  sophisticated. HB/Bingham/Casson residuals require shear-rate or flow-curve
  columns. Without those columns, they are background mechanisms only.
- Do not force YODEL. However, when phi plus particle/formulation/packing
  descriptors exist, packing/fmax/YODEL-style mechanisms should be considered
  fairly against other mechanisms.
- If current columns do not support a mechanism, keep it as not_applicable and
  do not force it into a selected hybrid.
- Lightweight mechanisms such as a superplasticizer monotonic/decay relation or
  packing-inspired transformed feature may be supported when columns exist.
- Cite searched titles/links when snippets are available.
- Mechanism candidates must cite exact searched source_ids. Do not cite only
  author/year when a source_id, title, link, or DOI is available.
- Apply a novelty filter. For this yield-stress project, novelty means the
  selected model must not be a simple copy of legacy Skill1-5 templates
  and must not be just a fixed local YODEL/Lian reference formula. It may use
  external literature primitives if source_id evidence is explicit.
- The novelty filter applies to MECHANISMS and to avoiding legacy templates, NOT
  to excluding standard ML models. Standard strong tabular baselines are VALID and
  ENCOURAGED candidate_models for this small tabular dataset and NEED NOT be novel:
  Random Forest, HistGradientBoosting/GBM (incl. XGBoost-style), ExtraTrees, SVR,
  Ridge, MLP. Include at least a Random Forest and a gradient-boosting model among
  candidate_models so the search can find the strongest simple baseline; for such
  standard models set novelty_filter to "standard_baseline" and
  why_not_fixed_local_template to "standard ML baseline, not a template".
- Standard baselines are evaluated by the fixed harness floor. Do not spend all
  candidate_models on those baselines; include at least 2 LLM-codegen model briefs
  that test genuinely different structures.
"""
    fallback = {
        "stage": "candidate_generation",
        "model_primitives": [
            {
                "id": "P0",
                "primitive": "Small-data nonlinear tabular regression with physically valid non-negative outputs.",
                "source_ids_or_links": [],
                "transferable_use_for_yield_stress": "Use as a safe fallback when mechanisms are under-supported by columns.",
                "constraints": "Must avoid target leakage and report mechanism_used=false if no equation is used.",
            }
        ],
        "candidate_models": [
            {
                "id": "M1",
                "name": "Tree ensemble regression",
                "family": "random_forest_or_gradient_boosting",
                "modeling_idea": "Use phi and sp_percent with nonlinear tree interactions.",
                "required_columns": data_profile.get("schema", {}).get("feature_columns", []),
                "strengths": "Robust on small tabular datasets.",
                "risks": "Limited extrapolation and weak physical interpretability.",
                "expected_artifacts": ["metrics/metrics.json", "trained_models/*.joblib"],
                "source_ids_or_links": [],
                "source_evidence_map": [],
                "novelty_filter": {"accepted": True, "reason": "Fallback is not a legacy Skill1-5 derivative; it is a generic small-data tabular model."},
                "why_not_skill_1_to_5_derivative": "Does not use the legacy two-target cascade, Skill1-5 equations, or old deterministic templates.",
                "why_not_fixed_local_template": "Does not hard-code local YODEL/Lian equations.",
                "feasibility_score_1_to_5": 5,
            },
            {
                "id": "M2",
                "name": "Gaussian Process regression",
                "family": "kernel_method",
                "modeling_idea": "Use an RBF/Matern kernel for uncertainty-aware small-data regression.",
                "required_columns": data_profile.get("schema", {}).get("feature_columns", []),
                "strengths": "Works well for small smooth datasets and gives uncertainty.",
                "risks": "May scale poorly and needs careful kernel normalization.",
                "expected_artifacts": ["metrics/metrics.json", "trained_models/*.joblib"],
                "source_ids_or_links": [],
                "source_evidence_map": [],
                "novelty_filter": {"accepted": True, "reason": "Kernel uncertainty model is independent from legacy Skill1-5 templates."},
                "why_not_skill_1_to_5_derivative": "No legacy two-target cascade or Skill1-5 mechanism is used.",
                "why_not_fixed_local_template": "Uses data-driven kernels, not a fixed local yield equation.",
                "feasibility_score_1_to_5": 4,
            },
            {
                "id": "M3",
                "name": "Physics-feature hybrid model",
                "family": "hybrid",
                "modeling_idea": "Create physics-inspired features only if supported by current columns, then fit a regularized regressor.",
                "required_columns": data_profile.get("schema", {}).get("feature_columns", []),
                "strengths": "Can combine interpretability and accuracy.",
                "risks": "Mechanism may be under-supported by available columns.",
                "expected_artifacts": ["logs/mechanism_report.json", "metrics/metrics.json"],
                "source_ids_or_links": [],
                "source_evidence_map": [],
                "novelty_filter": {"accepted": True, "reason": "Hybrid feature construction is allowed only if supported by external evidence/current columns."},
                "why_not_skill_1_to_5_derivative": "No legacy Skill1-5 model structure is reused.",
                "why_not_fixed_local_template": "Mechanism is optional and must be externally justified if used.",
                "feasibility_score_1_to_5": 3,
            },
        ],
        "candidate_fusion_specs": [
            {
                "id": "raw_ml",
                "name": "Raw ML baseline",
                "type": "raw_ml",
                "description": "Train the model directly on raw numeric features.",
                "requires_mechanism": False,
                "required_columns": [],
                "data_support": "supported",
                "data_support_reason": "Always needed as the no-mechanism comparison floor.",
                "leakage_safety_notes": "Uses only schema-approved feature columns.",
                "expected_benefit": "Provides the baseline for judging whether any mechanism/fusion adds value.",
                "risks": "Weak physical interpretability.",
                "origin": "fallback",
            },
            {
                "id": "mechanism_features",
                "name": "Mechanism feature augmentation",
                "type": "mechanism_features",
                "description": "Append mechanism-derived fold-local features to raw features before fitting the model.",
                "requires_mechanism": True,
                "required_columns": ["phi"],
                "data_support": "partial",
                "data_support_reason": "Executable when a mechanism passes data adequacy; quality depends on available physics inputs.",
                "leakage_safety_notes": "Mechanism parameters must be fitted inside each training fold only.",
                "expected_benefit": "Lets simple mechanisms contribute interpretable predictors without forcing the final model form.",
                "risks": "May only add weak process proxies when microstructure columns are absent.",
                "origin": "fallback",
            },
            {
                "id": "mechanism_residual",
                "name": "Mechanism residual correction",
                "type": "mechanism_residual",
                "description": "Use a fold-local mechanism base prediction and train the model on y minus that base.",
                "requires_mechanism": True,
                "required_columns": ["phi"],
                "data_support": "partial",
                "data_support_reason": "Executable when a mechanism can produce base_predict; effectiveness depends on mechanism quality.",
                "leakage_safety_notes": "The residual target is built only inside training folds; validation prediction never uses validation y.",
                "expected_benefit": "Tests whether the mechanism explains a stable component and ML only needs to correct the remainder.",
                "risks": "A poor mechanism base can make residual learning noisier than raw ML.",
                "origin": "fallback",
            },
        ],
        "candidate_executor_specs": [
            {
                "id": "raw_ml_executor",
                "fusion_id": "raw_ml",
                "executor_type": "builtin_raw_ml",
                "description": "Fit the selected model directly on schema-approved raw numeric features.",
                "required_columns": [],
                "fit_scope": "fold_local",
                "prediction_rule": "Prediction uses only fitted preprocessing/model state and input X; no validation/test targets are used.",
                "leakage_safety_notes": "All model fitting occurs inside each training fold only.",
                "origin": "fallback",
            },
            {
                "id": "mechanism_features_executor",
                "fusion_id": "mechanism_features",
                "executor_type": "builtin_mechanism_features",
                "description": "Fit mechanism parameters inside the train fold, append mechanism-derived features, then fit the selected model.",
                "required_columns": [],
                "fit_scope": "fold_local",
                "prediction_rule": "Prediction computes mechanism features from X and train-fold fitted mechanism parameters only; no validation/test targets are used.",
                "leakage_safety_notes": "Mechanism parameters and model parameters are fitted inside each training fold only.",
                "origin": "fallback",
            },
            {
                "id": "mechanism_residual_executor",
                "fusion_id": "mechanism_residual",
                "executor_type": "builtin_mechanism_residual",
                "description": "Fit a mechanism base predictor inside the train fold and train the model on y minus that base.",
                "required_columns": [],
                "fit_scope": "fold_local",
                "prediction_rule": "Prediction computes physics base from X, predicts residual from X, and adds them; no validation/test targets are used.",
                "leakage_safety_notes": "Residual targets are created only inside each training fold.",
                "origin": "fallback",
            },
        ],
        "candidate_mechanisms": [
            {
                "id": "P1",
                "name": "YODEL-like suspension scaling",
                "paper_or_source": "external search or local fallback required",
                "formula_or_relationship": "yield stress increases nonlinearly with packing/solid fraction and may depend on particle size.",
                "required_columns": ["phi", "particle_size_or_packing_proxy"],
                "role_options": ["feature", "baseline", "not_applicable"],
                "applicability_to_current_data": "Partial: phi exists, but particle size/packing descriptors may be missing.",
                "source_ids_or_links": [],
                "source_evidence_map": [],
                "feasibility_score_1_to_5": 2,
            }
        ],
        "candidate_hybrid_strategies": [
            {
                "id": "H1",
                "name": "Tree ensemble with physics-audit constraints",
                "model_id": "M1",
                "fusion_id": "raw_ml",
                "mechanism_ids": ["P1"],
                "combination_method": "Use robust tree ensemble as predictor; only audit YODEL-like mechanism as not implemented because required packing proxies are missing.",
                "required_columns": data_profile.get("schema", {}).get("feature_columns", []),
                "data_support": "partial",
                "data_support_reason": "phi exists, but particle size or explicit packing descriptors may be missing; fallback does not implement a fitted hidden packing parameter.",
                "implementation_plan": "Train data-driven tree ensemble and document rejected mechanism; do not set mechanism_used=true.",
                "expected_benefit": "Safe baseline without leakage.",
                "risks": "Does not yet combine mechanism in code.",
                "score_breakdown": {
                    "evidence": 2,
                    "data_support": 2,
                    "feasibility": 5,
                    "leakage_safety": 5,
                    "expected_performance": 4,
                    "interpretability": 2,
                },
                "total_score_1_to_10": 6,
                "selected": True,
            }
        ],
        "selected_combination": {
            "strategy_id": "H1",
            "model_id": "M1",
            "mechanism_ids": [],
            "uses_mechanism": False,
            "selection_rationale": "Fallback: choose robust tabular ML because candidate generation LLM was unavailable.",
            "fallback_if_mechanism_not_applicable": "Use data-driven model and document mechanisms as considered but not used.",
            "evaluation_protocol": "Primary 5-fold OOF plus anchor_validation when a real anchor CSV exists.",
        },
    }
    try:
        content = _chat_completion_content_with_process_timeout(
            "CandidateAgent LLM candidate generation",
            llm,
            [
                {"role": "system", "content": "Return only strict JSON. You are a pragmatic ML candidate selection agent."},
                {"role": "user", "content": prompt},
            ],
            0.5,
        )
        if content.startswith("```"):
            content = content.split("```", 2)[1].replace("json", "", 1).strip()
        payload = json.loads(content)
        if not isinstance(payload, dict):
            raise ValueError("candidate payload is not a JSON object")
    except Exception as exc:
        payload = fallback
        payload["note"] = f"Candidate generation fallback used: {type(exc).__name__}: {exc}"
    payload.setdefault("stage", "candidate_generation")
    return _apply_data_capability_audit_to_candidates(payload, data_profile)


def _profile_dataset(path: str) -> dict[str, Any]:
    df, meta = load_yield_dataframe(path)
    numeric = df.select_dtypes(include="number")
    describe = {}
    for col in numeric.columns:
        series = numeric[col]
        describe[col] = {
            "count": int(series.count()),
            "mean": float(series.mean()),
            "std": float(series.std()) if len(series) > 1 else 0.0,
            "min": float(series.min()),
            "max": float(series.max()),
        }
    profile = {
        "path": path,
        "schema": meta,
        "columns": df.columns.tolist(),
        "n_rows": int(len(df)),
        "numeric_summary": describe,
    }
    profile["data_capability_audit"] = _infer_data_capability_audit(profile)
    return profile


def _brief_data_profile(profile: dict[str, Any]) -> str:
    schema = profile.get("schema", {}) if isinstance(profile, dict) else {}
    features = schema.get("feature_columns", [])
    target = schema.get("target_column", "yield_stress")
    numeric = profile.get("numeric_summary", {}) if isinstance(profile, dict) else {}
    parts = [
        f"Dataset profile: schema={schema.get('source_schema', 'unknown')}, rows={profile.get('n_rows', 'unknown')}",
        f"Features={features}, target={target}",
    ]
    audit = profile.get("data_capability_audit", {})
    support = audit.get("mechanism_support", {}) if isinstance(audit, dict) else {}
    if support:
        parts.append(
            "Mechanism support: "
            + "; ".join(
                f"{key}={value.get('status')}"
                for key, value in support.items()
                if isinstance(value, dict)
            )
        )
    for key in list(features) + [target]:
        stats = numeric.get(key)
        if not isinstance(stats, dict):
            continue
        parts.append(
            f"{key}: mean={stats.get('mean', 0):.4g}, min={stats.get('min', 0):.4g}, max={stats.get('max', 0):.4g}"
        )
    notes = schema.get("notes") or []
    if notes:
        parts.append("Notes: " + " | ".join(str(item) for item in notes[:3]))
    return "\n".join(parts)


def _brief_requirement_analysis(user_prompt: str, profile: dict[str, Any]) -> str:
    schema = profile.get("schema", {}) if isinstance(profile, dict) else {}
    features = schema.get("feature_columns", [])
    target = schema.get("target_column", "yield_stress")
    return (
        "I understand the request as a yield-stress Research Mode run.\n"
        "- First perform external retrieval across Web Search/SearchAPI, arXiv, Semantic Scholar, and OpenAlex.\n"
        "- Extract literature model primitives and rheology mechanisms from auditable source_id evidence.\n"
        "- Generate 3-5 candidate model families and 3-5 candidate mechanisms separately.\n"
        "- If real industrial data are unavailable or too simple, use the reproducible physics-guided synthetic dataset, not LLM-invented CSV rows.\n"
        "- Apply a novelty filter: reject legacy Skill1-5 derivatives and fixed local-template copies.\n"
        "- Select one model-mechanism combination only if the current columns support it.\n"
        f"- Current normalized target is {target}; available feature columns are {features}.\n"
        "- Evaluation contract: always run 5-fold OOF on the active training dataset; select exactly one secondary evaluation, preferably Table 6 anchor validation when anchor data exist.\n"
        "- OperationAgent must generate runnable code, avoid leakage, report metrics, save predictions/models, and audit any mechanism used."
    )


def _brief_model_plan(model_plan: dict[str, Any]) -> str:
    content = str(model_plan.get("content") or "").strip()
    if not content:
        return "ModelAgent produced an empty plan."
    text = content
    if text.startswith("```"):
        text = text.strip("`")
    try:
        payload = json.loads(text)
    except Exception:
        return "ModelAgent plan summary:\n" + text[:2500]
    lines = [
        f"Selected approach: {payload.get('selected_approach', 'not specified')}",
        f"Features: {payload.get('features', 'not specified')}",
        f"Target: {payload.get('target', 'yield_stress')}",
        f"Evaluation: {payload.get('evaluation', 'not specified')}",
    ]
    rationale = str(payload.get("rationale") or "").strip()
    if rationale:
        lines.append("Rationale: " + rationale[:1200])
    mechanisms = payload.get("mechanisms_to_use")
    if mechanisms:
        lines.append("Mechanisms considered/recommended: " + json.dumps(mechanisms, ensure_ascii=False)[:1500])
    return "\n".join(lines)


def _brief_candidate_report(candidate_report: dict[str, Any]) -> str:
    primitives = candidate_report.get("model_primitives", [])
    models = candidate_report.get("candidate_models", [])
    mechanisms = candidate_report.get("candidate_mechanisms", [])
    fusions = candidate_report.get("candidate_fusion_specs", [])
    executors = candidate_report.get("candidate_executor_specs", [])
    hybrids = candidate_report.get("candidate_hybrid_strategies", [])
    selected = candidate_report.get("selected_combination", {})
    audit = candidate_report.get("data_capability_audit", {})
    lines = [
        f"CandidateAgent extracted {len(primitives)} model primitives, then generated {len(models)} model candidates, {len(mechanisms)} mechanism candidates, {len(fusions)} fusion specs, {len(executors)} executor specs, and {len(hybrids)} model-mechanism hybrid strategies.",
        "Model primitives:",
    ]
    if isinstance(audit, dict) and audit.get("mechanism_support"):
        lines.append("Data capability audit:")
        for key, value in audit.get("mechanism_support", {}).items():
            if isinstance(value, dict):
                lines.append(f"- {key}: {value.get('status')} | {value.get('reason')}")
    if candidate_report.get("selection_audit_note"):
        lines.append("Selection audit: " + str(candidate_report.get("selection_audit_note")))
    model_quality = candidate_report.get("candidate_model_quality_audit")
    if isinstance(model_quality, dict):
        lines.append(
            "Model candidate quality audit: "
            f"status={model_quality.get('status')}, "
            f"codegen_ready={model_quality.get('n_codegen_ready_after_repair')}, "
            f"added={model_quality.get('added_model_ids', [])}"
        )
    selection_audit = candidate_report.get("candidate_selection_audit")
    if isinstance(selection_audit, dict):
        summary = selection_audit.get("benchmark_report_summary", {})
        lines.append(
            "Benchmark audit: "
            f"before={selection_audit.get('selected_before_benchmark')}, "
            f"after={selection_audit.get('selected_after_benchmark')}, "
            f"changed={selection_audit.get('selection_changed_by_benchmark')}, "
            f"selected_oof_rmse={summary.get('selected_strategy_oof_rmse')}, "
            f"best_fixed_baseline_oof_rmse={summary.get('best_fixed_baseline_oof_rmse')}"
        )
    for item in primitives[:6]:
        lines.append(
            f"- [{item.get('id')}] {item.get('primitive')} | sources={item.get('source_ids_or_links', [])}"
        )
    lines.append("Candidate models:")
    for item in models[:5]:
        novelty = item.get("novelty_filter", {})
        novelty_text = ""
        if isinstance(novelty, dict):
            novelty_text = f", novelty={'pass' if novelty.get('accepted', True) is not False else 'reject'}"
        lines.append(
            f"- [{item.get('id')}] {item.get('name')} ({item.get('family')}), feasibility={item.get('feasibility_score_1_to_5')}/5{novelty_text}, sources={item.get('source_ids_or_links', [])}: {item.get('modeling_idea')}"
        )
    lines.append("Candidate mechanisms:")
    for item in mechanisms[:5]:
        lines.append(
            f"- [{item.get('id')}] {item.get('name')}, role={item.get('role_options')}, feasibility={item.get('feasibility_score_1_to_5')}/5, sources={item.get('source_ids_or_links', [])}: {item.get('formula_or_relationship')}"
        )
    lines.append("Candidate executor specs:")
    for item in executors[:5]:
        lines.append(
            f"- [{item.get('id')}] fusion={item.get('fusion_id')}, executor_type={item.get('executor_type')}, "
            f"harness_status={item.get('harness_status')}, graph={item.get('graph_status')}/{item.get('graph_mode')}: {item.get('description')}"
        )
        reason = str("; ".join(item.get("harness_rejection_reasons", []) or [])).strip()
        if reason:
            lines.append(f"  reason: {reason[:500]}")
    lines.append("Model-mechanism hybrid strategies:")
    for item in hybrids[:5]:
        score = item.get("total_score_1_to_10")
        if score is None and isinstance(item.get("score_breakdown"), dict):
            score = item["score_breakdown"].get("total_score_1_to_10")
        lines.append(
            f"- [{item.get('id')}] {item.get('name')}, model={item.get('model_id')}, mechanisms={item.get('mechanism_ids')}, data_support={item.get('data_support')}, score={score}/10, selected={item.get('selected')}: {item.get('combination_method')}"
        )
        reason = str(item.get("data_support_reason") or item.get("rejection_reason") or "").strip()
        if reason:
            lines.append(f"  reason: {reason[:500]}")
    if selected:
        lines.append(
            f"Selected combination: strategy={selected.get('strategy_id')}, model={selected.get('model_id')}, mechanisms={selected.get('mechanism_ids')}, uses_mechanism={selected.get('uses_mechanism')}, evaluation={selected.get('evaluation_protocol')}"
        )
        rationale = str(selected.get("selection_rationale") or "")
        if rationale:
            lines.append("Selection rationale: " + rationale[:1000])
    return "\n".join(lines)


def build_code_instructions(
    user_prompt: str,
    data_path: str,
    test_path: str,
    run_dir: str,
    search_report: dict[str, Any],
    model_plan: dict[str, Any],
    candidate_report: dict[str, Any] | None = None,
    synthetic_info: dict[str, Any] | None = None,
) -> str:
    train_profile = _profile_dataset(data_path)
    test_profile = _profile_dataset(test_path) if test_path and Path(test_path).exists() else None
    anchor_path = str((synthetic_info or {}).get("anchor_path") or "")
    synthetic_report_path = str((synthetic_info or {}).get("synthetic_report_path") or "")
    anchor_profile = _profile_dataset(anchor_path) if anchor_path and Path(anchor_path).exists() else None
    synthetic_report = {}
    if synthetic_report_path and Path(synthetic_report_path).exists():
        try:
            synthetic_report = json.loads(Path(synthetic_report_path).read_text(encoding="utf-8"))
        except Exception:
            synthetic_report = {}
    external_snippets = search_report.get("snippets") or []
    has_external_snippets = bool(external_snippets)
    references = json.dumps(REFERENCE_MECHANISM_NOTES, ensure_ascii=False, indent=2)
    search_report_json = json.dumps(search_report, ensure_ascii=False, indent=2)
    model_plan_json = json.dumps(model_plan, ensure_ascii=False, indent=2)
    candidate_report_json = json.dumps(candidate_report or {}, ensure_ascii=False, indent=2)
    train_profile_json = json.dumps(train_profile, ensure_ascii=False, indent=2)
    test_profile_json = json.dumps(test_profile, ensure_ascii=False, indent=2)
    anchor_profile_json = json.dumps(anchor_profile, ensure_ascii=False, indent=2)
    synthetic_info_json = json.dumps(synthetic_info or {}, ensure_ascii=False, indent=2)
    synthetic_report_json = json.dumps(synthetic_report, ensure_ascii=False, indent=2)
    if has_external_snippets:
        local_reference_section = (
            "External search returned snippets, so local reference mechanisms are NOT "
            "injected as model candidates for this run. Treat this section as omitted; "
            "any mechanistic model used must be justified from SearchAgent snippets or "
            "from the user's prompt, and must cite the searched title/link/source in "
            "mechanism_report.json."
        )
    else:
        local_reference_section = (
            "External search returned no snippets or local fallback was allowed. These "
            "references are background candidates, not mandatory model choices. If you "
            "use any of them or a variant, document the exact formula and paper/source "
            "in mechanism_report.json.\n"
            f"{references}"
        )

    return f"""
Build an LLM-generated AutoML training script for yield-stress regression.

The final code should be original to this run. Do not copy a fixed training
script from another project. Use the dataset profile and SearchAgent report,
then choose a modeling approach that is runnable in the current Python
environment.

{_core_yield_context() if has_external_snippets else YIELD_DOMAIN_CONTEXT}

# User request
{user_prompt}

# Training data profile
{train_profile_json}

# Fixed temporary test data profile
{test_profile_json}

# Synthetic data and Table 6 anchor configuration
{synthetic_info_json}

# Real Table 6 anchor profile
{anchor_profile_json}

# Synthetic data report
This report was generated by fixed project code, not by free-form LLM CSV
invention. If you train on the synthetic dataset, copy the key assumptions into
logs/synthetic_data_report.json or include them in metrics/metrics.json.
{synthetic_report_json}

# Reference mechanisms from local design materials
{local_reference_section}

# SearchAgent report
This stage runs before OperationAgent. It may include external snippets or a
fallback note when search is not configured. If you use a mechanistic claim from
a snippet, record the title/link/source in mechanism_report.json.
{search_report_json}

# CandidateAgent report
Models and mechanisms are separated here, then recombined into
candidate_hybrid_strategies. OperationAgent should first inspect
selected_combination.strategy_id and implement that hybrid strategy if
data_support is supported or partial with a safe implementation plan. If the
selected strategy is unsupported, OperationAgent must reject it explicitly in
mechanism_report.json and implement the fallback model.
This report has been post-processed by the deterministic data-capability audit.
Treat deterministic_data_score_1_to_10 and selected_by_data_capability_audit as
higher priority than raw LLM preference. If yodel_packing_audit says
must_report_if_not_selected=true and the selected strategy is not YODEL/packing,
mechanism_report.json must include rejected_yodel_packing_candidate with a
concrete data, performance, or numerical-stability reason.
When yodel_packing_audit.selected=false after benchmark, this must be a
structured object with candidate_id, reason, and benchmark_evidence; a free-text
note is not enough.
candidate_selection_audit and per-strategy benchmark_audit are run-local proxy
measurements, not fixed templates. Treat them as strong selection evidence. If
OperationAgent does not implement the benchmark-selected selected_combination,
it must record the rejected strategy and a concrete performance, data-support,
package-availability, or numerical-stability reason in mechanism_report.json.
If candidate_selection_audit.status is no_viable_anchor_candidate or
selection_blocked_by_anchor_viability=true, this is a CandidateAgent/data-layer
failure, not a normal OperationAgent fallback. Do not present any implemented
failed-anchor strategy as research-successful; if execution proceeds after
revision exhaustion, mark research_confidence="low" and explain that no viable
anchor candidate was found.
If OperationAgent implements the benchmark-selected selected_combination, it
must record selected_hybrid_strategy and benchmark_alignment in metrics or
mechanism_report. benchmark_alignment must name the selected_strategy_id,
benchmark_proxy_model, benchmark_feature_mode, implemented_model,
implemented_feature_mode, alignment_status, and deviation_reason when there is
any fallback or partial implementation.
It must also record implementation_validation in metrics or mechanism_report.
implementation_validation must compare the final implementation against the
benchmark-selected strategy using benchmark_selected_strategy,
benchmark_oof_rmse, actual_oof_rmse, benchmark_anchor_r2 when available,
actual_anchor_r2 when available, generalization_gap, and validation_status. If
anchor R2 is negative or the final implementation is materially worse than the
benchmark proxy, validation_status must be degraded, failed_generalization, or
completed_with_warning, not validated.
OperationAgent must also run the benchmark-selected proxy as an explicit
candidate. If candidate_selection_audit selected proxy_model="mlp" and
feature_mode="sp_decay", the generated script must actually evaluate an MLP
candidate with that feature mode under the same 5-fold OOF and anchor protocol.
Save benchmark_selected_proxy_result in metrics/metrics.json or
logs/mechanism_report.json with strategy_id, proxy_model, feature_mode,
oof_metrics, and anchor_validation when anchor exists. You may also evaluate
HGBR/GPR/Ridge/etc., but do not replace the selected proxy with a different
family without first reporting the proxy's real metrics.
{candidate_report_json}

# ModelAgent plan
Use this as guidance, not fixed code. OperationAgent may adapt it if needed for
runnable implementation, but must preserve the data/evaluation/artifact contract.
{model_plan_json}

# Run configuration
- data_path: {data_path}
- test_path: {test_path}
- run_dir: {run_dir}
- Primary target after schema normalization: yield_stress
- Synthetic anchor path, when available: {anchor_path}
- Synthetic data report path, when available: {synthetic_report_path}

# Fixed project schema utility
- Import and use:
  from knowledge.yield_schema import load_yield_dataframe
- Do not redefine a local function named load_yield_dataframe. The imported
  project utility owns schema normalization and leakage/reference-column
  exclusion, including phi_max_eff_reference and other *_reference columns.
- Use feature columns from the returned metadata when possible, and still drop
  sample_id/id/source/data_fidelity plus any target/leakage/reference column
  before constructing X.
- Build a reusable feature/preprocessing transform with fit/transform semantics.
  During OOF, fit imputation, encoding, scaling, and fold-local mechanism
  feature parameters only on the training fold, then transform the validation
  fold with the same fitted schema. For final anchor validation, fit on the full
  active training data, then transform the anchor data with the same schema.
  If using pandas get_dummies, save the training dummy columns and call
  reindex(columns=train_columns, fill_value=0) for validation/anchor.
- Save preprocessing/preprocessing_audit.json with raw_feature_columns,
  final_feature_columns used by the selected final model, optional
  engineered_feature_columns considered, train_feature_count, anchor_feature_count
  when anchor validation exists,
  anchor_aligned_to_train_schema, fold_local_preprocessing,
  final_model_fit_scope, and leakage_columns_excluded. train_feature_count and
  anchor_feature_count must match len(final_feature_columns).

# Fixed project baseline utility
- Import and use:
  from knowledge.yield_baselines import run_fixed_yield_baselines
- Do not redefine a local function named run_fixed_yield_baselines.
- The baseline utility is fixed project code. Use it to evaluate stable
  comparison baselines, then append any generated/custom model result to the
  same metrics table. Do not ask the LLM to reinvent the baseline
  implementations each run.
- run_fixed_yield_baselines returns a dict-like bundle, not a plain list. Read
  fixed rows from baseline_bundle["baseline_results"]. Each row uses keys such
  as name, mean_rmse, oof_rmse, mean_r2, and oof_r2. Do not iterate over the
  top-level dict as if it were a list of model rows.
- Do not write NaN or Infinity into metrics/metrics.json or mechanism_report.
  If a baseline metric is not applicable, use None/null or omit that optional
  field. Required R2/RMSE/MAE/MAPE metrics must be finite when mathematically
  possible.
- Fixed baselines are reference controls, not eligible final research models.
  The final `selected_model` must be a generated/custom/research candidate.
  Still compare it against the best fixed baseline and record:
  best_baseline, best_baseline_metric, best_research_model,
  best_research_metric, selection_metric or selection_metric_name,
  selection_metric_direction, beats_best_baseline, and research_model_status.
  best_baseline_metric and best_research_metric must use the same metric named
  by selection_metric. For example, if selection_metric is oof_rmse, both best
  metric values must be RMSE values, not R2 values.
  If the generated research model does not beat the best fixed baseline under
  the declared selection metric, keep the generated model as selected_model but
  set beats_best_baseline=false and research_model_status to
  "underperforms_baseline" or "completed_with_warning". Do not present this as a
  scientifically strong result.

Freedom:
- You may choose data-driven, physics-informed, hybrid, ensemble,
  multi-fidelity, symbolic/structured, or other suitable methods.
- You may choose architecture, loss, target transform, optimizer, scheduler,
  and hyperparameters.
- When feasible, compare multiple candidate model families rather than fixing a
  single default model. Examples include tree ensembles, kernel methods,
  Gaussian Process, regularized linear/symbolic features, neural networks, and
  physics-informed or hybrid variants if supported by available columns.
- Always include a baseline comparison table when training data are available.
  Use knowledge.yield_baselines.run_fixed_yield_baselines for this table. It
  evaluates the fixed project baselines:
  1. DummyRegressor(mean) as the no-skill baseline,
  2. Ridge or LinearRegression as the simple linear baseline,
  3. an unconstrained tree ensemble such as RandomForestRegressor or
     HistGradientBoostingRegressor,
  4. a monotonic HistGradientBoostingRegressor when sklearn supports
     monotonic_cst and the column semantics justify a monotonic direction.
     For temporary Lian cement-paste data where sp_percent is documented as a
     superplasticizer/dispersant/plasticizer dosage, the normalized feature order
     [phi, sp_percent] may use monotonic_cst=[0, -1] to encode no forced
     monotonicity on phi and a non-increasing relation with dispersant dosage.
     Do not apply this direction when the same column or a future column
     represents curing agent/hardener dosage; curing can cause dilution,
     reaction, crosslinking, and time-dependent strengthening, so the sign is
     stage-dependent unless time/stage variables support a specific constraint.
  If sklearn does not support monotonic_cst, record this as a rejected
  mechanism-aware baseline and continue.
- Generated custom/research models should be appended to the same comparison
  table as non-fixed candidates. Select one final model using the same
  selection metric, but keep all fixed baseline rows in metrics/metrics.json.
- Include the benchmark-selected proxy candidate in that comparison table. Mark
  it with benchmark_selected_proxy=true, strategy_id, proxy_model, and
  feature_mode so the verifier can audit whether the CandidateBenchmark choice
  was actually executed.
- The fixed baseline table is for judging whether the generated research model
  adds value. Never select a fixed baseline as the final research model. If all
  generated candidates are weaker than the best baseline, report that explicitly
  and either revise the generated model or finish with
  research_model_status="underperforms_baseline".
- If physics-inspired features are used, include an additional physics-feature
  baseline using the same evaluation protocol so the report can separate model
  family gains from mechanism-feature gains.
- If the selected hybrid is YODEL/packing/fmax-style, implement it as a
  leakage-safe physical feature layer or baseline plus ML residual correction.
  A minimal acceptable implementation is a fold-local packing index such as
  max(phi-phi_c,0)^p / max(phi_m_eff-phi, eps), where phi_m_eff/fmax is fitted
  inside each training fold or fixed from a documented literature constant.
  Never use phi_max_eff_reference or any *_reference column as a sample-level
  feature. Compare the same model family with and without the packing feature
  layer under the same OOF protocol.
- If a mechanism/physics feature is used, run an ablation under the same 5-fold
  OOF protocol: same model family without the mechanism feature versus with the
  mechanism feature. Save mechanism_ablation in metrics/metrics.json with
  without_mechanism, with_mechanism, delta_rmse, delta_r2, and
  mechanism_improves_metric. If mechanism_improves_metric is false, mechanism
  claims must say the mechanism was attempted/audited but did not improve the
  selected metric.
- If mechanism_used=true, save mechanism_effect_size with
  relative_rmse_improvement, relative_rmse_improvement_percent, and
  claim_strength. relative_rmse_improvement must be a fraction such as 0.013
  for 1.3%; relative_rmse_improvement_percent must be 1.3. Use
  claim_strength="marginal" when relative RMSE improvement is below 1%; do not
  present marginal gains as strong mechanism validation.
- Evaluation protocol is constrained for this yield project:
  1. Always run 5-fold OOF on the active training dataset.
  2. Select exactly one secondary evaluation.
  3. If a real Table 6 anchor dataset exists, choose anchor validation as the
     secondary evaluation. Train the final model on the active training data,
     then evaluate it on the anchor dataset without mixing anchor rows into OOF.
  4. If no real anchor exists and sample_size >= 300, repeated K-fold is allowed
     as the one secondary evaluation.
  5. If no real anchor exists but clear process/regime variables exist, a
     physics-aware split may be selected as the one secondary evaluation.
  6. Do not run more than one secondary evaluation unless explicitly requested.
- When using synthetic data, do not claim final industrial performance. Label
  it as low-fidelity workflow/model-structure validation and report anchor
  validation metrics separately.
- You may train a final model on the full normalized training CSV after the
  unbiased evaluation artifacts are saved.
- You may decide not to use a mechanistic model if the data does not support it,
  but you must explicitly say so in mechanism_report.json.

Training progress visibility:
- Print readable training progress to stdout. At minimum print:
  1. dataset shape and selected feature columns,
  2. candidate model names before evaluation,
  3. fold/repeat progress with RMSE, MAE, and R2 for each evaluated candidate,
  4. selected model and selection metric,
  5. final full-data training start/end,
  6. fixed temporary test evaluation if used,
  7. artifact save paths.
- Keep logs concise but informative; these stdout lines are shown in the live
  dashboard as the training process.

Scientific reporting requirement:
- If any mechanism is used, list every mechanism with paper/source and exact
  formula/relationship. Include why the current columns support that mechanism.
  Also include source_ids and source_evidence entries copied from searched
  snippets: source_id, title, link_or_doi, provider, and used_claim.
- If a candidate_hybrid_strategy is selected, implement its combination_method
  in code, not just in the report. If it cannot be safely implemented, record
  the rejected strategy id, model_id, mechanism_ids, and reason in
  mechanism_report.json, then use the fallback.
- When SearchAgent snippets are available, mechanistic claims should cite the
  searched title/link/source rather than relying on local reference notes.
- Record planned_model, actual_model, selected_hybrid_strategy, and
  fallback_reason/package_availability when the implemented model differs from
  CandidateAgent's selected model (for example, XGBoost unavailable because
  libomp is missing and sklearn RandomForest/HistGradientBoosting is used).
- planned_model and actual_model are mandatory scientific audit fields. If
  CandidateAgent/ModelAgent selected XGBoost, PINN, Gaussian Process, or another
  specific model family but the code implements a different family, record a
  non-empty fallback_reason. If there is no fallback reason, revise the code to
  implement the planned family or mark the planned strategy rejected.
- Optional dependencies such as xgboost, lightgbm, catboost, torch, or pysr
  must not be imported at top level. Import them only inside a try/except that
  catches Exception/OSError/ImportError and falls back to sklearn. For this
  macOS environment, prefer sklearn HistGradientBoostingRegressor,
  RandomForestRegressor, GaussianProcessRegressor, MLPRegressor, or PyTorch
  only if already importable. Never crash only because lightgbm/xgboost/catboost
  cannot load native libraries such as libomp.
- Any fitted physical or transform parameter (for example phi_ref, phi_m,
  monotonic thresholds, or scaler parameters) must be fit inside each training
  fold/split for evaluation, then refit on full training data only for the final
  deployed model. If a parameter is a fixed literature constant, document the
  source_id and value.
- Do not print "Yield pipeline completed successfully" until metrics,
  predictions, model artifact, preprocessing artifact,
  preprocessing/preprocessing_audit.json, and mechanism_report are saved. The
  outer framework still performs deterministic guardrail verification after this
  line.
- Before every json.dump/json.dumps call, recursively convert numpy and pandas
  scalar/array objects into plain Python JSON types. Include a helper equivalent
  to:
  def to_jsonable(obj):
      if isinstance(obj, dict): return dict((str(k), to_jsonable(v)) for k, v in obj.items())
      if isinstance(obj, (list, tuple)): return [to_jsonable(v) for v in obj]
      if isinstance(obj, (np.integer,)): return int(obj)
      if isinstance(obj, (np.floating,)): return float(obj) if np.isfinite(float(obj)) else None
      if isinstance(obj, (np.bool_,)): return bool(obj)
      if isinstance(obj, np.ndarray): return obj.tolist()
      if isinstance(obj, float): return obj if np.isfinite(obj) else None
      if hasattr(obj, "item"):
          value = obj.item()
          return value if not isinstance(value, float) or np.isfinite(value) else None
      return obj
  Then call json.dump(to_jsonable(metrics), f, indent=2). A generated run that
  crashes with "Object of type bool_/float32/int64 is not JSON serializable" is
  not acceptable.
- If you use only a data-driven model, explain that no mechanistic equation was
  used and record searched/reference mechanisms that were considered but not
  implemented.
- In metrics/metrics.json, include the evaluation protocol name and a brief
  rationale for why it is appropriate for this dataset.
- In metrics/metrics.json, include primary_evaluation="5-fold OOF" and
  secondary_evaluation. If anchor_path exists, secondary_evaluation must be
  "anchor_validation" and metrics must include anchor_validation R2/RMSE/MAE/MAPE
  when mathematically possible.
- All MAPE values in metrics/metrics.json and mechanism_report.json must be
  percentages, not fractions. sklearn.metrics.mean_absolute_percentage_error
  returns a fraction; multiply it by 100 before saving.
- Copy or save the synthetic data audit to logs/synthetic_data_report.json when
  synthetic training data are used.
- In metrics/metrics.json or mechanism_report.json, include selected_model and
  candidate_models with enough detail to explain why the final model was chosen.
- In metrics/metrics.json, include baseline_results or candidate_models with
  every evaluated baseline/model, selected_model, selection_metric, and whether
  each entry used mechanism constraints or physics-inspired features.
- In metrics/metrics.json, include research_confidence. When using synthetic
  data with anchor validation, set research_confidence based primarily on anchor
  validation, not synthetic OOF:
  low if anchor R2 < 0 or the research model underperforms the best baseline;
  medium if anchor R2 >= 0.3 and the model does not underperform baseline;
  high only if anchor R2 >= 0.6, the model beats baseline, and mechanism claims
  are supported by ablation or explicitly marked as no-gain.
- If anchor R2 is negative, do not mark the run as validated or scientifically
  successful. Set research_confidence="low", record generalization_gap, and use
  research_model_status such as completed_with_warning, underperforms_baseline,
  or needs_research_revision.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_data_path = os.getenv("YIELD_DATA_PATH", DEFAULT_LIAN_DATA_PATH)
    default_test_path = os.getenv("YIELD_TEST_PATH", DEFAULT_LIAN_TEST_PATH)
    synthetic_env = os.getenv("YIELD_SYNTHETIC_DATA")
    if synthetic_env is None:
        default_synthetic = not Path(default_data_path).exists()
    else:
        default_synthetic = synthetic_env.lower() in {"1", "true", "yes", "on"}
    parser.add_argument("--data-path", default=default_data_path)
    parser.add_argument("--test-path", default=default_test_path)
    parser.add_argument("--run-dir", default=os.getenv("YIELD_RUN_DIR", f"agent_workspace/runs/yield_search_{_now_id()}"))
    parser.add_argument("--llm", default=os.getenv("YIELD_LLM", DEFAULT_LLM))
    parser.add_argument("--n-revise", type=int, default=int(os.getenv("YIELD_N_REVISE", str(LOW_TOKEN_N_REVISE if LOW_TOKEN_MODE else 2))))
    parser.add_argument("--operation-attempts", type=int, default=int(os.getenv("YIELD_OPERATION_ATTEMPTS", "5")))
    parser.add_argument(
        "--random-state",
        type=int,
        default=int(os.getenv("YIELD_RANDOM_STATE", "42")),
        help="Base random seed for candidate benchmark, free-search CV splits, and generated estimators.",
    )
    parser.add_argument("--prompt", default=os.getenv("YIELD_PROMPT", "Build an AutoML model for high-solid-content slurry yield-stress prediction. Search for suitable rheology-informed or data-driven methods, then generate runnable training code."))
    parser.add_argument("--external-search", action=argparse.BooleanOptionalAction, default=os.getenv("YIELD_EXTERNAL_SEARCH", "1").lower() in {"1", "true", "yes", "on"})
    parser.add_argument("--require-search-results", action=argparse.BooleanOptionalAction, default=os.getenv("YIELD_REQUIRE_SEARCH_RESULTS", "1").lower() in {"1", "true", "yes", "on"})
    parser.add_argument("--synthetic-data", action=argparse.BooleanOptionalAction, default=default_synthetic)
    parser.add_argument("--synthetic-n", type=int, default=int(os.getenv("YIELD_SYNTHETIC_N", "800")))
    parser.add_argument("--query", action="append", default=None, help="Additional external-search query. Can be passed multiple times.")
    return parser.parse_args()


class YieldAgentManager:
    """Yield-specific manager that preserves the multi-agent control flow."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.run_dir = Path(args.run_dir)
        self.state = "INIT"
        self.stage_records: list[dict[str, Any]] = []
        self.synthetic_info: dict[str, Any] = {"enabled": bool(args.synthetic_data)}
        self.train_profile: dict[str, Any] = {}
        self.search_report: dict[str, Any] = {}
        self.candidate_report: dict[str, Any] = {}
        self.candidate_benchmark_report: dict[str, Any] = {}
        self.candidate_selection_audit: dict[str, Any] = {}
        self.model_plan: dict[str, Any] = {}
        self.operation_instructions = ""
        self.manager_revision_round = 0
        self.revision_notes: list[dict[str, Any]] = []

    def _current_random_state(self) -> int:
        return int(self.args.random_state) + int(self.manager_revision_round)

    def _json_default(self, value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        return str(value)

    def _write_json(self, filename: str, payload: dict[str, Any]) -> Path:
        path = self.run_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=self._json_default),
            encoding="utf-8",
        )
        return path

    def _transition(self, state: str, summary: str = "", details: dict[str, Any] | None = None) -> None:
        self.state = state
        record = {
            "state": state,
            "summary": summary,
            "details": details or {},
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        self.stage_records.append(record)
        if self.run_dir.exists():
            self._write_json(
                "manager_trace.json",
                {
                    "current_state": self.state,
                    "stages": self.stage_records,
                },
            )
        if summary:
            _emit_event("manager", "Agent Manager:", f"[{state}] {summary}", mirror=False)

    def _prepare_run(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        os.environ["YIELD_RUN_DIR"] = str(self.run_dir.resolve())
        self._transition(
            "INIT",
            "Initialized yield-stress AgentManager run context.",
            {
                "run_dir": str(self.run_dir.resolve()),
                "llm": self.args.llm,
                "n_revise": int(self.args.n_revise),
                "operation_attempts": int(self.args.operation_attempts),
                "random_state": int(self.args.random_state),
                "effective_random_state": self._current_random_state(),
                "external_search": bool(self.args.external_search),
                "require_search_results": bool(self.args.require_search_results),
                "synthetic_data": bool(self.args.synthetic_data),
            },
        )
        self._write_json(
            "manager_run_config.json",
            {
                "prompt": self.args.prompt,
                "data_path": self.args.data_path,
                "test_path": self.args.test_path,
                "run_dir": str(self.run_dir.resolve()),
                "llm": self.args.llm,
                "n_revise": int(self.args.n_revise),
                "operation_attempts": int(self.args.operation_attempts),
                "random_state": int(self.args.random_state),
                "effective_random_state": self._current_random_state(),
                "external_search": bool(self.args.external_search),
                "require_search_results": bool(self.args.require_search_results),
                "synthetic_data": bool(self.args.synthetic_data),
                "synthetic_n": int(self.args.synthetic_n),
                "extra_queries": self.args.query or [],
            },
        )

    def _run_data_agent(self) -> None:
        self._transition("DATA", "DataAgent preparing active training and anchor data.")
        data_path = str(Path(self.args.data_path).resolve())
        test_path = str(Path(self.args.test_path).resolve())
        anchor_compatibility_note = ""
        if self.args.synthetic_data:
            print("YIELD_STAGE: synthetic_data")
            generated = ensure_default_synthetic_yield_data(
                output_dir=self.run_dir / "data",
                n_samples=max(100, int(self.args.synthetic_n)),
                overwrite=True,
            )
            self.synthetic_info.update(generated)
            self.synthetic_info["n_samples"] = max(100, int(self.args.synthetic_n))
            self.synthetic_info["reason"] = (
                "Real industrial data are unavailable and the temporary data are too small/simple "
                "for complex model-mechanism research."
            )
            self.synthetic_info["active_training_data"] = generated["synthetic_path"]
            self.synthetic_info["secondary_anchor_data"] = generated["anchor_path"]
            data_path = generated["synthetic_path"]
            test_path = generated["anchor_path"]
            os.environ["YIELD_ANCHOR_PATH"] = generated["anchor_path"]
            os.environ["YIELD_SYNTHETIC_REPORT_PATH"] = generated["synthetic_report_path"]
            _emit_event(
                "data",
                "Data Agent:",
                (
                    "Synthetic data mode enabled.\n"
                    f"Active training CSV: {generated['synthetic_path']}\n"
                    f"Real Table 6 anchor CSV: {generated['anchor_path']}\n"
                    f"Synthetic audit report: {generated['synthetic_report_path']}"
                ),
                mirror=False,
            )

        if not self.args.synthetic_data and test_path and Path(test_path).exists():
            try:
                _train_df, train_meta = load_yield_dataframe(data_path)
                _test_df, test_meta = load_yield_dataframe(test_path)
                default_lian_anchor = Path(test_path).resolve() == Path(DEFAULT_LIAN_TEST_PATH).resolve()
                incompatible_generated_anchor = (
                    str(train_meta.get("source_schema") or "") == "generated_yield_process_202607"
                    and str(test_meta.get("source_schema") or "").startswith("lian2025")
                )
                allow_incompatible = str(os.environ.get("YIELD_ALLOW_INCOMPATIBLE_ANCHOR") or "").lower() in {
                    "1", "true", "yes", "on"
                }
                if default_lian_anchor and incompatible_generated_anchor and not allow_incompatible:
                    anchor_compatibility_note = (
                        "Default Lian Table6 anchor disabled for generated_yield_process_202607 training data; "
                        "schemas/target scales are incompatible. Set YIELD_ALLOW_INCOMPATIBLE_ANCHOR=1 to force it."
                    )
                    test_path = ""
                    os.environ.pop("YIELD_ANCHOR_PATH", None)
                    _emit_event("data", "Data Agent:", anchor_compatibility_note, mirror=False)
            except Exception as exc:
                anchor_compatibility_note = (
                    f"Anchor compatibility audit skipped: {type(exc).__name__}: {exc}"
                )

        os.environ["YIELD_DATA_PATH"] = data_path
        os.environ["YIELD_TEST_PATH"] = test_path
        self.train_profile = _profile_dataset(os.environ["YIELD_DATA_PATH"])
        _emit_event("data", "Data Agent:", _brief_data_profile(self.train_profile), mirror=False)
        self._write_json(
            "data_agent_report.json",
            {
                "active_training_data": os.environ["YIELD_DATA_PATH"],
                "test_or_anchor_data": os.environ["YIELD_TEST_PATH"],
                "anchor_compatibility_note": anchor_compatibility_note,
                "synthetic_info": self.synthetic_info,
                "profile": self.train_profile,
                "capability_audit": _infer_data_capability_audit(self.train_profile),
            },
        )
        self._transition(
            "DATA_DONE",
            "DataAgent completed data profile and capability audit.",
            {
                "rows": self.train_profile.get("n_rows"),
                "columns": self.train_profile.get("n_columns"),
                "schema": (self.train_profile.get("schema") or {}).get("source_schema"),
            },
        )

    def _emit_requirement_context(self) -> None:
        self._transition("REQUIREMENT", "AgentManager analyzing task requirements.")
        _emit_event("user", "You:", self.args.prompt, mirror=False)
        _emit_event(
            "manager",
            "Agent Manager:",
            (
                "Task parsed: yield-stress AutoML regression.\n"
                f"External search enabled={self.args.external_search}, "
                f"require_search_results={self.args.require_search_results}.\n"
                f"Synthetic data enabled={self.args.synthetic_data}; "
                f"active_training_data={os.environ['YIELD_DATA_PATH']}.\n"
                "Flow: AgentManager -> DataAgent -> SearchAgent -> CandidateAgent -> "
                "ModelAgent -> OperationAgent -> YieldGuardrails."
            ),
            mirror=False,
        )
        _emit_event(
            "manager",
            "Requirement Analysis:",
            _brief_requirement_analysis(self.args.prompt, self.train_profile),
            mirror=False,
        )
        self._write_json(
            "requirement_analysis.json",
            {
                "prompt": self.args.prompt,
                "task": "yield_stress_regression",
                "brief": _brief_requirement_analysis(self.args.prompt, self.train_profile),
                "required_flow": [
                    "DataAgent",
                    "SearchAgent",
                    "CandidateAgent",
                    "ModelAgent",
                    "OperationAgent",
                    "YieldGuardrails",
                ],
            },
        )
        self._transition("REQUIREMENT_DONE", "Requirement analysis recorded.")

    def _run_search_agent(self) -> bool:
        self._transition("SEARCH", "SearchAgent retrieving external yield-stress evidence.")
        print("YIELD_STAGE: search")
        self.search_report = run_search_stage(
            self.args.prompt,
            self.train_profile,
            self.args.external_search,
            self.args.query,
        )
        (self.run_dir / "search_report.json").write_text(
            json.dumps(self.search_report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"YIELD_SEARCH_SNIPPETS: {len(self.search_report.get('snippets', []))}")
        if self.args.external_search and self.args.require_search_results and not self.search_report.get("snippets"):
            result = {
                "rcode": 2,
                "stage": "search",
                "action_result": (
                    "External search was required, but SearchAgent returned zero snippets. "
                    "OperationAgent was not executed. Check search API quota/configuration, "
                    "or run with --no-require-search-results to allow local-reference fallback."
                ),
                "search_report": self.search_report,
                "code": "",
                "error_logs": ["External search returned zero snippets."],
            }
            self._save_result(result)
            print("YIELD_STAGE: search_failed")
            print(result["action_result"])
            self._transition("FAILED", "SearchAgent returned no required external snippets.", result)
            return False
        self._transition(
            "SEARCH_DONE",
            "SearchAgent completed source retrieval.",
            {
                "snippet_count": len(self.search_report.get("snippets", [])),
                "provider_summary": self.search_report.get("provider_summary", {}),
            },
        )
        return True

    def _run_candidate_agent(self, manager_feedback: str = "") -> None:
        self._transition(
            "CANDIDATE",
            "CandidateAgent revising model/mechanism candidates from manager feedback."
            if manager_feedback
            else "CandidateAgent generating and filtering model/mechanism candidates.",
        )
        print("YIELD_STAGE: candidates")
        _emit_event(
            "model",
            "CandidateAgent:",
            (
                "Revising candidates from previous execution failure."
                if manager_feedback
                else "Extracting literature model primitives, generating candidate models/mechanisms, and applying novelty filter."
            ),
            mirror=False,
        )
        self.candidate_report = run_candidate_stage(
            self.args.llm,
            self.args.prompt,
            self.train_profile,
            self.search_report,
            manager_feedback=manager_feedback,
        )
        self.candidate_report["manager_revision_round"] = self.manager_revision_round
        if manager_feedback:
            self.candidate_report["manager_feedback"] = manager_feedback
        self._run_candidate_benchmark_audit(manager_feedback=manager_feedback)
        (self.run_dir / "candidate_report.json").write_text(
            json.dumps(self.candidate_report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if manager_feedback:
            (self.run_dir / f"candidate_report_revision_{self.manager_revision_round}.json").write_text(
                json.dumps(self.candidate_report, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        _emit_event("model", "CandidateAgent:", _brief_candidate_report(self.candidate_report), mirror=False)
        print(
            "YIELD_CANDIDATES_SAVED: "
            f"{len(self.candidate_report.get('candidate_models', []))} models, "
            f"{len(self.candidate_report.get('candidate_mechanisms', []))} mechanisms, "
            f"{len(self.candidate_report.get('candidate_fusion_specs', []))} fusion_specs, "
            f"{len(self.candidate_report.get('candidate_executor_specs', []))} executor_specs"
        )
        self._transition(
            "CANDIDATE_DONE",
            "CandidateAgent completed candidate report.",
            {
                "candidate_models": len(self.candidate_report.get("candidate_models", [])),
                "candidate_mechanisms": len(self.candidate_report.get("candidate_mechanisms", [])),
                "candidate_fusion_specs": len(self.candidate_report.get("candidate_fusion_specs", [])),
                "candidate_executor_specs": len(self.candidate_report.get("candidate_executor_specs", [])),
                "hybrid_strategies": len(self.candidate_report.get("candidate_hybrid_strategies", [])),
                "benchmark_selected_strategy": (
                    self.candidate_selection_audit.get("selected_after_benchmark")
                    if isinstance(self.candidate_selection_audit, dict)
                    else None
                ),
            },
        )

    def _run_candidate_benchmark_audit(self, manager_feedback: str = "") -> None:
        self._transition(
            "CANDIDATE_BENCH",
            "AgentManager running lightweight candidate benchmark and selection audit.",
        )
        print("YIELD_STAGE: candidate_benchmark")
        anchor_path = os.environ.get("YIELD_ANCHOR_PATH") or os.environ.get("YIELD_TEST_PATH") or ""
        try:
            self.candidate_benchmark_report = run_candidate_benchmark(
                os.environ["YIELD_DATA_PATH"],
                self.candidate_report,
                anchor_path=anchor_path,
                random_state=self._current_random_state(),
                max_rows=600,
                n_splits=5,
            )
            self.candidate_report, self.candidate_selection_audit = apply_benchmark_selection(
                self.candidate_report,
                self.candidate_benchmark_report,
            )
        except Exception as exc:
            self.candidate_benchmark_report = {
                "stage": "candidate_benchmark",
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "purpose": "lightweight proxy ranking for CandidateAgent strategies; not the final generated training model",
            }
            self.candidate_selection_audit = {
                "stage": "candidate_selection_audit",
                "status": "failed",
                "selected_before_benchmark": (
                    self.candidate_report.get("selected_combination", {}).get("strategy_id")
                    if isinstance(self.candidate_report.get("selected_combination"), dict)
                    else None
                ),
                "selected_after_benchmark": (
                    self.candidate_report.get("selected_combination", {}).get("strategy_id")
                    if isinstance(self.candidate_report.get("selected_combination"), dict)
                    else None
                ),
                "selection_changed_by_benchmark": False,
                "error": f"{type(exc).__name__}: {exc}",
                "note": "Candidate benchmark failed; ModelAgent should not treat the current selection as benchmark-validated.",
            }
            self.candidate_report["candidate_selection_audit"] = self.candidate_selection_audit

        self._write_json("candidate_benchmark_report.json", self.candidate_benchmark_report)
        self._write_json("candidate_selection_audit.json", self.candidate_selection_audit)
        shift_audit = self.candidate_benchmark_report.get("synthetic_anchor_shift_audit")
        if isinstance(shift_audit, dict):
            shift_path = self.run_dir / "data" / "synthetic_anchor_shift_report.json"
            shift_path.parent.mkdir(parents=True, exist_ok=True)
            shift_path.write_text(
                json.dumps(shift_audit, ensure_ascii=False, indent=2, default=self._json_default),
                encoding="utf-8",
            )
        if manager_feedback:
            self._write_json(
                f"candidate_benchmark_report_revision_{self.manager_revision_round}.json",
                self.candidate_benchmark_report,
            )
            self._write_json(
                f"candidate_selection_audit_revision_{self.manager_revision_round}.json",
                self.candidate_selection_audit,
            )
        selected_after = self.candidate_selection_audit.get("selected_after_benchmark")
        selected_score = (
            self.candidate_selection_audit.get("benchmark_report_summary", {}).get("selected_strategy_score_1_to_10")
            if isinstance(self.candidate_selection_audit.get("benchmark_report_summary"), dict)
            else None
        )
        _emit_event(
            "model",
            "CandidateBenchmark:",
            (
                "Lightweight candidate benchmark completed.\n"
                f"Selected strategy after benchmark: {selected_after}\n"
                f"Selected score: {selected_score}\n"
                f"Changed selection: {self.candidate_selection_audit.get('selection_changed_by_benchmark')}"
            ),
            mirror=False,
        )
        self._transition(
            "CANDIDATE_BENCH_DONE",
            "Candidate benchmark and selection audit recorded.",
            {
                "selected_after_benchmark": selected_after,
                "selection_changed": self.candidate_selection_audit.get("selection_changed_by_benchmark"),
                "benchmark_status": self.candidate_benchmark_report.get("status", "completed"),
            },
        )

    def _run_model_agent(self, manager_feedback: str = "") -> None:
        self._transition(
            "PLAN",
            "ModelAgent revising implementation-ready plan from manager feedback."
            if manager_feedback
            else "ModelAgent creating implementation-ready plan.",
        )
        print("YIELD_STAGE: model_plan")
        _emit_event(
            "model",
            "ModelAgent:",
            (
                f"Replanning from manager revision feedback with {len(self.search_report.get('snippets', []))} external snippets."
                if manager_feedback
                else f"Planning with {len(self.search_report.get('snippets', []))} external snippets and normalized data schema."
            ),
            mirror=False,
        )
        self.model_plan = run_model_plan_stage(
            self.args.llm,
            self.args.prompt,
            self.train_profile,
            self.search_report,
            self.candidate_report,
            manager_feedback=manager_feedback,
        )
        self.model_plan["manager_revision_round"] = self.manager_revision_round
        if manager_feedback:
            self.model_plan["manager_feedback"] = manager_feedback
        (self.run_dir / "model_plan.json").write_text(
            json.dumps(self.model_plan, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if manager_feedback:
            (self.run_dir / f"model_plan_revision_{self.manager_revision_round}.json").write_text(
                json.dumps(self.model_plan, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        _emit_event("model", "ModelAgent:", _brief_model_plan(self.model_plan), mirror=False)
        self._transition(
            "PLAN_DONE",
            "ModelAgent completed model plan.",
            {
                "selected_model_family": self.model_plan.get("selected_model_family")
                or self.model_plan.get("model_family")
                or self.model_plan.get("selected_model"),
                "mechanism_role": self.model_plan.get("mechanism_role"),
            },
        )

    def _build_operation_instructions(self) -> str:
        return build_code_instructions(
            user_prompt=self.args.prompt,
            data_path=os.environ["YIELD_DATA_PATH"],
            test_path=os.environ["YIELD_TEST_PATH"],
            run_dir=os.environ["YIELD_RUN_DIR"],
            search_report=self.search_report,
            model_plan=self.model_plan,
            candidate_report=self.candidate_report,
            synthetic_info=self.synthetic_info,
        )

    def _pre_execution_review(self) -> dict[str, Any]:
        checks = []

        def add_check(name: str, passed: bool, detail: str) -> None:
            checks.append({"name": name, "passed": bool(passed), "detail": detail})

        snippets = self.search_report.get("snippets", [])
        add_check(
            "external_search_evidence",
            (not self.args.require_search_results) or bool(snippets),
            f"snippets={len(snippets)}, require_search_results={self.args.require_search_results}",
        )
        add_check(
            "candidate_models_available",
            bool(self.candidate_report.get("candidate_models")),
            f"candidate_models={len(self.candidate_report.get('candidate_models', []))}",
        )
        add_check(
            "candidate_mechanisms_audited",
            "candidate_mechanisms" in self.candidate_report,
            f"candidate_mechanisms={len(self.candidate_report.get('candidate_mechanisms', []))}",
        )
        selection_audit = self.candidate_report.get("candidate_selection_audit")
        add_check(
            "candidate_selection_benchmarked",
            isinstance(selection_audit, dict)
            and selection_audit.get("status") == "completed"
            and bool(selection_audit.get("benchmark_validated")),
            (
                f"selected_after_benchmark={selection_audit.get('selected_after_benchmark') if isinstance(selection_audit, dict) else None}, "
                f"status={selection_audit.get('status') if isinstance(selection_audit, dict) else 'missing'}"
            ),
        )
        selected_benchmark = (
            selection_audit.get("selected_strategy_benchmark")
            if isinstance(selection_audit, dict)
            else None
        )
        selected_anchor_status = (
            str(selected_benchmark.get("anchor_generalization_status") or "").strip().lower()
            if isinstance(selected_benchmark, dict)
            else ""
        )
        add_check(
            "candidate_selected_anchor_not_failed",
            selected_anchor_status != "failed",
            (
                f"selected_after_benchmark={selection_audit.get('selected_after_benchmark') if isinstance(selection_audit, dict) else None}, "
                f"anchor_generalization_status={selected_anchor_status or 'missing'}"
            ),
        )
        add_check(
            "model_plan_available",
            bool(self.model_plan),
            f"model_plan_keys={sorted(self.model_plan.keys()) if isinstance(self.model_plan, dict) else []}",
        )
        schema = self.train_profile.get("schema", {}) if isinstance(self.train_profile, dict) else {}
        add_check(
            "target_schema_normalized",
            schema.get("target_column") == "yield_stress",
            f"target_column={schema.get('target_column')}",
        )
        add_check(
            "operation_contract_built",
            "yield-stress regression" in self.operation_instructions.lower()
            and "metrics/metrics.json" in self.operation_instructions
            and "predictions" in self.operation_instructions.lower(),
            "OperationAgent instructions include the yield task and required output artifacts",
        )

        review = {
            "passed": all(item["passed"] for item in checks),
            "checks": checks,
            "operation_attempt_budget": max(1, int(self.args.operation_attempts)),
            "manager_decision": "proceed_to_operation" if all(item["passed"] for item in checks) else "blocked_before_operation",
        }
        self._write_json("pre_execution_review.json", review)
        if self.manager_revision_round:
            self._write_json(f"pre_execution_review_revision_{self.manager_revision_round}.json", review)
        _emit_event(
            "manager",
            "Agent Manager:",
            "Pre-execution review: " + review["manager_decision"],
            mirror=False,
        )
        return review

    def _yield_user_requirements(self) -> dict[str, Any]:
        return {
            "task": "yield_stress_regression",
            "problem": {
                "downstream_task": "yield_stress_regression",
                "description": "LLM-generated AutoML for high-solid-content slurry yield-stress prediction.",
            },
            "dataset": [
                {
                    "source": "user-upload",
                    "path": os.environ["YIELD_DATA_PATH"],
                    "test_path": os.environ["YIELD_TEST_PATH"],
                    "anchor_path": os.environ.get("YIELD_ANCHOR_PATH", ""),
                    "synthetic_report_path": os.environ.get("YIELD_SYNTHETIC_REPORT_PATH", ""),
                }
            ],
        }

    def _obtain_yield_plugin_source(self) -> tuple[str, str, list[str]]:
        """Return (plugin_source, source_origin, logs).

        source_origin == 'llm' when the LLM produced a preflight-valid plugin;
        'reference_fallback' when we substitute the deterministic reference plugin.
        """
        logs: list[str] = []
        try:
            op = OperationAgent(
                user_requirements=self._yield_user_requirements(),
                llm=self.args.llm,
                code_path="/yield_plugin",
                task="yield_stress_regression",
            )
            gen = op.generate_yield_plugin(
                self.operation_instructions,
                n_attempts=max(1, int(self.args.operation_attempts)),
            )
            if gen.get("ok") and gen.get("plugin_source"):
                return str(gen["plugin_source"]), "llm", list(gen.get("error_logs") or [])
            logs = list(gen.get("error_logs") or []) + [
                "LLM plugin generation did not yield a valid plugin; using deterministic reference plugin."
            ]
        except Exception as exc:
            logs = [f"LLM plugin generation raised: {type(exc).__name__}: {exc}; using reference plugin."]
        ref_path = Path(__file__).resolve().parent / "operation_agent" / "yield_reference_plugin.py"
        return ref_path.read_text(encoding="utf-8"), "reference_fallback", logs

    def _run_plugin_harness(self) -> dict[str, Any]:
        """Primary yield execution path: LLM-authored constrained plugin executed
        inside the deterministic harness, then lightweight plugin-mode acceptance."""
        self._transition("EXEC", "OperationAgent running the structured plugin harness.")
        print("YIELD_STAGE: operation")
        started = time.time()
        anchor_path = os.environ.get("YIELD_ANCHOR_PATH") or os.environ.get("YIELD_TEST_PATH")
        anchor_expected = bool(anchor_path and Path(anchor_path).exists())
        random_state = (
            (self.candidate_benchmark_report.get("benchmark_protocol") or {}).get("random_state")
            if isinstance(self.candidate_benchmark_report, dict)
            else None
        )

        plugin_source, source_origin, gen_logs = self._obtain_yield_plugin_source()

        def _run(source: str, origin: str) -> dict[str, Any]:
            return execute_plugin_candidate_artifacts(
                os.environ["YIELD_DATA_PATH"],
                self.candidate_report,
                self.candidate_benchmark_report,
                self.run_dir,
                source,
                anchor_path=anchor_path,
                random_state=random_state,
                n_splits=5,
                source_origin=origin,
            )

        try:
            result = _run(plugin_source, source_origin)
        except Exception as exc:
            # e.g. an LLM plugin that passed preflight but broke the harness contract:
            # substitute the deterministic reference plugin so the run stays auditable.
            gen_logs.append(f"Plugin harness error ({type(exc).__name__}: {exc}); using reference plugin.")
            ref_path = Path(__file__).resolve().parent / "operation_agent" / "yield_reference_plugin.py"
            result = _run(ref_path.read_text(encoding="utf-8"), "reference_fallback")

        result.setdefault("error_logs", []).extend(gen_logs)
        verification = verify_yield_plugin_harness_run(
            str(self.run_dir), anchor_expected=anchor_expected, started_at=started
        )
        result["plugin_harness_verification"] = {
            "passed": verification.passed,
            "reasons": verification.reasons,
            "warnings": verification.warnings,
        }
        if not verification.passed:
            result["rcode"] = 1
            result.setdefault("error_logs", []).extend(verification.reasons)
            _emit_event(
                "operation",
                "OperationAgent:",
                "Plugin harness acceptance failed:\n" + "\n".join(f"- {r}" for r in verification.reasons),
                mirror=False,
            )
        else:
            _emit_event(
                "operation",
                "OperationAgent:",
                f"Structured plugin harness accepted (candidate_source={result.get('candidate_source')}).",
                mirror=False,
            )
        return result

    def _mechanism_brief(self, spec: dict[str, Any]) -> str:
        """Compose the searched-mechanism description handed to the LLM to implement."""
        parts = []
        for key in ("id", "name", "paper_or_source", "formula_or_relationship",
                    "required_columns", "applicability_to_current_data"):
            val = spec.get(key)
            if val:
                parts.append(f"{key}: {val}")
        return "\n".join(parts) or json.dumps(spec, ensure_ascii=False)

    def _model_brief(self, spec: dict[str, Any], sample_df: pd.DataFrame | None = None) -> str:
        """Compose the searched-model description handed to the LLM to implement."""
        parts = []
        for key in ("id", "name", "family", "model_kind", "complex_model_contract",
                    "modeling_idea", "required_columns",
                    "strengths", "risks", "feasibility_score_1_to_5",
                    "data_fit_score_1_to_5", "data_fit_reasons"):
            val = spec.get(key)
            if val:
                parts.append(f"{key}: {val}")
        guidance = _model_design_guidance_from_sample(sample_df)
        if guidance:
            parts.append("current_data_model_guidance: " + json.dumps(guidance, ensure_ascii=False))
        return "\n".join(parts) or json.dumps(spec, ensure_ascii=False)

    def _seed_mechanism_for(self, spec: dict[str, Any]):
        """Keyword fallback: map a searched mechanism spec onto a seed Mechanism."""
        from knowledge.yield_mechanisms import HERSCHEL_BULKLEY, LIAN_PACKING, YODEL

        text = _candidate_text(spec).lower()
        if "lian" in text:
            return LIAN_PACKING
        if "herschel" in text or "bulkley" in text or "shear" in text:
            return HERSCHEL_BULKLEY
        return YODEL

    def _obtain_searched_mechanisms(self, mech_specs, sample_df):
        """Turn searched mechanism specs into executable DynamicMechanisms.

        Each spec's formula is handed to the LLM (generate_yield_mechanism); on
        failure we fall back to a keyword-matched seed mechanism so the free
        search still has a mechanism arm. Returns (mechanisms, mech_by_id,
        audit, logs)."""
        from knowledge.yield_mechanisms import YODEL

        mechanisms: list[Any] = []
        mech_by_id: dict[str, Any] = {}
        audit: list[dict[str, Any]] = []
        logs: list[str] = []

        op = None
        try:
            op = OperationAgent(
                user_requirements=self._yield_user_requirements(),
                llm=self.args.llm,
                code_path="/yield_mechanism",
                task="yield_stress_regression",
            )
        except Exception as exc:
            logs.append(f"OperationAgent init failed for mechanism generation: {exc}")

        limit = int(os.environ.get("YIELD_MAX_SEARCHED_MECHANISMS", "4"))
        for spec in [s for s in (mech_specs or []) if isinstance(s, dict)][:limit]:
            got, origin = None, "llm"
            if op is not None:
                try:
                    gen = op.generate_yield_mechanism(
                        self._mechanism_brief(spec), sample_df,
                        n_attempts=max(1, int(self.args.operation_attempts)),
                    )
                    if gen.get("ok") and gen.get("mechanism") is not None:
                        got = gen["mechanism"]
                    else:
                        logs.extend(gen.get("error_logs") or [])
                except Exception as exc:
                    logs.append(f"mechanism generation raised for {spec.get('id')}: {exc}")
            if got is None:
                got, origin = self._seed_mechanism_for(spec), "seed_fallback"
            if got is None or got.id in mech_by_id:
                continue
            mechanisms.append(got)
            mech_by_id[got.id] = got
            entry = {"searched_id": spec.get("id"), "mechanism_id": got.id,
                     "origin": origin, "paper": getattr(got, "paper", "")}
            # persist the accepted mechanism source so the run is reproducible and
            # the LLM-written physics is auditable (not just its id/paper).
            src = getattr(got, "source", None)
            if src:
                try:
                    mech_dir = Path(self.run_dir) / "logs" / "mechanisms"
                    mech_dir.mkdir(parents=True, exist_ok=True)
                    src_path = mech_dir / f"{got.id}.py"
                    src_path.write_text(src, encoding="utf-8")
                    entry["source_path"] = str(src_path)
                except Exception as exc:
                    logs.append(f"could not persist mechanism source for {got.id}: {exc}")
            audit.append(entry)

        if not mechanisms:
            mechanisms.append(YODEL)
            mech_by_id[YODEL.id] = YODEL
            audit.append({"searched_id": None, "mechanism_id": YODEL.id, "origin": "seed_default"})
            logs.append("No searched mechanism available; using YODEL seed as the mechanism arm.")
        return mechanisms, mech_by_id, audit, logs

    def _obtain_searched_models(self, model_specs, sample_x, y_sample, fixed_family_ids):
        """Turn searched model specs into executable DynamicModels.

        LLM-written models are bounded factories (MODEL_SPEC + make_estimator);
        the harness still owns CV/evaluation. Fixed families remain the baseline
        floor. Returns (model_ids, model_by_id, audit, logs).
        """
        model_ids: list[str] = []
        model_by_id: dict[str, Any] = {}
        audit: list[dict[str, Any]] = []
        logs: list[str] = []

        op = None
        try:
            op = OperationAgent(
                user_requirements=self._yield_user_requirements(),
                llm=self.args.llm,
                code_path="/yield_model",
                task="yield_stress_regression",
            )
        except Exception as exc:
            logs.append(f"OperationAgent init failed for model generation: {exc}")

        fixed_ids = {str(x) for x in (fixed_family_ids or [])}
        limit = int(os.environ.get("YIELD_MAX_SEARCHED_MODELS", "3"))
        for spec in [s for s in (model_specs or []) if isinstance(s, dict)][:limit]:
            got, origin = None, "llm"
            if op is not None:
                try:
                    gen = op.generate_yield_model(
                        self._model_brief(spec, sample_x), sample_x, y_sample,
                        n_attempts=max(1, int(self.args.operation_attempts)),
                    )
                    if gen.get("ok") and gen.get("model") is not None:
                        got = gen["model"]
                    else:
                        logs.extend(gen.get("error_logs") or [])
                except Exception as exc:
                    logs.append(f"model generation raised for {spec.get('id')}: {exc}")
            if got is None:
                audit.append({"searched_id": spec.get("id"), "model_id": None,
                              "origin": "generation_failed", "name": spec.get("name")})
                continue

            # Avoid collisions with fixed families such as rf/hgb/pinn.
            base_id = str(got.id)
            if base_id in fixed_ids or base_id in model_by_id:
                suffix = 1
                new_id = f"llm_{base_id}"
                while new_id in fixed_ids or new_id in model_by_id:
                    suffix += 1
                    new_id = f"llm_{base_id}_{suffix}"
                got.id = new_id
                got.spec["id"] = new_id

            model_ids.append(got.id)
            model_by_id[got.id] = got
            entry = {"searched_id": spec.get("id"), "model_id": got.id,
                     "origin": origin, "name": getattr(got, "name", ""),
                     "model_kind": getattr(got, "model_kind", None)}
            src = getattr(got, "source", None)
            if src:
                try:
                    model_dir = Path(self.run_dir) / "logs" / "models"
                    model_dir.mkdir(parents=True, exist_ok=True)
                    src_path = model_dir / f"{got.id}.py"
                    src_path.write_text(src, encoding="utf-8")
                    entry["source_path"] = str(src_path)
                except Exception as exc:
                    logs.append(f"could not persist model source for {got.id}: {exc}")
            audit.append(entry)

        return model_ids, model_by_id, audit, logs

    def _plan_free_search_materialization(
        self,
        candidate_report: dict[str, Any],
        sample_df: pd.DataFrame,
        model_families: list[str],
    ) -> dict[str, Any]:
        """Plan {mechanism spec x model spec/family x fusion spec} before codegen.

        This is the search-space layer: it uses only structured specs, schema
        support, fusion readiness, and budget. LLM code is generated only for
        specs that survive this planned shortlist.
        """
        from knowledge.yield_joint_search import Candidate, apply_candidate_budget, build_pair_menu

        columns = list(sample_df.columns)
        n_samples = len(sample_df)
        raw_mech_specs = [s for s in (candidate_report.get("candidate_mechanisms") or []) if isinstance(s, dict)]
        raw_model_specs = [s for s in (candidate_report.get("candidate_models") or []) if isinstance(s, dict)]
        fusion_specs = candidate_report.get("candidate_fusion_specs", []) or None

        planned_mechanisms: list[Any] = []
        mech_spec_by_id: dict[str, dict[str, Any]] = {}
        skipped_mechanisms: list[dict[str, Any]] = []
        for spec in sorted(raw_mech_specs, key=_spec_score, reverse=True):
            shell = _PlannedMechanismSpec(spec)
            adequacy = shell.data_adequacy(columns)
            entry = {
                "searched_id": spec.get("id"),
                "planned_id": shell.id,
                "name": spec.get("name"),
                "adequacy": adequacy,
                "score": _spec_score(spec),
            }
            if adequacy.get("status") == "rejected":
                entry["reason"] = adequacy.get("reason")
                skipped_mechanisms.append(entry)
                continue
            if shell.id in mech_spec_by_id:
                entry["reason"] = f"duplicate planned mechanism id '{shell.id}'"
                skipped_mechanisms.append(entry)
                continue
            planned_mechanisms.append(shell)
            mech_spec_by_id[shell.id] = spec

        planned_model_ids: list[str] = []
        model_spec_by_plan_id: dict[str, dict[str, Any]] = {}
        skipped_models: list[dict[str, Any]] = []
        fixed_set = {str(f) for f in model_families}
        for spec in sorted(raw_model_specs, key=_spec_score, reverse=True):
            required = _required_columns_from_spec(spec)
            missing = [c for c in required if c not in columns]
            plan_prefix = "llm_latent" if _is_complex_model_spec(spec) else "llm_spec"
            plan_id = f"{plan_prefix}_{_slug_id(spec.get('id') or spec.get('name'), 'model')}"
            if not _model_spec_should_codegen(spec):
                skipped_models.append({
                    "searched_id": spec.get("id"),
                    "planned_id": plan_id,
                    "name": spec.get("name"),
                    "reason": (
                        f"covered by fixed_family floor '{_model_spec_duplicate_family(spec)}'; "
                        "not sent to LLM model codegen"
                    ),
                    "score": _spec_score(spec),
                })
                continue
            if missing:
                skipped_models.append({
                    "searched_id": spec.get("id"),
                    "planned_id": plan_id,
                    "name": spec.get("name"),
                    "reason": f"missing required columns: {missing}",
                    "score": _spec_score(spec),
                })
                continue
            if plan_id in model_spec_by_plan_id or plan_id in fixed_set:
                skipped_models.append({
                    "searched_id": spec.get("id"),
                    "planned_id": plan_id,
                    "name": spec.get("name"),
                    "reason": f"duplicate/conflicting planned model id '{plan_id}'",
                    "score": _spec_score(spec),
                })
                continue
            planned_model_ids.append(plan_id)
            model_spec_by_plan_id[plan_id] = spec

        model_ids_for_plan = list(model_families)
        for plan_id in planned_model_ids:
            if plan_id not in model_ids_for_plan:
                model_ids_for_plan.append(plan_id)

        planned_menu = build_pair_menu(
            [None] + planned_mechanisms,
            model_ids_for_plan,
            columns=columns,
            n_samples=n_samples,
            fusion_specs=fusion_specs,
        )

        plan_candidates: list[Candidate] = []
        for entry in planned_menu.get("admitted", []):
            mech_id = entry.get("mechanism")
            model_id = str(entry.get("model_family") or "")
            fusion_id = str(entry.get("fusion_id") or entry.get("fusion_mode") or "")
            origin = "planned_llm_model" if model_id in model_spec_by_plan_id else "fixed_family"
            plan_candidates.append(Candidate(
                f"{mech_id or 'raw'}::{model_id}::{fusion_id}",
                lambda: None,
                spec={
                    "mechanism": mech_id,
                    "model_family": model_id,
                    "model_origin": origin,
                    "fusion_mode": entry.get("fusion_mode"),
                    "fusion_id": fusion_id,
                    "fusion_spec": entry.get("fusion_spec"),
                    "status": entry.get("status"),
                },
            ))

        selected_plan_candidates, budget_audit = apply_candidate_budget(plan_candidates)
        if not selected_plan_candidates and plan_candidates:
            selected_plan_candidates = [plan_candidates[0]]

        selected_mech_ids = []
        selected_model_ids = []
        selected_fusion_ids = []
        for cand in selected_plan_candidates:
            spec = cand.spec or {}
            mech_id = spec.get("mechanism")
            model_id = str(spec.get("model_family") or "")
            fusion_id = str(spec.get("fusion_id") or "")
            if mech_id and mech_id not in selected_mech_ids:
                selected_mech_ids.append(str(mech_id))
            if model_id and model_id not in selected_model_ids:
                selected_model_ids.append(model_id)
            if fusion_id and fusion_id not in selected_fusion_ids:
                selected_fusion_ids.append(fusion_id)

        pre_codegen_mechanism_specs = [
            mech_spec_by_id[mid] for mid in selected_mech_ids if mid in mech_spec_by_id
        ]
        pre_codegen_model_specs = [
            model_spec_by_plan_id[mid] for mid in selected_model_ids if mid in model_spec_by_plan_id
        ]
        try:
            mechanism_codegen_cap = max(0, int(os.environ.get("YIELD_MAX_SEARCHED_MECHANISMS", "4")))
        except Exception:
            mechanism_codegen_cap = 4
        try:
            model_codegen_cap = max(0, int(os.environ.get("YIELD_MAX_SEARCHED_MODELS", "3")))
        except Exception:
            model_codegen_cap = 3
        selected_mechanism_specs = pre_codegen_mechanism_specs[:mechanism_codegen_cap]
        selected_model_specs, model_codegen_audit = _select_model_specs_for_codegen(
            pre_codegen_model_specs, model_codegen_cap
        )
        selected_fixed_families = [mid for mid in selected_model_ids if mid in fixed_set]
        if not selected_fixed_families:
            selected_fixed_families = list(model_families[:2])
        selected_fusion_specs = [
            spec for spec in planned_menu.get("fusion_specs", [])
            if str(spec.get("id") or "") in set(selected_fusion_ids)
        ] or list(planned_menu.get("fusion_specs", []))

        audit = {
            "stage": "planned_materialization",
            "policy": "specs are combined and screened before LLM code generation; only selected specs are materialized.",
            "n_input_mechanism_specs": len(raw_mech_specs),
            "n_input_model_specs": len(raw_model_specs),
            "n_planned_mechanism_specs": len(planned_mechanisms),
            "n_planned_llm_model_specs": len(planned_model_ids),
            "fixed_model_families_considered": list(model_families),
            "n_plan_candidates_before_budget": len(plan_candidates),
            "n_plan_candidates_selected": len(selected_plan_candidates),
            "codegen_caps": {
                "max_llm_mechanisms": mechanism_codegen_cap,
                "max_llm_models": model_codegen_cap,
                "max_llm_model_hparam_configs": int(os.environ.get("YIELD_MAX_LLM_MODEL_CONFIGS", "0") or 0),
            },
            "model_codegen_selection": model_codegen_audit,
            "pre_codegen_selected_mechanism_spec_ids": [str(s.get("id") or "") for s in pre_codegen_mechanism_specs],
            "pre_codegen_selected_llm_model_spec_ids": [str(s.get("id") or "") for s in pre_codegen_model_specs],
            "selected_mechanism_spec_ids": [str(s.get("id") or "") for s in selected_mechanism_specs],
            "selected_llm_model_spec_ids": [str(s.get("id") or "") for s in selected_model_specs],
            "selected_fixed_model_families": selected_fixed_families,
            "selected_fusion_ids": [str(s.get("id") or "") for s in selected_fusion_specs],
            "budget": budget_audit,
            "pair_menu_summary": {
                "n_pairs": planned_menu.get("n_pairs"),
                "n_admitted": planned_menu.get("n_admitted"),
                "n_rejected": planned_menu.get("n_rejected"),
                "fusion_specs_before_execution_dedupe": planned_menu.get("fusion_specs_before_execution_dedupe"),
                "fusion_specs_after_execution_dedupe": planned_menu.get("fusion_specs_after_execution_dedupe"),
                "fusion_spec_duplicates": planned_menu.get("fusion_spec_duplicates"),
            },
            "selected_plan_candidates": [
                {
                    "name": cand.name,
                    "mechanism": (cand.spec or {}).get("mechanism"),
                    "model_family": (cand.spec or {}).get("model_family"),
                    "model_origin": (cand.spec or {}).get("model_origin"),
                    "fusion_id": (cand.spec or {}).get("fusion_id"),
                    "fusion_mode": (cand.spec or {}).get("fusion_mode"),
                }
                for cand in selected_plan_candidates
            ],
            "skipped_mechanisms": skipped_mechanisms,
            "skipped_models": skipped_models,
        }
        try:
            plan_dir = Path(self.run_dir) / "logs"
            plan_dir.mkdir(parents=True, exist_ok=True)
            (plan_dir / "materialization_plan.json").write_text(
                json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception:
            pass

        return {
            "selected_mechanism_specs": selected_mechanism_specs,
            "selected_model_specs": selected_model_specs,
            "selected_fixed_model_families": selected_fixed_families,
            "selected_fusion_specs": selected_fusion_specs,
            "audit": audit,
        }

    def _run_free_search_harness(self) -> dict[str, Any]:
        """Direction-④ execution mode: FREE search over searched mechanisms x
        searched models x fusion modes, scored by the deterministic joint-search
        harness, with the champion persisted and lightly verified."""
        from knowledge.yield_joint_search import make_candidate_builder, run_free_search, search_budget_config
        from knowledge.yield_schema import TARGET_COLUMN, load_yield_dataframe
        from operation_agent.yield_free_search_harness import (
            finalize_and_write,
            model_families_from_candidates,
            verify_free_search_run,
        )

        self._transition("EXEC", "OperationAgent running the mechanism x model x fusion-mode free search.")
        print("YIELD_STAGE: operation")
        started = time.time()
        budget_cfg = search_budget_config()
        budget_defaults = {
            "quick": {"YIELD_MAX_SEARCHED_MODELS": "2", "YIELD_MAX_SEARCHED_MECHANISMS": "2"},
            "normal": {"YIELD_MAX_SEARCHED_MODELS": "3", "YIELD_MAX_SEARCHED_MECHANISMS": "3"},
            "full": {"YIELD_MAX_SEARCHED_MODELS": "5", "YIELD_MAX_SEARCHED_MECHANISMS": "5"},
        }.get(str(budget_cfg.get("mode") or "normal"), {})
        for key, value in budget_defaults.items():
            os.environ.setdefault(key, value)
        llm_config_defaults = {"quick": 1, "normal": 3, "full": 6, "unlimited": 0}
        os.environ.setdefault(
            "YIELD_MAX_LLM_MODEL_CONFIGS",
            str(llm_config_defaults.get(str(budget_cfg.get("mode") or "normal"),
                                        budget_cfg.get("max_hparams_per_model") or 2)),
        )
        _emit_event(
            "operation",
            "OperationAgent:",
            "Free-search budget: " + json.dumps(budget_cfg, ensure_ascii=False, sort_keys=True),
            mirror=False,
        )
        data_path = os.environ["YIELD_DATA_PATH"]
        anchor_path = os.environ.get("YIELD_ANCHOR_PATH") or os.environ.get("YIELD_TEST_PATH")
        anchor_expected = bool(anchor_path and Path(anchor_path).exists())

        sample_df, _ = load_yield_dataframe(data_path)
        sample_df = sample_df.head(64)
        y_sample = pd.to_numeric(sample_df[TARGET_COLUMN], errors="coerce").to_numpy(float)
        sample_x = sample_df.drop(columns=[TARGET_COLUMN], errors="ignore")

        candidate_report = self.candidate_report if isinstance(self.candidate_report, dict) else {}
        model_families = model_families_from_candidates(candidate_report.get("candidate_models", []) or [])
        materialization = self._plan_free_search_materialization(candidate_report, sample_df, model_families)
        materialization_audit = materialization.get("audit") or {}
        _emit_event(
            "operation",
            "OperationAgent:",
            "Planned materialization: "
            f"{len(materialization.get('selected_mechanism_specs') or [])} mechanism specs, "
            f"{len(materialization.get('selected_model_specs') or [])} LLM model specs, "
            f"{len(materialization.get('selected_fixed_model_families') or [])} fixed families, "
            f"{len(materialization.get('selected_fusion_specs') or [])} fusion specs.",
            mirror=False,
        )
        llm_model_ids, model_by_id, model_audit, model_logs = self._obtain_searched_models(
            materialization.get("selected_model_specs") or [], sample_x, y_sample,
            materialization.get("selected_fixed_model_families") or model_families
        )
        selected_fixed_families = materialization.get("selected_fixed_model_families") or model_families
        model_search_ids = list(selected_fixed_families) + [m for m in llm_model_ids if m not in selected_fixed_families]
        mechanisms, mech_by_id, mech_audit, gen_logs = self._obtain_searched_mechanisms(
            materialization.get("selected_mechanism_specs") or [], sample_df
        )

        builder = make_candidate_builder(mech_by_id, model_by_id)
        result = run_free_search(
            [None] + mechanisms, model_search_ids, data_path,
            candidate_builder=builder, anchor_path=anchor_path,
            fusion_specs=materialization.get("selected_fusion_specs")
            or candidate_report.get("candidate_fusion_specs", []) or None,
            n_splits=5, random_state=self._current_random_state(), min_anchor_r2=0.0,
        )
        result["random_state"] = self._current_random_state()
        result["base_random_state"] = int(self.args.random_state)
        result["materialization_plan"] = materialization_audit

        op_result = finalize_and_write(self.run_dir, result, data_path, builder,
                                       searched_mechanisms=mech_audit,
                                       searched_models=model_audit)
        op_result["materialization_plan"] = materialization_audit
        op_result["search_budget"] = result.get("budget")
        op_result["search_timing"] = result.get("timing")
        op_result.setdefault("error_logs", []).extend(gen_logs)
        op_result.setdefault("error_logs", []).extend(model_logs)

        verification = verify_free_search_run(
            str(self.run_dir), started_at=started, anchor_expected=anchor_expected
        )
        op_result["free_search_verification"] = verification
        if not verification["passed"]:
            op_result["rcode"] = 1
            op_result["error_logs"].extend(verification["reasons"])
            _emit_event("operation", "OperationAgent:",
                        "Free-search acceptance failed:\n" + "\n".join(f"- {r}" for r in verification["reasons"]),
                        mirror=False)
        else:
            champ = op_result.get("champion") or {}
            _emit_event("operation", "OperationAgent:",
                        f"Free search accepted: champion={champ.get('name')} "
                        f"(pool={(result.get('search') or {}).get('champion_pool_basis')}).",
                        mirror=False)
        return op_result

    def _run_operation_agent(self) -> dict[str, Any]:
        self._transition("PRE_EXEC", "AgentManager building operation contract and checking readiness.")
        self.operation_instructions = self._build_operation_instructions()
        review = self._pre_execution_review()
        if not review["passed"]:
            return {
                "rcode": 3,
                "stage": "pre_execution_review",
                "action_result": "AgentManager blocked OperationAgent because pre-execution review failed.",
                "code": "",
                "error_logs": [json.dumps(review, ensure_ascii=False)],
                "pre_execution_review": review,
            }

        _exec_mode = os.environ.get("YIELD_EXECUTION_MODE", "free_search").strip().lower()
        if _exec_mode == "free_search":
            return self._run_free_search_harness()
        if _exec_mode != "llm_freeform":
            return self._run_plugin_harness()

        self._transition("EXEC", "OperationAgent generating and executing training code.")
        print("YIELD_STAGE: operation")
        result = _run_operation_agent_with_process_timeout(
            self._yield_user_requirements(),
            self.args.llm,
            "/yield_search_generated",
            "yield_stress_regression",
            self.operation_instructions,
            max(1, int(self.args.operation_attempts)),
        )
        error_text = "\n".join(str(item) for item in (result.get("error_logs", []) or []))
        action_text = str(result.get("action_result") or "")
        combined_result_text = (error_text + "\n" + action_text).lower()
        connection_failed = any(
            token in combined_result_text
            for token in ("connection error", "connectionerror", "api connection", "timeout", "timed out", "连接")
        )
        structural_artifact_failure = any(
            token in combined_result_text
            for token in (
                "no fresh metrics json artifact",
                "no fresh prediction csv",
                "no fresh model artifact",
                "no fresh preprocessing artifact",
                "mechanism_report.json is missing or invalid",
                "yodel/packing audit requires",
                "generated script exited with non-zero return code",
                "yield preflight static check failed",
                "preflight static check failed",
                "reads leakage/reference columns",
                "reads leakage",
            )
        )
        try:
            result_rcode = int(result.get("rcode", -1))
        except Exception:
            result_rcode = -1

        def run_benchmark_proxy_fallback(fallback_reason: str, note: str) -> dict[str, Any]:
            _emit_event(
                "operation",
                "OperationAgent:",
                note,
                mirror=False,
            )
            try:
                fallback = execute_selected_benchmark_proxy_artifacts(
                    os.environ["YIELD_DATA_PATH"],
                    self.candidate_report,
                    self.candidate_benchmark_report,
                    self.run_dir,
                    anchor_path=os.environ.get("YIELD_ANCHOR_PATH") or os.environ.get("YIELD_TEST_PATH"),
                    random_state=(
                        (self.candidate_benchmark_report.get("benchmark_protocol") or {}).get("random_state")
                        if isinstance(self.candidate_benchmark_report, dict)
                        else None
                    ),
                    n_splits=5,
                    fallback_reason=fallback_reason,
                )
                fallback["error_logs"] = list(result.get("error_logs", []) or []) + [
                    f"OperationAgent fallback executed benchmark-selected proxy: {fallback_reason}."
                ]
                return fallback
            except Exception as exc:
                result.setdefault("error_logs", []).append(
                    f"Benchmark-selected proxy fallback failed: {type(exc).__name__}: {exc}"
                )
                return result

        if result_rcode != 0 and connection_failed:
            result = run_benchmark_proxy_fallback(
                "operation_llm_connection_or_timeout",
                "LLM code generation failed with connection/timeout errors; executing benchmark-selected proxy fallback.",
            )
        elif result_rcode != 0 and structural_artifact_failure:
            result = run_benchmark_proxy_fallback(
                "operation_structural_artifact_failure",
                "Generated code failed required artifact/mechanism guardrails; executing benchmark-selected proxy fallback.",
            )

        generated_code = str(result.get("code") or "")
        if generated_code.strip():
            code_path = self.run_dir / "generated_code.py"
            code_path.write_text(generated_code, encoding="utf-8")
            _emit_event(
                "operation",
                "OperationAgent:",
                f"YIELD_CODE_SAVED: {code_path.resolve()}",
                mirror=False,
            )
            _emit_event(
                "operation",
                "OperationAgent:",
                f"YIELD_GENERATED_CODE\nPath: {code_path.resolve()}\n{generated_code}",
                mirror=False,
            )
        action_result = str(result.get("action_result") or "").strip()
        if action_result:
            _emit_event(
                "operation",
                "OperationAgent:",
                "YIELD_TRAINING_LOG\n" + action_result[-8000:],
                mirror=False,
            )
        return result

    def _research_quality_review(self) -> dict[str, Any]:
        metrics = _read_json_artifact(self.run_dir / "metrics" / "metrics.json")
        free_search_report = _read_json_artifact(self.run_dir / "metrics" / "free_search_report.json")
        mechanism_report = _read_json_artifact(self.run_dir / "logs" / "mechanism_report.json")
        candidate_audit = _read_json_artifact(self.run_dir / "candidate_selection_audit.json")
        checks: list[dict[str, Any]] = []

        def add_check(
            name: str,
            passed: bool,
            severity: str,
            detail: str,
            revision_hint: str,
        ) -> None:
            checks.append(
                {
                    "name": name,
                    "passed": bool(passed),
                    "severity": severity,
                    "detail": detail,
                    "revision_hint": revision_hint,
                }
            )

        if metrics is None and isinstance(free_search_report, dict):
            champion = free_search_report.get("champion") if isinstance(free_search_report, dict) else None
            champion = champion if isinstance(champion, dict) else {}
            oof_metrics = champion.get("oof_metrics") if isinstance(champion, dict) else None
            oof_metrics = oof_metrics if isinstance(oof_metrics, dict) else {}
            rmse = _numeric_metric(oof_metrics, ("rmse", "RMSE", "oof_rmse"))
            r2 = _numeric_metric(oof_metrics, ("r2", "R2", "oof_r2"))
            baseline_status = str(champion.get("baseline_status") or "").strip()
            champion_pool = str(free_search_report.get("champion_pool_basis") or "").strip()
            diagnostics = free_search_report.get("contribution_diagnostics")
            ranked = free_search_report.get("ranked")
            add_check(
                "free_search_report_available",
                True,
                "critical",
                "metrics/free_search_report.json is available.",
                "Free-search runs should save metrics/free_search_report.json.",
            )
            add_check(
                "free_search_champion_available",
                bool(champion.get("name")),
                "critical",
                f"champion={champion.get('name')}",
                "Free-search report must record the selected champion candidate.",
            )
            add_check(
                "free_search_oof_metrics_available",
                rmse is not None and r2 is not None,
                "critical",
                f"rmse={rmse}, r2={r2}",
                "Free-search report must record finite OOF RMSE/R2 for the champion.",
            )
            add_check(
                "free_search_baseline_comparison_recorded",
                bool(baseline_status or champion_pool),
                "major",
                f"baseline_status={baseline_status or 'missing'}, champion_pool={champion_pool or 'missing'}",
                "Free-search report must state whether the champion beats, ties, or loses to the fixed baseline.",
            )
            add_check(
                "free_search_contribution_diagnostics_available",
                isinstance(diagnostics, dict),
                "major",
                "contribution_diagnostics present" if isinstance(diagnostics, dict) else "contribution_diagnostics missing",
                "Free-search report should separate model contribution from mechanism contribution.",
            )
            add_check(
                "free_search_ranked_candidates_available",
                isinstance(ranked, list) and len(ranked) > 0,
                "major",
                f"ranked_count={len(ranked) if isinstance(ranked, list) else 0}",
                "Free-search report should include the ranked candidate table.",
            )
            failing = [item for item in checks if not item["passed"] and item["severity"] in {"critical", "major"}]
            return {
                "passed": not failing,
                "status": "passed" if not failing else "needs_research_revision",
                "checks": checks,
                "issues": [str(item["detail"]) for item in failing],
                "revision_hints": [str(item["revision_hint"]) for item in failing],
                "metrics_source": "metrics/free_search_report.json",
                "champion": champion.get("name"),
                "champion_rmse": rmse,
                "champion_r2": r2,
                "baseline_status": baseline_status,
            }

        if metrics is None:
            return {
                "passed": False,
                "status": "missing_metrics",
                "checks": [
                    {
                        "name": "metrics_available",
                        "passed": False,
                        "severity": "critical",
                        "detail": "metrics/metrics.json is missing or invalid.",
                        "revision_hint": "OperationAgent must save metrics before research quality can be reviewed.",
                    }
                ],
                "issues": ["metrics/metrics.json is missing or invalid."],
            }

        anchor = _nested_dict(metrics, "anchor_validation", "anchor_validation_metrics", "anchor_metrics")
        anchor_r2 = _numeric_metric(anchor, ("r2", "R2", "anchor_r2"))
        if anchor_r2 is None:
            anchor_r2 = _numeric_metric(metrics, ("anchor_r2", "anchor_R2"))
        anchor_path = os.environ.get("YIELD_ANCHOR_PATH")
        if anchor_path and Path(anchor_path).exists():
            add_check(
                "anchor_validation_available",
                anchor_r2 is not None,
                "critical",
                f"anchor_r2={anchor_r2}",
                "Evaluate the final model on the real anchor dataset and record anchor_validation.r2.",
            )
        if anchor_r2 is not None:
            add_check(
                "anchor_generalization_nonnegative",
                anchor_r2 >= 0,
                "critical",
                f"anchor_r2={anchor_r2:.6g}",
                "Anchor R2 is negative. Reconsider candidate strategy, synthetic-real gap, feature pipeline, and final implementation before accepting the run.",
            )
            add_check(
                "anchor_generalization_minimum_signal",
                anchor_r2 >= 0.3,
                "major",
                f"anchor_r2={anchor_r2:.6g}",
                "Anchor R2 is weak. Prefer strategies that improve real-anchor validation rather than only synthetic OOF.",
            )

        confidence = str(metrics.get("research_confidence") or "").strip().lower()
        add_check(
            "research_confidence_not_low",
            confidence not in {"", "low"},
            "major",
            f"research_confidence={confidence or 'missing'}",
            "Low research confidence should trigger another Candidate/Model revision when revision budget remains.",
        )

        beats = metrics.get("beats_best_baseline")
        add_check(
            "beats_fixed_baseline",
            beats is True,
            "major",
            f"beats_best_baseline={beats}",
            "The research model must beat fixed baselines under the declared selection metric, or the strategy should be revised.",
        )

        implementation = _nested_dict(metrics, "implementation_validation")
        if implementation is None and isinstance(mechanism_report, dict):
            implementation = _nested_dict(mechanism_report, "implementation_validation")
        if implementation is None:
            add_check(
                "implementation_validation_available",
                False,
                "major",
                "implementation_validation missing",
                "Record actual-vs-benchmark implementation metrics and revise if the final implementation drifts from the benchmark-selected strategy.",
            )
        else:
            benchmark_oof = _numeric_metric(implementation, ("benchmark_oof_rmse", "benchmark_selected_oof_rmse"))
            actual_oof = _numeric_metric(implementation, ("actual_oof_rmse", "implemented_oof_rmse", "final_oof_rmse"))
            if benchmark_oof is not None and actual_oof is not None:
                add_check(
                    "actual_oof_not_much_worse_than_benchmark",
                    actual_oof <= benchmark_oof * 1.25,
                    "major",
                    f"benchmark_oof_rmse={benchmark_oof:.6g}, actual_oof_rmse={actual_oof:.6g}",
                    "Final implementation is much worse than the benchmark proxy; implement the benchmark-selected proxy directly or explain a stronger fallback.",
                )
            benchmark_anchor_r2 = _numeric_metric(implementation, ("benchmark_anchor_r2", "benchmark_selected_anchor_r2"))
            actual_anchor_r2 = _numeric_metric(implementation, ("actual_anchor_r2", "implemented_anchor_r2", "final_anchor_r2"))
            if actual_anchor_r2 is not None:
                add_check(
                    "actual_anchor_generalization_nonnegative",
                    actual_anchor_r2 >= 0,
                    "critical",
                    f"actual_anchor_r2={actual_anchor_r2:.6g}",
                    "Final implementation fails real-anchor generalization. Revise candidate/model instead of accepting synthetic-only performance.",
                )
            if benchmark_anchor_r2 is not None and actual_anchor_r2 is not None:
                add_check(
                    "actual_anchor_close_to_benchmark",
                    actual_anchor_r2 >= benchmark_anchor_r2 - 0.2,
                    "critical",
                    f"benchmark_anchor_r2={benchmark_anchor_r2:.6g}, actual_anchor_r2={actual_anchor_r2:.6g}",
                    "The final implementation lost the benchmark-selected strategy's anchor performance. Run the selected proxy as a required candidate with its own metrics before selecting a fallback.",
                )
                proxy_result = metrics.get("benchmark_selected_proxy_result")
                if not isinstance(proxy_result, dict) and isinstance(mechanism_report, dict):
                    proxy_result = mechanism_report.get("benchmark_selected_proxy_result")
                add_check(
                    "benchmark_selected_proxy_executed",
                    isinstance(proxy_result, dict),
                    "critical",
                    "benchmark_selected_proxy_result present" if isinstance(proxy_result, dict) else "benchmark_selected_proxy_result missing",
                    "OperationAgent must execute the benchmark-selected proxy model/feature mode as an explicit candidate and save benchmark_selected_proxy_result.",
                )

        ablation = _nested_dict(metrics, "mechanism_ablation")
        if ablation is None and isinstance(mechanism_report, dict):
            ablation = _nested_dict(mechanism_report, "mechanism_ablation", "ablation_study")
        mechanism_used = isinstance(mechanism_report, dict) and mechanism_report.get("mechanism_used") is True
        if mechanism_used:
            improves = ablation.get("mechanism_improves_metric") if isinstance(ablation, dict) else None
            delta_rmse = _numeric_metric(ablation, ("delta_rmse", "rmse_delta")) if isinstance(ablation, dict) else None
            effect = _nested_dict(metrics, "mechanism_effect_size")
            if effect is None and isinstance(mechanism_report, dict):
                effect = _nested_dict(mechanism_report, "mechanism_effect_size")
            relative = _numeric_metric(effect, ("relative_rmse_improvement", "relative_improvement")) if isinstance(effect, dict) else None
            add_check(
                "mechanism_ablation_positive",
                improves is True and (delta_rmse is None or delta_rmse > 0) and (relative is None or relative >= 0.01),
                "major",
                f"mechanism_improves_metric={improves}, delta_rmse={delta_rmse}, relative_rmse_improvement={relative}",
                "Mechanism did not show a meaningful same-protocol gain. CandidateAgent should either revise the mechanism or treat it as attempted/no-gain.",
            )

        selected_benchmark = (
            candidate_audit.get("selected_strategy_benchmark")
            if isinstance(candidate_audit, dict)
            else None
        )
        benchmark_status = (
            selected_benchmark.get("anchor_generalization_status")
            if isinstance(selected_benchmark, dict)
            else None
        )
        if benchmark_status:
            add_check(
                "benchmark_anchor_status_not_failed",
                str(benchmark_status).lower() != "failed",
                "major",
                f"selected_strategy_anchor_generalization_status={benchmark_status}",
                "Candidate benchmark selected strategy has failed anchor generalization; revise candidate selection with stronger anchor weighting.",
            )

        failing = [item for item in checks if not item["passed"] and item["severity"] in {"critical", "major"}]
        status = "passed" if not failing else "needs_research_revision"
        return {
            "passed": not failing,
            "status": status,
            "checks": checks,
            "issues": [str(item["detail"]) for item in failing],
            "revision_hints": [str(item["revision_hint"]) for item in failing],
            "anchor_r2": anchor_r2,
            "research_confidence": confidence,
        }

    def _post_execution_review(self, result: dict[str, Any], remaining_revisions: int = 0) -> dict[str, Any]:
        self._transition("POST_EXEC", "AgentManager reviewing OperationAgent result and mandatory artifacts.")
        if os.environ.get("YIELD_EXECUTION_MODE", "free_search").strip().lower() == "free_search":
            required_artifacts = {
                "run_result": self.run_dir / "run_result.json",
                "free_search_report": self.run_dir / "metrics" / "free_search_report.json",
                "predictions": self.run_dir / "metrics" / "predictions.csv",
                "mechanism_report": self.run_dir / "logs" / "mechanism_report.json",
                "predict_script": self.run_dir / "predict.py",
                "champion_model": self.run_dir / "trained_models" / "champion.joblib",
            }
        else:
            required_artifacts = {
                "run_result": self.run_dir / "run_result.json",
                "metrics": self.run_dir / "metrics" / "metrics.json",
                "predictions": self.run_dir / "predictions" / "yield_predictions.csv",
                "mechanism_report": self.run_dir / "logs" / "mechanism_report.json",
                "synthetic_data_report": self.run_dir / "logs" / "synthetic_data_report.json",
            }
        artifact_status = {
            name: {
                "path": str(path),
                "exists": path.exists(),
                "size": path.stat().st_size if path.exists() else 0,
            }
            for name, path in required_artifacts.items()
        }
        artifact_passed = result.get("rcode") == 0
        research_quality = self._research_quality_review() if artifact_passed else {
            "passed": False,
            "status": "not_reviewed_due_to_execution_failure",
            "checks": [],
            "issues": [],
            "revision_hints": [],
        }
        quality_passed = bool(research_quality.get("passed"))
        passed = artifact_passed and (quality_passed or remaining_revisions <= 0)
        if artifact_passed and quality_passed:
            decision = "accepted"
        elif artifact_passed and not quality_passed and remaining_revisions > 0:
            decision = "revise_candidate_and_model_research_quality"
        elif artifact_passed and not quality_passed:
            decision = "accepted_with_research_warning"
        elif remaining_revisions > 0:
            decision = "revise_candidate_and_model"
        else:
            decision = "needs_manual_review"
        review = {
            "passed": passed,
            "result_rcode": result.get("rcode"),
            "manager_state": self.state,
            "manager_revision_round": self.manager_revision_round,
            "remaining_manager_revisions": max(0, int(remaining_revisions)),
            "artifact_status": artifact_status,
            "research_quality_review": research_quality,
            "error_log_count": len(result.get("error_logs", []) or []),
            "decision": decision,
        }
        self._write_json("post_execution_review.json", review)
        if self.manager_revision_round:
            self._write_json(f"post_execution_review_revision_{self.manager_revision_round}.json", review)
        next_state = "END" if review["passed"] else "REV" if remaining_revisions > 0 else "FAILED"
        self._transition(
            next_state,
            (
                "AgentManager accepted run."
                if review["decision"] == "accepted"
                else "AgentManager accepted run with research-quality warning after exhausting revisions."
                if review["decision"] == "accepted_with_research_warning"
                else "AgentManager will revise candidate/model plan from execution feedback."
                if remaining_revisions > 0
                else "AgentManager marked run for manual review after exhausting revisions."
            ),
            review,
        )
        return review

    def _build_manager_revision_feedback(self, result: dict[str, Any], post_review: dict[str, Any]) -> str:
        artifact_status = post_review.get("artifact_status", {}) if isinstance(post_review, dict) else {}
        missing_artifacts = [
            name
            for name, status in artifact_status.items()
            if isinstance(status, dict) and (not status.get("exists") or int(status.get("size") or 0) <= 0)
        ]
        error_logs = [str(item) for item in (result.get("error_logs", []) or [])]
        quality = post_review.get("research_quality_review", {}) if isinstance(post_review, dict) else {}
        quality_issues = quality.get("issues", []) if isinstance(quality, dict) else []
        quality_hints = quality.get("revision_hints", []) if isinstance(quality, dict) else []
        selection_audit = self.candidate_selection_audit if isinstance(self.candidate_selection_audit, dict) else {}
        anchor_viability = selection_audit.get("anchor_viability_summary") if isinstance(selection_audit, dict) else {}
        shift_audit = selection_audit.get("synthetic_anchor_shift_audit") if isinstance(selection_audit, dict) else {}
        shift_summary = {}
        if isinstance(shift_audit, dict):
            shift_summary = {
                "status": shift_audit.get("status"),
                "reason": shift_audit.get("reason"),
                "shared_feature_count": shift_audit.get("shared_feature_count"),
                "train_only_features": shift_audit.get("train_only_features"),
                "anchor_only_features": shift_audit.get("anchor_only_features"),
                "recommended_anchor_feature_columns": shift_audit.get("recommended_anchor_feature_columns"),
                "severe_shift_feature_columns": shift_audit.get("severe_shift_feature_columns"),
            }
        strategy_summaries = []
        for item in selection_audit.get("strategy_benchmark_summaries", []) or []:
            if not isinstance(item, dict):
                continue
            strategy_summaries.append(
                {
                    "strategy_id": item.get("strategy_id"),
                    "proxy_model": item.get("proxy_model"),
                    "feature_mode": item.get("feature_mode"),
                    "evaluated_base_feature_columns": item.get("evaluated_base_feature_columns"),
                    "oof_r2": item.get("oof_r2"),
                    "anchor_r2": item.get("anchor_r2"),
                    "anchor_rmse": item.get("anchor_rmse"),
                    "anchor_generalization_status": item.get("anchor_generalization_status"),
                    "anchor_viability_status": item.get("anchor_viability_status"),
                    "anchor_shift_risk": item.get("anchor_shift_risk"),
                    "audited_selection_score_1_to_10": item.get("audited_selection_score_1_to_10"),
                }
            )
        feedback = (
            f"Manager revision round {self.manager_revision_round} after OperationAgent result review.\n"
            f"- Result rcode: {result.get('rcode')}\n"
            f"- Post-execution decision: {post_review.get('decision')}\n"
            f"- Missing or empty artifacts: {missing_artifacts or 'none reported'}\n"
            f"- Candidate anchor viability: {anchor_viability or 'not available'}\n"
            f"- Synthetic-anchor shift summary: {shift_summary or 'not available'}\n"
            f"- Candidate benchmark details: {strategy_summaries[:6] or 'not available'}\n"
            f"- Research quality issues: {quality_issues or 'none reported'}\n"
            f"- Research quality revision hints: {quality_hints or 'none reported'}\n"
            f"- Last action output:\n{str(result.get('action_result') or '')[-4000:]}\n"
            f"- Last error logs:\n{chr(10).join(error_logs[-2:])[-4000:] if error_logs else 'none'}\n"
            "Revision instruction: update CandidateAgent and ModelAgent outputs before the next OperationAgent call. "
            "Prefer a simpler runnable fallback if the previous model failed due to package availability, JSON serialization, "
            "missing artifacts, leakage risk, unsupported mechanism columns, guardrail violations, negative anchor R2, "
            "benchmark-to-implementation degradation, or no-gain mechanism ablation. Preserve the yield schema, 5-fold "
            "OOF, anchor validation, baseline comparison, feature-schema preprocessing audit, and mandatory artifact contract. "
            "When no viable anchor candidate exists, build the next candidate list around recommended_anchor_feature_columns, "
            "and either exclude severe_shift_feature_columns or include them only in an explicit ablation/reweighting candidate. "
            "If the benchmark-selected proxy looked good but the final implementation failed anchor validation, require "
            "OperationAgent to run the benchmark-selected proxy as an explicit candidate and save benchmark_selected_proxy_result "
            "before choosing a fallback."
        )
        note = {
            "manager_revision_round": self.manager_revision_round,
            "feedback": feedback,
            "result_rcode": result.get("rcode"),
            "post_execution_review": post_review,
            "error_logs": error_logs,
        }
        self.revision_notes.append(note)
        self._write_json(f"manager_revision_{self.manager_revision_round}.json", note)
        self._write_json("manager_revision_notes.json", {"revision_notes": self.revision_notes})
        _emit_event("manager", "Agent Manager:", "Revision feedback prepared for CandidateAgent/ModelAgent.", mirror=False)
        return feedback

    def _save_result(self, result: dict[str, Any], filename: str = "run_result.json", announce: bool = True) -> Path:
        status_path = self.run_dir / filename
        status_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        if announce:
            print(f"YIELD_RUN_DIR: {self.run_dir.resolve()}")
            print(f"YIELD_RESULT_SAVED: {status_path.resolve()}")
        return status_path

    def run(self) -> int:
        try:
            self._prepare_run()
            self._run_data_agent()
            self._emit_requirement_context()
            if not self._run_search_agent():
                return 2
            self._run_candidate_agent()
            self._run_model_agent()
            max_manager_revisions = max(0, int(self.args.n_revise))
            for manager_round in range(max_manager_revisions + 1):
                self.manager_revision_round = manager_round
                result = self._run_operation_agent()
                result["manager_revision_round"] = manager_round
                result["operation_attempt_budget"] = max(1, int(self.args.operation_attempts))
                self._save_result(result, f"run_result_manager_round_{manager_round}.json", announce=False)
                self._save_result(result)
                remaining_revisions = max_manager_revisions - manager_round
                post_review = self._post_execution_review(result, remaining_revisions=remaining_revisions)
                if post_review.get("passed"):
                    return 0
                if remaining_revisions <= 0:
                    return 1
                self.manager_revision_round = manager_round + 1
                feedback = self._build_manager_revision_feedback(result, post_review)
                self._run_candidate_agent(manager_feedback=feedback)
                self._run_model_agent(manager_feedback=feedback)
            return 1
        except Exception as exc:
            result = {
                "rcode": -1,
                "stage": self.state,
                "action_result": f"YieldAgentManager failed in state {self.state}: {type(exc).__name__}: {exc}",
                "code": "",
                "error_logs": [f"{type(exc).__name__}: {exc}"],
            }
            self._save_result(result)
            self._transition("FAILED", "AgentManager crashed.", result)
            raise


def main() -> int:
    return YieldAgentManager(parse_args()).run()


if __name__ == "__main__":
    raise SystemExit(main())
