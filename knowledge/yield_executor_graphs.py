"""Declarative executor-graph handling for yield free-search.

An executor graph is a small, JSON-like workflow that describes how mechanism
pieces and model pieces are connected. It is not executable Python. The harness
can validate the graph, classify graphs that match currently supported
execution patterns, and leave new patterns as planned-only until an interpreter
exists.
"""

from __future__ import annotations

from typing import Any


EXECUTABLE_GRAPH_MODES = {
    "raw_ml",
    "mechanism_features",
    "mechanism_residual",
    "hidden_parameter_physics",
    "multi_fidelity_base_residual",
}

ALLOWED_GRAPH_OPS = {
    "select_raw_features",
    "fit_mechanism_params",
    "compute_mechanism_features",
    "compute_mechanism_base",
    "fit_model",
    "predict_model",
    "fit_residual_model",
    "predict_residual_model",
    "add_predictions",
    "infer_latent_targets",
    "fit_latent_model",
    "predict_latent_model",
    "compute_physics_output",
    "fit_latent_physics_model",
    "predict_latent_physics_model",
    "fit_correction_model",
    "predict_correction_model",
    "fit_low_fidelity_base",
    "fit_high_fidelity_residual",
}

FIT_ONLY_OPS = {
    "fit_mechanism_params",
    "fit_model",
    "fit_residual_model",
    "infer_latent_targets",
    "fit_latent_model",
    "fit_latent_physics_model",
    "fit_correction_model",
    "fit_low_fidelity_base",
    "fit_high_fidelity_residual",
}

PREDICT_ONLY_OPS = {
    "predict_model",
    "predict_residual_model",
    "predict_latent_model",
    "predict_latent_physics_model",
    "predict_correction_model",
    "add_predictions",
}

TARGET_USING_OPS = {
    "fit_mechanism_params",
    "fit_model",
    "fit_residual_model",
    "infer_latent_targets",
    "fit_latent_model",
    "fit_latent_physics_model",
    "fit_correction_model",
    "fit_low_fidelity_base",
    "fit_high_fidelity_residual",
}

FORBIDDEN_GRAPH_KEYS = {"code", "source_code", "python_code", "script", "callable", "lambda"}


def _slug(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    return "".join(ch for ch in text if ch.isalnum() or ch == "_").strip("_")


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def normalize_executor_graph(graph: Any, *, default_id: str = "executor_graph") -> dict[str, Any]:
    """Normalize graph-like input into {id, steps}.

    Supported user-facing shapes:
    - {"steps": [{"op": "..."}]}
    - [{"op": "..."}, {"op": "..."}]
    - ["select_raw_features", "fit_model", "predict_model"]
    """
    if isinstance(graph, list):
        raw = {"id": default_id, "steps": graph}
    elif isinstance(graph, dict):
        raw = dict(graph)
        raw.setdefault("id", default_id)
        if "steps" not in raw and "graph" in raw:
            raw["steps"] = raw.get("graph")
    else:
        raw = {"id": default_id, "steps": []}

    steps: list[dict[str, Any]] = []
    for idx, item in enumerate(_as_list(raw.get("steps"))):
        if isinstance(item, str):
            step = {"id": f"s{idx + 1}", "op": item}
        elif isinstance(item, dict):
            step = dict(item)
            step.setdefault("id", f"s{idx + 1}")
            step["op"] = step.get("op") or step.get("operation") or step.get("type")
        else:
            step = {"id": f"s{idx + 1}", "op": ""}
        step["id"] = _slug(step.get("id") or f"s{idx + 1}") or f"s{idx + 1}"
        step["op"] = _slug(step.get("op"))
        if "inputs" in step and not isinstance(step["inputs"], list):
            step["inputs"] = [step["inputs"]]
        if "outputs" in step and not isinstance(step["outputs"], list):
            step["outputs"] = [step["outputs"]]
        steps.append(step)

    return {
        **raw,
        "id": _slug(raw.get("id") or default_id) or default_id,
        "steps": steps,
    }


def builtin_executor_graph(mode: str, *, graph_id: str | None = None) -> dict[str, Any]:
    """Return the graph form of the three currently executable patterns."""
    mode = _slug(mode)
    gid = _slug(graph_id or f"{mode}_graph") or f"{mode}_graph"
    if mode == "raw_ml":
        steps = [
            {"id": "raw_features", "op": "select_raw_features", "phase": "fit_predict", "outputs": ["raw_features"]},
            {"id": "fit_model", "op": "fit_model", "phase": "fit", "inputs": ["raw_features", "train_y"], "outputs": ["model"]},
            {"id": "predict_model", "op": "predict_model", "phase": "predict", "inputs": ["raw_features", "model"], "outputs": ["prediction"]},
        ]
    elif mode == "mechanism_features":
        steps = [
            {"id": "raw_features", "op": "select_raw_features", "phase": "fit_predict", "outputs": ["raw_features"]},
            {"id": "fit_mechanism", "op": "fit_mechanism_params", "phase": "fit", "inputs": ["train_X", "train_y"], "outputs": ["mechanism_params"]},
            {"id": "mechanism_features", "op": "compute_mechanism_features", "phase": "fit_predict", "inputs": ["X", "mechanism_params"], "outputs": ["mechanism_features"]},
            {"id": "fit_model", "op": "fit_model", "phase": "fit", "inputs": ["raw_features", "mechanism_features", "train_y"], "outputs": ["model"]},
            {"id": "predict_model", "op": "predict_model", "phase": "predict", "inputs": ["raw_features", "mechanism_features", "model"], "outputs": ["prediction"]},
        ]
    elif mode == "mechanism_residual":
        steps = [
            {"id": "raw_features", "op": "select_raw_features", "phase": "fit_predict", "outputs": ["raw_features"]},
            {"id": "fit_mechanism", "op": "fit_mechanism_params", "phase": "fit", "inputs": ["train_X", "train_y"], "outputs": ["mechanism_params"]},
            {"id": "mechanism_base", "op": "compute_mechanism_base", "phase": "fit_predict", "inputs": ["X", "mechanism_params"], "outputs": ["physics_base"]},
            {"id": "fit_residual", "op": "fit_residual_model", "phase": "fit", "inputs": ["raw_features", "physics_base", "train_y"], "outputs": ["residual_model"]},
            {"id": "predict_residual", "op": "predict_residual_model", "phase": "predict", "inputs": ["raw_features", "residual_model"], "outputs": ["residual_prediction"]},
            {"id": "final_prediction", "op": "add_predictions", "phase": "predict", "inputs": ["physics_base", "residual_prediction"], "outputs": ["prediction"]},
        ]
    elif mode in {"hidden_parameter_physics", "end_to_end_physics_layer"}:
        steps = [
            {"id": "raw_features", "op": "select_raw_features", "phase": "fit_predict", "outputs": ["raw_features"]},
            {"id": "fit_mechanism", "op": "fit_mechanism_params", "phase": "fit", "inputs": ["train_X", "train_y"], "outputs": ["mechanism_params"]},
            {"id": "fit_latent_physics", "op": "fit_latent_physics_model", "phase": "fit", "inputs": ["raw_features", "train_y"], "outputs": ["latent_physics_model"]},
            {"id": "predict_latent_physics", "op": "predict_latent_physics_model", "phase": "predict", "inputs": ["raw_features", "latent_physics_model"], "outputs": ["latent_prediction"]},
            {"id": "physics_output", "op": "compute_physics_output", "phase": "predict", "inputs": ["X", "latent_prediction", "mechanism_params"], "outputs": ["prediction"]},
        ]
    elif mode == "multi_fidelity_base_residual":
        steps = [
            {"id": "raw_features", "op": "select_raw_features", "phase": "fit_predict", "outputs": ["raw_features"]},
            {
                "id": "fit_lf_base",
                "op": "fit_low_fidelity_base",
                "phase": "fit",
                "inputs": ["raw_features", "train_y"],
                "outputs": ["low_fidelity_model"],
                "fidelity_column": "data_fidelity",
                "low_fidelity_label": "low_fidelity",
                "high_fidelity_label": "high_fidelity",
            },
            {
                "id": "fit_hf_residual",
                "op": "fit_high_fidelity_residual",
                "phase": "fit",
                "inputs": ["raw_features", "train_y", "low_fidelity_model"],
                "outputs": ["high_fidelity_residual_model"],
                "fidelity_column": "data_fidelity",
                "high_fidelity_label": "high_fidelity",
            },
            {
                "id": "predict_lf_base",
                "op": "predict_model",
                "phase": "predict",
                "inputs": ["raw_features", "low_fidelity_model"],
                "outputs": ["physics_base"],
            },
            {
                "id": "predict_hf_residual",
                "op": "predict_correction_model",
                "phase": "predict",
                "inputs": ["raw_features", "high_fidelity_residual_model"],
                "outputs": ["residual_prediction"],
            },
            {
                "id": "final_prediction",
                "op": "add_predictions",
                "phase": "predict",
                "inputs": ["physics_base", "residual_prediction"],
                "outputs": ["prediction"],
            },
        ]
    else:
        steps = []
    return normalize_executor_graph({"id": gid, "mode_hint": mode, "steps": steps}, default_id=gid)


def graph_ops(graph: dict[str, Any]) -> list[str]:
    return [str(step.get("op") or "") for step in graph.get("steps", []) if isinstance(step, dict)]


def classify_executor_graph(graph: dict[str, Any]) -> str | None:
    """Classify graphs that match currently implemented interpreter patterns."""
    ops = graph_ops(normalize_executor_graph(graph))
    op_set = set(ops)
    if not ops:
        return None
    if op_set <= {"select_raw_features", "fit_model", "predict_model"} and {"fit_model", "predict_model"} <= op_set:
        return "raw_ml"
    if {"fit_mechanism_params", "compute_mechanism_features", "fit_model", "predict_model"} <= op_set:
        return "mechanism_features"
    if {
        "fit_mechanism_params",
        "compute_mechanism_base",
        "fit_residual_model",
        "predict_residual_model",
        "add_predictions",
    } <= op_set:
        return "mechanism_residual"
    if {"fit_mechanism_params", "fit_latent_physics_model", "predict_latent_physics_model", "compute_physics_output"} <= op_set:
        return "hidden_parameter_physics"
    if {
        "fit_low_fidelity_base",
        "fit_high_fidelity_residual",
        "predict_model",
        "predict_correction_model",
        "add_predictions",
    } <= op_set:
        return "multi_fidelity_base_residual"
    return None


def executor_graph_rejection_reasons(graph: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    normalized = normalize_executor_graph(graph)
    steps = normalized.get("steps", [])
    if not steps:
        reasons.append(f"executor_graph '{normalized.get('id')}' has no steps.")
        return reasons

    seen_ids: set[str] = set()
    fit_seen = False
    predict_seen = False
    for idx, step in enumerate(steps):
        sid = str(step.get("id") or f"s{idx + 1}")
        op = str(step.get("op") or "")
        phase = _slug(step.get("phase") or "")
        if sid in seen_ids:
            reasons.append(f"executor_graph '{normalized.get('id')}' has duplicate step id '{sid}'.")
        seen_ids.add(sid)
        bad_keys = [key for key in FORBIDDEN_GRAPH_KEYS if key in step and step.get(key)]
        if bad_keys:
            reasons.append(f"executor_graph step '{sid}' contains executable-code fields: {bad_keys}.")
        if op not in ALLOWED_GRAPH_OPS:
            reasons.append(f"executor_graph step '{sid}' has unknown op '{op}'.")
            continue
        if phase and phase not in {"fit", "predict", "fit_predict"}:
            reasons.append(f"executor_graph step '{sid}' has unsupported phase '{phase}'.")
        if op in FIT_ONLY_OPS and phase == "predict":
            reasons.append(f"executor_graph step '{sid}' op '{op}' cannot run in predict phase.")
        if op in PREDICT_ONLY_OPS and phase == "fit":
            reasons.append(f"executor_graph step '{sid}' op '{op}' cannot be fit-only.")
        uses_target = bool(step.get("uses_target")) or any(
            str(x).lower() in {"y", "train_y", "target", "yield_stress"}
            for x in _as_list(step.get("inputs"))
        )
        if uses_target and op not in TARGET_USING_OPS:
            reasons.append(f"executor_graph step '{sid}' uses target but op '{op}' is not allowed to consume target.")
        if uses_target and phase == "predict":
            reasons.append(f"executor_graph step '{sid}' uses target in predict phase.")
        if op.startswith("fit_") or phase == "fit":
            fit_seen = True
        if op.startswith("predict_") or op == "add_predictions" or phase == "predict":
            predict_seen = True

    if not fit_seen:
        reasons.append(f"executor_graph '{normalized.get('id')}' has no fit step.")
    if not predict_seen:
        reasons.append(f"executor_graph '{normalized.get('id')}' has no predict step.")
    return reasons


def executor_graph_readiness(graph: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_executor_graph(graph)
    reasons = executor_graph_rejection_reasons(normalized)
    mode = classify_executor_graph(normalized)
    if reasons:
        status = "malformed_or_unsafe"
    elif mode in EXECUTABLE_GRAPH_MODES:
        status = "executable"
    else:
        status = "planned_only"
    return {"status": status, "reasons": reasons, "mode": mode, "graph": normalized}
