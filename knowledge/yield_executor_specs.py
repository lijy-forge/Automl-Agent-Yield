"""Structured executor-spec handling for yield free-search.

Executor specs describe *how the harness would execute* a fusion spec. They are
declarative JSON-like plans, not executable Python. This gives CandidateAgent a
place to propose new execution ideas while the fixed harness keeps authority
over what can actually run.
"""

from __future__ import annotations

from typing import Any

from knowledge.yield_executor_graphs import (
    builtin_executor_graph,
    executor_graph_readiness,
    normalize_executor_graph,
)


SUPPORTED_EXECUTOR_TYPES = {
    "builtin_raw_ml",
    "builtin_mechanism_features",
    "builtin_mechanism_residual",
    "hidden_parameter_physics_layer",
    "multi_fidelity_base_residual",
}

PLANNED_EXECUTOR_TYPES = {
    "latent_parameter_regressor",
    "physics_loss_regularizer",
}

FUSION_TO_BUILTIN_EXECUTOR = {
    "raw_ml": "builtin_raw_ml",
    "mechanism_features": "builtin_mechanism_features",
    "mechanism_residual": "builtin_mechanism_residual",
    "hidden_parameter_physics": "hidden_parameter_physics_layer",
    "multi_fidelity_base_residual": "multi_fidelity_base_residual",
}

BUILTIN_EXECUTOR_TO_FUSION = {
    value: key for key, value in FUSION_TO_BUILTIN_EXECUTOR.items()
}

LEAKAGE_SAFE_FIT_SCOPES = {"fold_local", "train_fold_only", "train_only", "full_train_after_cv"}
FORBIDDEN_PREDICTION_TOKENS = (
    "validation y",
    "valid y",
    "test y",
    "anchor y",
    "y_valid",
    "y_test",
    "target during predict",
)
FORBIDDEN_CODE_FIELDS = (
    "code",
    "source_code",
    "python_code",
    "script",
    "exec_code",
    "callable",
)


def _forbidden_prediction_hits(text: str) -> list[str]:
    """Return held-out-target tokens unless they are clearly negated."""
    pred = str(text or "").strip().lower()
    hits: list[str] = []
    for token in FORBIDDEN_PREDICTION_TOKENS:
        if token not in pred:
            continue
        negated = any(
            phrase in pred
            for phrase in (
                f"no {token}",
                f"never use {token}",
                f"never uses {token}",
                f"without {token}",
            )
        )
        if not negated:
            hits.append(token)
    return hits

COMMON_REQUIRED_FIELDS = (
    "id",
    "fusion_id",
    "executor_type",
    "description",
    "executor_graph",
    "fit_scope",
    "prediction_rule",
    "leakage_safety_notes",
)

TYPE_REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    "builtin_raw_ml": COMMON_REQUIRED_FIELDS,
    "builtin_mechanism_features": COMMON_REQUIRED_FIELDS,
    "builtin_mechanism_residual": COMMON_REQUIRED_FIELDS,
    "latent_parameter_regressor": (
        "id",
        "fusion_id",
        "executor_type",
        "description",
        "latent",
        "target_rule",
        "physics_formula",
        "required_columns",
        "fit_scope",
        "prediction_rule",
        "leakage_safety_notes",
    ),
    "hidden_parameter_physics_layer": (
        "id",
        "fusion_id",
        "executor_type",
        "description",
        "latent",
        "decoder_rule",
        "physics_formula",
        "required_columns",
        "fit_scope",
        "prediction_rule",
        "leakage_safety_notes",
    ),
    "physics_loss_regularizer": (
        "id",
        "fusion_id",
        "executor_type",
        "description",
        "physics_formula",
        "loss_terms",
        "required_columns",
        "fit_scope",
        "prediction_rule",
        "leakage_safety_notes",
    ),
    "multi_fidelity_base_residual": (
        "id",
        "fusion_id",
        "executor_type",
        "description",
        "executor_graph",
        "fidelity_column",
        "low_fidelity_label",
        "high_fidelity_label",
        "required_columns",
        "fit_scope",
        "prediction_rule",
        "leakage_safety_notes",
    ),
}

_ALIASES = {
    "builtin": "builtin_auto",
    "raw": "builtin_raw_ml",
    "raw_ml": "builtin_raw_ml",
    "feature": "builtin_mechanism_features",
    "features": "builtin_mechanism_features",
    "mechanism_features": "builtin_mechanism_features",
    "residual": "builtin_mechanism_residual",
    "mechanism_residual": "builtin_mechanism_residual",
    "latent": "latent_parameter_regressor",
    "latent_parameter": "latent_parameter_regressor",
    "hidden_parameter": "hidden_parameter_physics_layer",
    "physics_layer": "hidden_parameter_physics_layer",
    "physics_loss": "physics_loss_regularizer",
    "pinn_loss": "physics_loss_regularizer",
    "multi_fidelity": "multi_fidelity_base_residual",
    "fidelity_residual": "multi_fidelity_base_residual",
}


def slugify_id(value: Any, default: str = "executor_spec") -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    text = "".join(ch for ch in text if ch.isalnum() or ch == "_").strip("_")
    return text or default


def normalize_executor_type(value: Any, *, fusion_type: str | None = None) -> str:
    raw = slugify_id(value, default="builtin_auto")
    mapped = _ALIASES.get(raw, raw)
    if mapped == "builtin_auto" and fusion_type:
        return FUSION_TO_BUILTIN_EXECUTOR.get(slugify_id(fusion_type), mapped)
    return mapped


def builtin_executor_spec_for_fusion_type(
    fusion_type: str,
    *,
    fusion_id: str | None = None,
) -> dict[str, Any]:
    """Return the declarative executor spec for a currently implemented fusion."""
    fusion = slugify_id(fusion_type)
    executor_type = FUSION_TO_BUILTIN_EXECUTOR.get(fusion, "unsupported")
    fid = slugify_id(fusion_id or fusion, default=fusion)
    descriptions = {
        "raw_ml": "Use schema-approved raw numeric features and fit the model directly.",
        "mechanism_features": "Fit mechanism parameters inside each train fold, append mechanism features, then fit the model.",
        "mechanism_residual": "Fit mechanism parameters inside each train fold, predict a physics base, and fit the model on the train-fold residual.",
        "hidden_parameter_physics": "Fit mechanism parameters inside each train fold, learn latent m1_eff(X), then compute yield stress through a physics layer.",
        "multi_fidelity_base_residual": "Fit a low-fidelity base model on LF rows, then fit a small high-fidelity residual calibration on HF rows.",
    }
    prediction_rules = {
        "raw_ml": "During prediction, transform raw features with the fitted preprocessing/model only; no target values are used.",
        "mechanism_features": "During prediction, compute mechanism features from X and train-fold fitted mechanism parameters; no validation/test targets are used.",
        "mechanism_residual": "During prediction, compute physics base from X and train-fold fitted mechanism parameters, then add the model residual; no validation/test targets are used.",
        "hidden_parameter_physics": "During prediction, infer latent m1_eff from X and compute tau=m1_eff*g(X); no validation/test targets are used.",
        "multi_fidelity_base_residual": "During prediction, compute the LF-base prediction from X and add the HF residual calibration; no validation/test targets are used.",
    }
    spec = {
        "id": f"{fid}_executor",
        "fusion_id": fid,
        "executor_type": executor_type,
        "description": descriptions.get(fusion, "Unsupported fusion executor."),
        "executor_graph": builtin_executor_graph(fusion, graph_id=f"{fid}_graph"),
        "required_columns": [],
        "fit_scope": "fold_local",
        "prediction_rule": prediction_rules.get(fusion, ""),
        "leakage_safety_notes": "All fitted quantities are learned inside each training fold only; validation/test targets are never used.",
        "implementation_authority": "fixed_harness",
        "origin": "builtin",
    }
    if fusion == "hidden_parameter_physics":
        spec.update(
            {
                "latent": "m1_eff",
                "decoder_rule": "m1_eff is constrained positive before multiplying the mechanism unit response.",
                "physics_formula": "tau = m1_eff(X) * g_mechanism(X; fitted fold-local params, m1=1)",
                "required_columns": ["phi"],
            }
        )
    if fusion == "multi_fidelity_base_residual":
        spec.update(
            {
                "fidelity_column": "data_fidelity",
                "low_fidelity_label": "low_fidelity",
                "high_fidelity_label": "high_fidelity",
                "required_columns": ["data_fidelity"],
            }
        )
    return spec


def default_executor_specs() -> list[dict[str, Any]]:
    return [
        builtin_executor_spec_for_fusion_type("raw_ml"),
        builtin_executor_spec_for_fusion_type("mechanism_features"),
        builtin_executor_spec_for_fusion_type("mechanism_residual"),
        builtin_executor_spec_for_fusion_type("hidden_parameter_physics"),
        builtin_executor_spec_for_fusion_type("multi_fidelity_base_residual"),
    ]


def normalize_executor_spec(
    item: Any,
    *,
    default_fusion_id: str | None = None,
    default_fusion_type: str | None = None,
    default_origin: str = "candidate_agent",
) -> dict[str, Any]:
    """Return a normalized declarative executor spec.

    This function deliberately preserves unknown/planned executor types for the
    judge to classify. It never turns arbitrary code into an executable object.
    """
    if isinstance(item, str):
        executor_type = normalize_executor_type(item, fusion_type=default_fusion_type)
        fusion_id = slugify_id(default_fusion_id or BUILTIN_EXECUTOR_TO_FUSION.get(executor_type, "fusion"))
        return {
            "id": f"{fusion_id}_executor",
            "fusion_id": fusion_id,
            "executor_type": executor_type,
            "description": "",
            "executor_graph": (
                builtin_executor_graph(BUILTIN_EXECUTOR_TO_FUSION.get(executor_type, ""))
                if executor_type in SUPPORTED_EXECUTOR_TYPES else None
            ),
            "required_columns": [],
            "fit_scope": "fold_local",
            "prediction_rule": "",
            "leakage_safety_notes": "",
            "implementation_authority": "fixed_harness" if executor_type in SUPPORTED_EXECUTOR_TYPES else "planned",
            "origin": "legacy_string",
        }

    spec = dict(item or {}) if isinstance(item, dict) else {}
    fusion_id = slugify_id(
        spec.get("fusion_id")
        or spec.get("fusion")
        or spec.get("fusion_spec_id")
        or default_fusion_id
        or "fusion"
    )
    executor_type = normalize_executor_type(
        spec.get("executor_type")
        or spec.get("type")
        or spec.get("executor")
        or spec.get("id")
        or "builtin_auto",
        fusion_type=default_fusion_type,
    )
    required_columns = spec.get("required_columns") or []
    if isinstance(required_columns, str):
        required_columns = [required_columns]
    executor_graph = spec.get("executor_graph") or spec.get("graph") or spec.get("execution_graph")
    if executor_graph:
        executor_graph = normalize_executor_graph(
            executor_graph,
            default_id=f"{fusion_id}_{executor_type}_graph",
        )
    elif executor_type in SUPPORTED_EXECUTOR_TYPES:
        executor_graph = builtin_executor_graph(
            BUILTIN_EXECUTOR_TO_FUSION.get(executor_type, ""),
            graph_id=f"{fusion_id}_graph",
        )
    normalized = {
        **spec,
        "id": slugify_id(spec.get("id") or f"{fusion_id}_{executor_type}", default=f"{fusion_id}_executor"),
        "fusion_id": fusion_id,
        "executor_type": executor_type,
        "description": str(spec.get("description") or spec.get("execution_plan") or ""),
        "executor_graph": executor_graph,
        "required_columns": [str(c) for c in required_columns],
        "fit_scope": str(spec.get("fit_scope") or spec.get("training_scope") or ""),
        "prediction_rule": str(spec.get("prediction_rule") or spec.get("predict_rule") or spec.get("inference_rule") or ""),
        "leakage_safety_notes": str(spec.get("leakage_safety_notes") or spec.get("leakage_safety") or spec.get("anti_leakage") or ""),
        "implementation_authority": str(
            spec.get("implementation_authority")
            or ("fixed_harness" if executor_type in SUPPORTED_EXECUTOR_TYPES else "planned")
        ),
        "origin": str(spec.get("origin") or default_origin),
    }
    for key in (
        "latent",
        "target_rule",
        "physics_formula",
        "decoder_rule",
        "loss_terms",
        "fidelity_column",
        "low_fidelity_label",
        "high_fidelity_label",
        "bounds",
    ):
        if key in spec:
            normalized[key] = spec.get(key)
    return normalized


def normalize_executor_specs(items: Any | None, *, include_defaults: bool = False) -> list[dict[str, Any]]:
    if not items:
        return default_executor_specs() if include_defaults else []
    if isinstance(items, (str, dict)):
        items = [items]
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items or []:
        spec = normalize_executor_spec(item)
        key = str(spec["id"])
        if key in seen:
            continue
        seen.add(key)
        out.append(spec)
    return out


def attach_executor_specs_to_fusion_specs(
    fusion_specs: list[dict[str, Any]],
    executor_specs: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Attach separately proposed executor specs to matching fusion specs."""
    by_fusion: dict[str, dict[str, Any]] = {}
    for executor in executor_specs or []:
        if not isinstance(executor, dict):
            continue
        by_fusion.setdefault(str(executor.get("fusion_id") or ""), executor)

    out: list[dict[str, Any]] = []
    for fusion in fusion_specs:
        spec = dict(fusion)
        fid = str(spec.get("id") or "")
        if not spec.get("executor_spec") and fid in by_fusion:
            spec["executor_spec"] = by_fusion[fid]
        out.append(spec)
    return out


def executor_spec_rejection_reasons(
    spec: dict[str, Any],
    columns: list[str] | tuple[str, ...],
    *,
    fusion_type: str | None = None,
    fusion_id: str | None = None,
) -> list[str]:
    reasons: list[str] = []
    executor_type = str(spec.get("executor_type") or "")
    required = TYPE_REQUIRED_FIELDS.get(executor_type, COMMON_REQUIRED_FIELDS)
    missing_fields = [
        field for field in required
        if field not in spec
        or spec.get(field) in (None, "")
        or (field == "required_columns" and executor_type in PLANNED_EXECUTOR_TYPES and not spec.get(field))
    ]
    if missing_fields:
        reasons.append(
            f"executor_spec '{spec.get('id')}' is missing required fields for executor_type '{executor_type}': {missing_fields}."
        )

    code_fields = [field for field in FORBIDDEN_CODE_FIELDS if field in spec and spec.get(field)]
    if code_fields:
        reasons.append(
            f"executor_spec '{spec.get('id')}' must be declarative; executable code fields are not allowed: {code_fields}."
        )

    graph = spec.get("executor_graph")
    graph_status = None
    graph_mode = None
    if graph:
        graph_audit = executor_graph_readiness(graph)
        graph_status = graph_audit["status"]
        graph_mode = graph_audit["mode"]
        reasons.extend(f"executor_graph: {reason}" for reason in graph_audit["reasons"])
    else:
        reasons.append(f"executor_spec '{spec.get('id')}' must include executor_graph steps.")

    expected_executor = FUSION_TO_BUILTIN_EXECUTOR.get(slugify_id(fusion_type)) if fusion_type else None
    if executor_type in SUPPORTED_EXECUTOR_TYPES and expected_executor and executor_type != expected_executor:
        reasons.append(
            f"executor_spec '{spec.get('id')}' executor_type '{executor_type}' does not match fusion_type '{fusion_type}'."
        )
    expected_mode = BUILTIN_EXECUTOR_TO_FUSION.get(executor_type)
    if executor_type in SUPPORTED_EXECUTOR_TYPES and graph_mode and expected_mode and graph_mode != expected_mode:
        reasons.append(
            f"executor_spec '{spec.get('id')}' executor_graph mode '{graph_mode}' does not match executor_type '{executor_type}'."
        )
    if executor_type in SUPPORTED_EXECUTOR_TYPES and graph_status == "planned_only":
        reasons.append(
            f"executor_spec '{spec.get('id')}' uses a graph pattern the current harness cannot interpret as '{expected_mode}'."
        )

    if fusion_id and str(spec.get("fusion_id") or "") not in {"", str(fusion_id)}:
        reasons.append(
            f"executor_spec '{spec.get('id')}' targets fusion_id '{spec.get('fusion_id')}', not '{fusion_id}'."
        )

    notes = str(spec.get("leakage_safety_notes") or "").lower()
    if not notes:
        reasons.append(f"executor_spec '{spec.get('id')}' must state leakage_safety_notes.")
    elif not any(token in notes for token in ("fold", "train", "no validation", "no test", "no anchor", "no target leakage")):
        reasons.append(
            f"executor_spec '{spec.get('id')}' leakage_safety_notes must mention fold-local/train-only fitting or no validation/test target use."
        )

    fit_scope = str(spec.get("fit_scope") or "").strip().lower()
    if fit_scope and fit_scope not in LEAKAGE_SAFE_FIT_SCOPES:
        reasons.append(
            f"executor_spec '{spec.get('id')}' has unsafe/unclear fit_scope '{spec.get('fit_scope')}'. "
            f"Expected one of {sorted(LEAKAGE_SAFE_FIT_SCOPES)}."
        )

    pred = str(spec.get("prediction_rule") or "").strip().lower()
    if pred:
        bad = _forbidden_prediction_hits(pred)
        if bad:
            reasons.append(
                f"executor_spec '{spec.get('id')}' prediction_rule appears to use held-out targets: {bad}."
            )

    if executor_type not in SUPPORTED_EXECUTOR_TYPES:
        if executor_type in PLANNED_EXECUTOR_TYPES:
            reasons.append(
                f"executor_spec '{spec.get('id')}' has no implemented interpreter for planned executor_type '{executor_type}' yet."
            )
        else:
            reasons.append(
                f"executor_spec '{spec.get('id')}' has unknown executor_type '{executor_type}'. Supported executable executor types are "
                f"{sorted(SUPPORTED_EXECUTOR_TYPES)}; planned-only known types are {sorted(PLANNED_EXECUTOR_TYPES)}."
            )

    cols = {str(c) for c in columns}
    missing = [str(c) for c in spec.get("required_columns", []) if str(c) not in cols]
    if missing:
        reasons.append(f"executor_spec '{spec.get('id')}' requires missing columns: {missing}.")
    return reasons


def executor_spec_readiness(
    spec: dict[str, Any],
    columns: list[str] | tuple[str, ...],
    *,
    fusion_type: str | None = None,
    fusion_id: str | None = None,
) -> dict[str, Any]:
    """Audit whether an executor spec may be interpreted by the current harness."""
    reasons = executor_spec_rejection_reasons(spec, columns, fusion_type=fusion_type, fusion_id=fusion_id)
    executor_type = str(spec.get("executor_type") or "")
    graph_audit = executor_graph_readiness(spec["executor_graph"]) if spec.get("executor_graph") else {
        "status": None,
        "mode": None,
        "reasons": [],
    }
    if not reasons:
        status = "executable"
    elif any(
        token in r
        for r in reasons
        for token in (
            "missing required fields",
            "unsafe",
            "held-out targets",
            "leakage",
            "must be declarative",
            "does not match",
            "targets fusion_id",
        )
    ):
        status = "malformed_or_unsafe"
    elif any("missing columns" in r or "requires missing columns" in r for r in reasons):
        status = "missing_data"
    elif executor_type in PLANNED_EXECUTOR_TYPES and any("no implemented interpreter" in r for r in reasons):
        status = "planned_only"
    else:
        status = "not_executable"
    return {
        "status": status,
        "reasons": reasons,
        "graph_status": graph_audit.get("status"),
        "graph_mode": graph_audit.get("mode"),
        "graph_reasons": graph_audit.get("reasons", []),
    }
