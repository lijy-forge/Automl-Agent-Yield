"""Structured fusion-spec handling for yield free-search.

Fusion specs describe *how* a mechanism and a model are combined. They are
structured JSON-like objects, not executable code. This keeps the combination
axis searchable while the harness remains the only execution authority.
"""

from __future__ import annotations

from typing import Any

from knowledge.yield_executor_specs import (
    builtin_executor_spec_for_fusion_type,
    executor_spec_readiness,
    normalize_executor_spec,
)


SUPPORTED_FUSION_TYPES = {
    "raw_ml",
    "mechanism_features",
    "mechanism_residual",
    "hidden_parameter_physics",
    "multi_fidelity_base_residual",
}
PLANNED_FUSION_TYPES = {
    "latent_physics_calibration",
    "physics_loss",
}

COMMON_REQUIRED_FIELDS = ("id", "name", "type", "description")
TYPE_REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    "raw_ml": ("id", "name", "type"),
    "mechanism_features": ("id", "name", "type", "description", "leakage_safety_notes"),
    "mechanism_residual": ("id", "name", "type", "description", "leakage_safety_notes"),
    "latent_physics_calibration": (
        "id",
        "name",
        "type",
        "description",
        "latent",
        "physics_formula",
        "required_columns",
        "fit_scope",
        "prediction_rule",
        "leakage_safety_notes",
    ),
    "physics_loss": (
        "id",
        "name",
        "type",
        "description",
        "physics_formula",
        "required_columns",
        "fit_scope",
        "prediction_rule",
        "leakage_safety_notes",
    ),
    "multi_fidelity_base_residual": (
        "id",
        "name",
        "type",
        "description",
        "required_columns",
        "fit_scope",
        "prediction_rule",
        "leakage_safety_notes",
        "fidelity_column",
        "low_fidelity_label",
        "high_fidelity_label",
    ),
    "hidden_parameter_physics": (
        "id",
        "name",
        "type",
        "description",
        "latent",
        "physics_formula",
        "required_columns",
        "fit_scope",
        "prediction_rule",
        "leakage_safety_notes",
    ),
}

LEAKAGE_SAFE_FIT_SCOPES = {"fold_local", "train_fold_only", "train_only", "full_train_after_cv"}
FORBIDDEN_PREDICTION_TOKENS = ("validation y", "valid y", "test y", "anchor y", "y_valid", "y_test", "target during predict")


def _forbidden_prediction_hits(text: str) -> list[str]:
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

DEFAULT_FUSION_SPECS: tuple[dict[str, Any], ...] = (
    {
        "id": "raw_ml",
        "name": "Raw ML baseline",
        "type": "raw_ml",
        "description": "Train the model on raw numeric features only.",
        "requires_mechanism": False,
        "required_columns": [],
        "leakage_safety_notes": "Uses only schema-approved features; no target-derived features are built.",
        "executor": "builtin",
        "origin": "builtin_fallback",
    },
    {
        "id": "mechanism_features",
        "name": "Mechanism feature augmentation",
        "type": "mechanism_features",
        "description": "Append fold-local mechanism features to raw numeric features, then train the model.",
        "requires_mechanism": True,
        "required_columns": [],
        "leakage_safety_notes": "Mechanism parameters are fit inside each train fold only; validation/test targets are never used to build features.",
        "executor": "builtin",
        "origin": "builtin_fallback",
    },
    {
        "id": "mechanism_residual",
        "name": "Mechanism residual correction",
        "type": "mechanism_residual",
        "description": "Fit a fold-local mechanism base prediction and train the model on the residual.",
        "requires_mechanism": True,
        "required_columns": [],
        "leakage_safety_notes": "Residual targets are computed only inside each train fold; validation/test targets are never used during prediction.",
        "executor": "builtin",
        "origin": "builtin_fallback",
    },
    {
        "id": "end_to_end_m1_eff",
        "name": "End-to-end latent m1_eff physics layer",
        "type": "hidden_parameter_physics",
        "description": (
            "Fit fold-local mechanism parameters, learn a latent m1_eff(X), "
            "then compute yield stress through the physical layer tau=m1_eff*g(X)."
        ),
        "requires_mechanism": True,
        "latent": "m1_eff",
        "physics_formula": "tau = m1_eff(X) * g_mechanism(X; fitted fold-local params, m1=1)",
        "required_columns": ["phi"],
        "fit_scope": "fold_local",
        "prediction_rule": "During prediction, infer m1_eff from X and compute the physics output; no validation/test targets are used.",
        "leakage_safety_notes": "Mechanism parameters and latent model are fit inside each train fold only; no validation/test target values are used.",
        "executor": "builtin",
        "origin": "builtin_fallback",
    },
    {
        "id": "multi_fidelity_base_residual",
        "name": "Low-fidelity base plus high-fidelity residual calibration",
        "type": "multi_fidelity_base_residual",
        "description": (
            "Use low-fidelity rows to fit a base model, then use high-fidelity "
            "train-fold rows to fit a small residual calibration."
        ),
        "requires_mechanism": False,
        "required_columns": ["data_fidelity"],
        "fidelity_column": "data_fidelity",
        "low_fidelity_label": "low_fidelity",
        "high_fidelity_label": "high_fidelity",
        "fit_scope": "fold_local",
        "prediction_rule": (
            "During prediction, compute the low-fidelity base prediction from X "
            "and add the high-fidelity residual calibration; no validation/test targets are used."
        ),
        "leakage_safety_notes": (
            "HF validation targets are never used. Each fold fits the LF base on LF rows "
            "and fits HF residual calibration only on the HF training fold."
        ),
        "executor": "builtin",
        "origin": "builtin_fallback",
    },
)


_ALIASES = {
    "raw": "raw_ml",
    "base": "raw_ml",
    "baseline": "raw_ml",
    "feature": "mechanism_features",
    "features": "mechanism_features",
    "mechanism_feature": "mechanism_features",
    "mechanism_feature_augmentation": "mechanism_features",
    "physics_feature": "mechanism_features",
    "physics_features": "mechanism_features",
    "residual": "mechanism_residual",
    "physics_residual": "mechanism_residual",
    "mechanism_base_residual": "mechanism_residual",
    "residual_correction": "mechanism_residual",
    "latent": "latent_physics_calibration",
    "latent_parameter": "latent_physics_calibration",
    "hidden_parameter": "hidden_parameter_physics",
    "hidden_physics_parameter": "hidden_parameter_physics",
    "hidden_parameter_physics_layer": "hidden_parameter_physics",
    "physics_layer": "hidden_parameter_physics",
    "latent_physics_layer": "hidden_parameter_physics",
    "end_to_end_m1_eff": "hidden_parameter_physics",
    "end_to_end_physics_layer": "hidden_parameter_physics",
    "multi_fidelity": "multi_fidelity_base_residual",
    "multifidelity": "multi_fidelity_base_residual",
    "fidelity_residual": "multi_fidelity_base_residual",
    "lf_hf_residual": "multi_fidelity_base_residual",
}


def slugify_id(value: Any, default: str = "fusion_spec") -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    text = "".join(ch for ch in text if ch.isalnum() or ch == "_").strip("_")
    return text or default


def normalize_fusion_type(value: Any) -> str:
    raw = slugify_id(value, default="mechanism_features")
    return _ALIASES.get(raw, raw)


def normalize_fusion_spec(item: Any, *, default_origin: str = "candidate_agent") -> dict[str, Any]:
    """Return a normalized fusion spec.

    Strings remain supported for backward compatibility and are converted into a
    built-in-style spec. Dict specs can be proposed by CandidateAgent/LLM, but
    only known ``type`` values are executable today.
    """

    if isinstance(item, str):
        fusion_type = normalize_fusion_type(item)
        spec_id = "end_to_end_m1_eff" if slugify_id(item) == "end_to_end_m1_eff" else fusion_type
        if fusion_type in SUPPORTED_FUSION_TYPES:
            for default in DEFAULT_FUSION_SPECS:
                if str(default.get("type") or "") != fusion_type:
                    continue
                spec = dict(default)
                if fusion_type == "hidden_parameter_physics" and spec_id != "end_to_end_m1_eff":
                    spec["id"] = spec_id
                    spec["name"] = str(item)
                spec["origin"] = "legacy_string"
                return normalize_fusion_spec(spec, default_origin="legacy_string")
        spec = {
            "id": spec_id,
            "name": fusion_type,
            "type": fusion_type,
            "description": "",
            "requires_mechanism": fusion_type != "raw_ml",
            "required_columns": [],
            "executor": "builtin" if fusion_type in SUPPORTED_FUSION_TYPES else "unsupported",
            "executor_spec": (
                builtin_executor_spec_for_fusion_type(fusion_type, fusion_id=spec_id)
                if fusion_type in SUPPORTED_FUSION_TYPES else None
            ),
            "origin": "legacy_string",
        }
        if fusion_type == "hidden_parameter_physics":
            spec.update(
                {
                    "id": spec_id,
                    "name": "End-to-end latent m1_eff physics layer",
                    "description": (
                        "Fit fold-local mechanism parameters, learn a latent m1_eff(X), "
                        "then compute yield stress through the physical layer tau=m1_eff*g(X)."
                    ),
                    "latent": "m1_eff",
                    "physics_formula": "tau = m1_eff(X) * g_mechanism(X; fitted fold-local params, m1=1)",
                    "required_columns": ["phi"],
                    "fit_scope": "fold_local",
                    "prediction_rule": "During prediction, infer m1_eff from X and compute the physics output; no validation/test targets are used.",
                    "leakage_safety_notes": "Mechanism parameters and latent model are fit inside each train fold only; no validation/test target values are used.",
                }
            )
        return spec

    spec = dict(item or {}) if isinstance(item, dict) else {}
    fusion_type = normalize_fusion_type(
        spec.get("type")
        or spec.get("fusion_type")
        or spec.get("mode")
        or spec.get("fusion_mode")
        or spec.get("id")
    )
    spec_id = slugify_id(spec.get("id") or fusion_type, default=fusion_type)
    requires = spec.get("requires_mechanism")
    if requires is None:
        requires = fusion_type != "raw_ml"
    required_columns = spec.get("required_columns") or []
    if isinstance(required_columns, str):
        required_columns = [required_columns]
    latent = spec.get("latent") or spec.get("latent_variable") or spec.get("hidden_parameter")
    physics_formula = spec.get("physics_formula") or spec.get("formula") or spec.get("output_formula")
    fit_scope = spec.get("fit_scope") or spec.get("parameter_fit_scope") or spec.get("training_scope")
    prediction_rule = spec.get("prediction_rule") or spec.get("predict_rule") or spec.get("inference_rule")
    leakage_notes = spec.get("leakage_safety_notes") or spec.get("leakage_safety") or spec.get("anti_leakage")
    executor_spec = spec.get("executor_spec") or spec.get("executor_plan")
    if executor_spec:
        executor_spec = normalize_executor_spec(
            executor_spec,
            default_fusion_id=spec_id,
            default_fusion_type=fusion_type,
        )
    elif fusion_type in SUPPORTED_FUSION_TYPES:
        executor_spec = builtin_executor_spec_for_fusion_type(fusion_type, fusion_id=spec_id)
    normalized = {
        **spec,
        "id": spec_id,
        "name": str(spec.get("name") or spec_id),
        "type": fusion_type,
        "description": str(spec.get("description") or spec.get("combination_method") or ""),
        "latent": str(latent or "") if latent is not None else "",
        "physics_formula": str(physics_formula or "") if physics_formula is not None else "",
        "fit_scope": str(fit_scope or "") if fit_scope is not None else "",
        "prediction_rule": str(prediction_rule or "") if prediction_rule is not None else "",
        "leakage_safety_notes": str(leakage_notes or "") if leakage_notes is not None else "",
        "requires_mechanism": bool(requires),
        "required_columns": [str(c) for c in required_columns],
        "executor": "builtin" if fusion_type in SUPPORTED_FUSION_TYPES else "unsupported",
        "executor_spec": executor_spec,
        "origin": str(spec.get("origin") or default_origin),
    }
    return normalized


def default_fusion_specs() -> list[dict[str, Any]]:
    return [
        normalize_fusion_spec(dict(spec), default_origin=str(spec.get("origin") or "builtin_fallback"))
        for spec in DEFAULT_FUSION_SPECS
    ]


def normalize_fusion_specs(items: Any | None) -> list[dict[str, Any]]:
    if not items:
        return default_fusion_specs()
    if isinstance(items, (str, dict)):
        items = [items]
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items or []:
        spec = normalize_fusion_spec(item)
        key = str(spec["id"])
        if key in seen:
            continue
        seen.add(key)
        out.append(spec)
    return out or default_fusion_specs()


def merge_with_default_fusion_specs(items: Any | None) -> list[dict[str, Any]]:
    """Normalize searched fusion specs and append missing built-in floor specs.

    This mirrors the model-family floor (rf/hgb/pinn/A-B-C-D): an LLM round may
    propose useful new fusion ideas, but it should not accidentally remove the
    executable baseline routes from the comparison.
    """
    specs = normalize_fusion_specs(items)
    seen = {str(spec.get("id") or "") for spec in specs}
    for spec in default_fusion_specs():
        key = str(spec.get("id") or "")
        if key not in seen:
            specs.append(spec)
            seen.add(key)
    return specs


def fusion_spec_rejection_reasons(spec: dict[str, Any], columns: list[str] | tuple[str, ...]) -> list[str]:
    reasons: list[str] = []
    fusion_type = str(spec.get("type") or "")
    required = TYPE_REQUIRED_FIELDS.get(fusion_type, COMMON_REQUIRED_FIELDS)
    missing_fields = [
        field for field in required
        if field not in spec
        or spec.get(field) in (None, "")
        or (field == "required_columns" and not spec.get("required_columns"))
    ]
    if missing_fields:
        reasons.append(f"fusion_spec '{spec.get('id')}' is missing required fields for type '{fusion_type}': {missing_fields}.")

    if fusion_type in {"mechanism_features", "mechanism_residual", "latent_physics_calibration", "physics_loss", "hidden_parameter_physics", "multi_fidelity_base_residual"}:
        notes = str(spec.get("leakage_safety_notes") or "").lower()
        if not notes:
            reasons.append(f"fusion_spec '{spec.get('id')}' must state leakage_safety_notes.")
        elif not any(token in notes for token in ("fold", "train", "no validation", "no test", "no anchor", "no target leakage")):
            reasons.append(
                f"fusion_spec '{spec.get('id')}' leakage_safety_notes must mention fold-local/train-only fitting or no validation/test target use."
            )

    if fusion_type in {"latent_physics_calibration", "physics_loss", "hidden_parameter_physics", "multi_fidelity_base_residual"}:
        fit_scope = str(spec.get("fit_scope") or "").strip().lower()
        if fit_scope and fit_scope not in LEAKAGE_SAFE_FIT_SCOPES:
            reasons.append(
                f"fusion_spec '{spec.get('id')}' has unsafe/unclear fit_scope '{spec.get('fit_scope')}'. "
                f"Expected one of {sorted(LEAKAGE_SAFE_FIT_SCOPES)}."
            )
        pred = str(spec.get("prediction_rule") or "").strip().lower()
        if pred:
            bad = _forbidden_prediction_hits(pred)
            if bad:
                reasons.append(
                    f"fusion_spec '{spec.get('id')}' prediction_rule appears to use held-out targets: {bad}."
                )

    executor_spec = spec.get("executor_spec")
    if executor_spec:
        executor_audit = executor_spec_readiness(
            executor_spec,
            columns,
            fusion_type=fusion_type,
            fusion_id=str(spec.get("id") or ""),
        )
        reasons.extend(f"executor_spec: {reason}" for reason in executor_audit["reasons"])
    elif fusion_type in SUPPORTED_FUSION_TYPES:
        reasons.append(f"fusion_spec '{spec.get('id')}' is executable only with a declared builtin executor_spec.")

    if fusion_type not in SUPPORTED_FUSION_TYPES:
        if fusion_type in PLANNED_FUSION_TYPES:
            if not executor_spec:
                reasons.append(
                    f"fusion_spec '{spec.get('id')}' has no executor_spec for planned type '{fusion_type}'."
                )
        else:
            reasons.append(
                f"fusion_spec '{spec.get('id')}' has unknown type '{fusion_type}'. Supported executable types are "
                f"{sorted(SUPPORTED_FUSION_TYPES)}; planned-only known types are {sorted(PLANNED_FUSION_TYPES)}."
            )
    cols = {str(c) for c in columns}
    missing = [str(c) for c in spec.get("required_columns", []) if str(c) not in cols]
    if missing:
        reasons.append(f"fusion_spec '{spec.get('id')}' requires missing columns: {missing}.")
    return reasons


def fusion_spec_readiness(spec: dict[str, Any], columns: list[str] | tuple[str, ...]) -> dict[str, Any]:
    """Audit whether a normalized spec may be executed by the current harness."""
    reasons = fusion_spec_rejection_reasons(spec, columns)
    fusion_type = str(spec.get("type") or "")
    executor_status = None
    executor_reasons: list[str] = []
    if spec.get("executor_spec"):
        executor_audit = executor_spec_readiness(
            spec["executor_spec"],
            columns,
            fusion_type=fusion_type,
            fusion_id=str(spec.get("id") or ""),
        )
        executor_status = executor_audit["status"]
        executor_reasons = executor_audit["reasons"]
    if not reasons:
        status = "executable"
    elif any(
        "missing required fields" in r
        or "unsafe" in r
        or "held-out targets" in r
        or "leakage" in r
        or "must be declarative" in r
        or "does not match" in r
        or "targets fusion_id" in r
        for r in reasons
    ):
        status = "malformed_or_unsafe"
    elif any("missing columns" in r or "requires missing columns" in r for r in reasons):
        status = "missing_data"
    elif fusion_type in PLANNED_FUSION_TYPES and any(("no implemented" in r or "no executor_spec" in r) for r in reasons):
        status = "planned_only"
    else:
        status = "not_executable"
    return {
        "status": status,
        "reasons": reasons,
        "executor_status": executor_status,
        "executor_reasons": executor_reasons,
    }


def builtin_fusion_mode(spec: dict[str, Any]) -> str:
    """Return the executable built-in mode for a normalized spec."""
    return normalize_fusion_type(spec.get("type"))
