"""Layer 4 — joint search + evaluation for direction-④.

A "candidate" is a thin estimator (anything with fold-local `fit(X_df, y)` /
`predict(X_df)`) that internally combines a mechanism and a model however it
likes. The candidates are the DYNAMIC MENU (generated per run — offline from the
seed combination templates, later written by the LLM from external search). This
module is the FIXED evaluation hygiene layer:

  * evaluate every candidate the SAME way — fold-local OOF + optional anchor,
  * anti-leakage by construction (fit sees train X,y; predict sees only X),
  * rank, pick the champion, and (finalize) refit the champion on full data.

It imports the existing metric/data helpers so scoring matches the rest of the
pipeline.
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from knowledge.yield_candidate_benchmark import (
    _evaluate_baseline_summary,
    _feature_columns,
    _metrics,
    _safe_float,
)
from knowledge.yield_fusion_specs import (
    builtin_fusion_mode,
    fusion_spec_readiness,
    normalize_fusion_spec,
    normalize_fusion_specs,
)
from knowledge.yield_executor_graphs import (
    builtin_executor_graph,
    executor_graph_readiness,
    normalize_executor_graph,
)
from knowledge.yield_schema import TARGET_COLUMN, load_yield_dataframe


_BUDGET_PRESETS: dict[str, dict[str, int]] = {
    # Smoke/dev: prove the graph, artifacts, and predict.py work.
    "quick": {
        "max_candidates": 60,
        "max_hparams_per_model": 1,
        "max_iter": 120,
        "n_estimators": 80,
    },
    # Day-to-day search: keep all axes represented but avoid duplicate explosion.
    "normal": {
        "max_candidates": 120,
        "max_hparams_per_model": 1,
        "max_iter": 220,
        "n_estimators": 180,
    },
    # Confirmation run: broad enough for reporting, still bounded.
    "full": {
        "max_candidates": 450,
        "max_hparams_per_model": 6,
        "max_iter": 600,
        "n_estimators": 500,
    },
    # Explicit escape hatch for one-off exhaustive debugging.
    "unlimited": {
        "max_candidates": 0,
        "max_hparams_per_model": 0,
        "max_iter": 0,
        "n_estimators": 0,
    },
}


def _env_int(name: str, default: int) -> int:
    try:
        raw = os.environ.get(name)
        return int(raw) if raw not in (None, "") else int(default)
    except Exception:
        return int(default)


def search_budget_config(mode: str | None = None) -> dict[str, Any]:
    """Runtime budget for candidate count and expensive estimator settings.

    The presets are workflow modes, not scientific claims:
    quick checks the pipeline, normal is the default daily search, and full is
    for confirmation. Env vars can override every cap.
    """
    raw_mode = str(mode or os.environ.get("YIELD_SEARCH_BUDGET") or "normal").strip().lower()
    if raw_mode not in _BUDGET_PRESETS:
        raw_mode = "normal"
    cfg = dict(_BUDGET_PRESETS[raw_mode])
    cfg["mode"] = raw_mode
    cfg["max_candidates"] = _env_int("YIELD_MAX_SEARCH_CANDIDATES", cfg["max_candidates"])
    cfg["max_hparams_per_model"] = _env_int("YIELD_MAX_HPARAMS_PER_MODEL", cfg["max_hparams_per_model"])
    cfg["max_iter"] = _env_int("YIELD_MAX_ESTIMATOR_ITER", cfg["max_iter"])
    cfg["n_estimators"] = _env_int("YIELD_MAX_ESTIMATOR_TREES", cfg["n_estimators"])
    return cfg


def _cap_training_value(kind: str, value: Any) -> Any:
    try:
        ivalue = int(value)
    except Exception:
        return value
    cap = int(search_budget_config().get(kind) or 0)
    return min(ivalue, cap) if cap > 0 else ivalue


def _cap_estimator_complexity(est):
    """Clamp common slow estimator params, including nested Pipeline params."""
    if not hasattr(est, "get_params") or not hasattr(est, "set_params"):
        return est
    cfg = search_budget_config()
    caps = {
        "max_iter": int(cfg.get("max_iter") or 0),
        "n_estimators": int(cfg.get("n_estimators") or 0),
    }
    updates: dict[str, Any] = {}
    try:
        params = est.get_params(deep=True)
    except Exception:
        return est
    for key, value in params.items():
        tail = str(key).split("__")[-1]
        cap = caps.get(tail, 0)
        if cap <= 0:
            continue
        if isinstance(value, bool):
            continue
        try:
            if int(value) > cap:
                updates[key] = cap
        except Exception:
            continue
    if updates:
        try:
            est.set_params(**updates)
        except Exception:
            pass
    return est


def _fusion_execution_key(spec: dict[str, Any]) -> tuple[Any, ...]:
    """Key for fusion specs that are execution-equivalent in the current harness."""
    mode = builtin_fusion_mode(spec)
    graph = ((spec.get("executor_spec") or {}).get("executor_graph")
             or builtin_executor_graph(mode, graph_id=f"{mode}_graph"))
    norm = normalize_executor_graph(graph)
    steps = tuple(
        (
            str(step.get("op") or ""),
            str(step.get("phase") or ""),
            tuple(str(x) for x in (step.get("inputs") or [])),
            tuple(str(x) for x in (step.get("outputs") or [])),
        )
        for step in norm.get("steps", [])
    )
    return (mode, steps)


def _fusion_readiness_rank(status: str) -> int:
    return {
        "executable": 0,
        "planned_only": 1,
        "missing_data": 2,
        "not_executable": 3,
        "malformed_or_unsafe": 4,
    }.get(str(status or ""), 9)


def _dedupe_fusion_specs_for_execution(
    specs: list[dict[str, Any]],
    columns: list[str] | tuple[str, ...],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Drop execution-equivalent fusion specs, preferring the safest runnable spec.

    CandidateAgent may propose a fusion spec with the same executor graph as a
    built-in fallback but with incomplete safety notes. A first-seen-only de-dupe
    would keep the malformed spec and drop the safe built-in one, effectively
    deleting an entire fusion route. This function ranks duplicates by readiness
    first, so executable fallbacks remain available.
    """
    kept: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    seen: dict[tuple[Any, ...], dict[str, Any]] = {}
    for spec in specs:
        key = _fusion_execution_key(spec)
        spec_id = str(spec.get("id") or "")
        audit = fusion_spec_readiness(spec, columns)
        rank = _fusion_readiness_rank(str(audit.get("status") or ""))
        if key in seen:
            previous = seen[key]
            previous_id = str(previous.get("id") or "")
            previous_rank = int(previous.get("rank", 9))
            if rank < previous_rank:
                kept[int(previous["index"])] = spec
                seen[key] = {
                    "id": spec_id,
                    "index": previous["index"],
                    "rank": rank,
                    "audit": audit,
                    "status": audit.get("status"),
                    "type": str(spec.get("type") or ""),
                }
                skipped.append({
                    "id": previous_id,
                    "type": str(previous.get("type") or ""),
                    "duplicate_of": spec_id,
                    "reason": (
                        "same executable fusion graph/type, but replaced by a higher-readiness spec "
                        f"({previous.get('status')} -> {audit.get('status')})"
                    ),
                    "readiness_status": previous.get("status"),
                    "duplicate_of_readiness_status": audit.get("status"),
                    "replaced_by_safer_spec": True,
                })
                continue
            skipped.append({
                "id": spec_id,
                "type": str(spec.get("type") or ""),
                "duplicate_of": previous_id,
                "reason": (
                    "same executable fusion graph/type as an earlier equal-or-higher-readiness spec"
                ),
                "readiness_status": audit.get("status"),
                "duplicate_of_readiness_status": previous.get("status"),
                "replaced_by_safer_spec": False,
            })
            continue
        seen[key] = {"id": spec_id, "index": len(kept), "rank": rank, "audit": audit, "status": audit.get("status"),
                     "type": str(spec.get("type") or "")}
        kept.append(spec)
    return kept, skipped


class Candidate:
    """A menu item: a label/spec + a zero-arg factory returning a fresh estimator."""

    def __init__(self, name: str, factory: Callable[[], Any], spec: dict[str, Any] | None = None):
        self.name = name
        self.factory = factory
        self.spec = spec or {}


def _candidate_uses_multifidelity(candidate: Candidate) -> bool:
    spec = candidate.spec or {}
    fspec = spec.get("fusion_spec") if isinstance(spec.get("fusion_spec"), dict) else {}
    return (
        str(spec.get("fusion_mode") or "") == "multi_fidelity_base_residual"
        or str(fspec.get("type") or "") == "multi_fidelity_base_residual"
    )


def _fidelity_config(spec: dict[str, Any] | None = None) -> dict[str, str]:
    fspec = (spec or {}).get("fusion_spec") if isinstance((spec or {}).get("fusion_spec"), dict) else {}
    executor = fspec.get("executor_spec") if isinstance(fspec.get("executor_spec"), dict) else {}
    return {
        "column": str(
            fspec.get("fidelity_column")
            or executor.get("fidelity_column")
            or "data_fidelity"
        ),
        "low": str(
            fspec.get("low_fidelity_label")
            or executor.get("low_fidelity_label")
            or "low_fidelity"
        ).strip().lower(),
        "high": str(
            fspec.get("high_fidelity_label")
            or executor.get("high_fidelity_label")
            or "high_fidelity"
        ).strip().lower(),
    }


def _fidelity_masks(df: pd.DataFrame, spec: dict[str, Any] | None = None) -> dict[str, Any] | None:
    cfg = _fidelity_config(spec)
    column = cfg["column"]
    if column not in df.columns:
        return None
    labels = df[column].astype(str).str.strip().str.lower()
    low_aliases = {cfg["low"], "lf", "low", "low_fidelity", "synthetic_low_fidelity"}
    high_aliases = {cfg["high"], "hf", "high", "high_fidelity", "real_hf"}
    low_mask = labels.isin(low_aliases).to_numpy(dtype=bool)
    high_mask = labels.isin(high_aliases).to_numpy(dtype=bool)
    if int(low_mask.sum()) < 1 or int(high_mask.sum()) < 2:
        return None
    return {
        "column": column,
        "low_label": cfg["low"],
        "high_label": cfg["high"],
        "low_mask": low_mask,
        "high_mask": high_mask,
        "low_count": int(low_mask.sum()),
        "high_count": int(high_mask.sum()),
    }


def _feature_columns_with_fidelity(
    feature_columns: list[str],
    df: pd.DataFrame,
    masks: dict[str, Any] | None,
) -> list[str]:
    cols = list(feature_columns)
    if masks is not None:
        column = str(masks.get("column") or "data_fidelity")
        if column in df.columns and column not in cols:
            cols.append(column)
    return cols


def evaluate_candidate(
    candidate: Candidate,
    df: pd.DataFrame,
    feature_columns: list[str],
    y: np.ndarray,
    *,
    n_splits: int,
    random_state: int,
    anchor_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Fold-local OOF (+ anchor) for one candidate. Returns metrics + predictions."""
    started = time.perf_counter()
    mf_masks = _fidelity_masks(df, candidate.spec)
    uses_mf = _candidate_uses_multifidelity(candidate)
    if uses_mf and mf_masks is None:
        return {
            "name": candidate.name,
            "spec": candidate.spec,
            "status": "failed",
            "error": "ValueError: multi_fidelity_base_residual requires both low_fidelity and high_fidelity rows.",
            "elapsed_seconds": float(time.perf_counter() - started),
        }

    if mf_masks is not None:
        meta_feature_columns = _feature_columns_with_fidelity(feature_columns, df, mf_masks)
        X = df[meta_feature_columns].copy()
        low_idx = np.flatnonzero(mf_masks["low_mask"])
        high_idx = np.flatnonzero(mf_masks["high_mask"])
        n_eff_splits = min(max(2, int(n_splits)), len(high_idx))
        splitter = KFold(n_splits=n_eff_splits, shuffle=True, random_state=random_state)
        oof = np.zeros(len(high_idx), dtype=float)
        fold_rows: list[dict[str, Any]] = []
        try:
            for fold, (hf_train_rel, hf_valid_rel) in enumerate(splitter.split(high_idx), start=1):
                hf_train_idx = high_idx[hf_train_rel]
                valid_idx = high_idx[hf_valid_rel]
                train_idx = (
                    np.concatenate([low_idx, hf_train_idx])
                    if uses_mf else hf_train_idx
                )
                est = candidate.factory()
                est.fit(X.iloc[train_idx], y[train_idx])
                oof[hf_valid_rel] = np.asarray(est.predict(X.iloc[valid_idx]), dtype=float)
                fold_rows.append({
                    "fold": int(fold),
                    "n_lf_train": int(len(low_idx) if uses_mf else 0),
                    "n_hf_train": int(len(hf_train_idx)),
                    "n_hf_valid": int(len(valid_idx)),
                })
        except Exception as exc:
            return {"name": candidate.name, "spec": candidate.spec, "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "elapsed_seconds": float(time.perf_counter() - started)}

        result: dict[str, Any] = {
            "name": candidate.name,
            "spec": candidate.spec,
            "status": "evaluated",
            "oof_metrics": _metrics(y[high_idx], oof),
            "_oof_pred": oof,
            "_oof_indices": high_idx,
            "elapsed_seconds": float(time.perf_counter() - started),
            "evaluation_protocol": {
                "type": "high_fidelity_oof",
                "candidate_uses_low_fidelity": bool(uses_mf),
                "n_low_fidelity_available": int(len(low_idx)),
                "n_high_fidelity_available": int(len(high_idx)),
                "n_splits": int(n_eff_splits),
                "folds": fold_rows,
                "metric_scope": "high_fidelity_rows_only",
            },
        }
        if anchor_df is not None and len(anchor_df) > 0:
            try:
                est = candidate.factory()
                full_train_idx = np.concatenate([low_idx, high_idx]) if uses_mf else high_idx
                est.fit(X.iloc[full_train_idx], y[full_train_idx])
                anchor_pred = np.asarray(est.predict(anchor_df[feature_columns].copy()), dtype=float)
                y_anchor = pd.to_numeric(anchor_df[TARGET_COLUMN], errors="coerce").to_numpy(float)
                result["anchor_metrics"] = _metrics(y_anchor, anchor_pred)
                result["_anchor_pred"] = anchor_pred
            except Exception as exc:
                result["anchor_metrics"] = None
                result["anchor_error"] = f"{type(exc).__name__}: {exc}"
            result["elapsed_seconds"] = float(time.perf_counter() - started)
        return result

    X = df[feature_columns].copy()
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof = np.zeros(len(df), dtype=float)
    try:
        for train_idx, valid_idx in splitter.split(X):
            est = candidate.factory()
            est.fit(X.iloc[train_idx], y[train_idx])       # train only
            oof[valid_idx] = np.asarray(est.predict(X.iloc[valid_idx]), dtype=float)  # no y
    except Exception as exc:
        return {"name": candidate.name, "spec": candidate.spec, "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_seconds": float(time.perf_counter() - started)}

    result: dict[str, Any] = {
        "name": candidate.name,
        "spec": candidate.spec,
        "status": "evaluated",
        "oof_metrics": _metrics(y, oof),
        "_oof_pred": oof,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    if anchor_df is not None and len(anchor_df) > 0:
        try:
            est = candidate.factory()
            est.fit(X, y)                                   # full train
            anchor_pred = np.asarray(est.predict(anchor_df[feature_columns].copy()), dtype=float)
            y_anchor = pd.to_numeric(anchor_df[TARGET_COLUMN], errors="coerce").to_numpy(float)
            result["anchor_metrics"] = _metrics(y_anchor, anchor_pred)
            result["_anchor_pred"] = anchor_pred
        except Exception as exc:
            result["anchor_metrics"] = None
            result["anchor_error"] = f"{type(exc).__name__}: {exc}"
        result["elapsed_seconds"] = float(time.perf_counter() - started)
    return result


# ---- feasibility filter (pair-level admissibility) ----

# models that need a minimum sample size to be trustworthy (small-data infeasible)
_LATENT_ARCHS = {"mlp2_hidden16", "mlp2_hidden64", "mlp3_hidden16", "branched_hidden16"}
_MODEL_MIN_SAMPLES = {
    "mlp": 60,
    "gpr": 40,
    "kernel": 40,
    "pinn": 60,
    "mlp2_hidden16": 60,
    "mlp2_hidden64": 60,
    "mlp3_hidden16": 60,
    "branched_hidden16": 60,
}


def _is_latent_model_family(model_family: Any) -> bool:
    fam = str(model_family or "").strip().lower()
    if fam in _LATENT_ARCHS:
        return True
    return (
        fam.startswith("llm_latent_")
        or fam.startswith("model_latent_")
        or (
            "latent" in fam
            and ("physics" in fam or "m1" in fam or "hidden_parameter" in fam)
        )
    )


def _normalize_fusion_mode(fusion_mode: str | None) -> str:
    return builtin_fusion_mode(normalize_fusion_spec(fusion_mode or "mechanism_features"))


def screen_pair(mechanism, model_family, *, columns, n_samples,
                fusion_mode: str | None = None, fusion_spec: dict[str, Any] | None = None,
                fidelity_values: list[str] | tuple[str, ...] | None = None,
                high_fidelity_count: int | None = None) -> dict[str, Any]:
    """Feasibility of ONE (mechanism, model, fusion_spec) triple.

    Returns {status: admitted|degraded|rejected, reasons: [...], mechanism_adequacy}.
    Mechanism side uses data_adequacy (reject/degrade); model side checks sample
    adequacy. Rejected pairs are recorded (not silently dropped) — mechanism
    rejection under missing observables is itself a reportable result.
    """
    reasons: list[str] = []
    status = "admitted"
    fspec = normalize_fusion_spec(fusion_spec or fusion_mode or "mechanism_features")
    mode = builtin_fusion_mode(fspec)

    readiness = fusion_spec_readiness(fspec, columns)
    spec_reasons = readiness["reasons"]
    if spec_reasons:
        return {"status": "rejected",
                "reasons": spec_reasons,
                "mechanism_adequacy": {"status": readiness["status"], "reason": "fusion spec rejected before mechanism screen."}}

    if mode == "multi_fidelity_base_residual":
        values = {str(v).strip().lower() for v in (fidelity_values or [])}
        if values and not ({"low_fidelity", "high_fidelity"} <= values):
            return {"status": "rejected",
                    "reasons": [
                        "multi_fidelity_base_residual requires both data_fidelity=low_fidelity and high_fidelity rows."
                    ],
                    "mechanism_adequacy": {"status": "not_applicable", "reason": "multi-fidelity screen failed before model fit."}}
        if mechanism is not None:
            return {"status": "rejected",
                    "reasons": ["multi_fidelity_base_residual v1 is a raw-feature LF-base + HF-residual route; mechanism-specific duplicates are skipped."],
                    "mechanism_adequacy": {"status": "not_applicable", "reason": "multi-fidelity v1 does not consume a mechanism."}}

    if mode == "raw_ml" and mechanism is not None:
        return {"status": "rejected",
                "reasons": ["raw_ml uses no mechanism; this duplicate mechanism-specific raw arm is skipped."],
                "mechanism_adequacy": {"status": "not_applicable", "reason": "raw_ml has no mechanism."}}

    if mode not in {"raw_ml", "multi_fidelity_base_residual"} and mechanism is None:
        return {"status": "rejected",
                "reasons": [f"{mode} requires a mechanism; raw arm only admits raw_ml."],
                "mechanism_adequacy": {"status": "not_applicable", "reason": "no mechanism (raw model arm)."}}

    if mechanism is not None:
        adq = mechanism.data_adequacy(columns)
        if adq["status"] == "rejected":
            return {"status": "rejected", "reasons": [adq["reason"]], "mechanism_adequacy": adq}
        if adq["status"] == "degraded":
            status = "degraded"
            reasons.append(adq["reason"])
    else:
        adq = {"status": "not_applicable", "reason": "no mechanism (raw model arm)."}

    if mode == "mechanism_features" and not hasattr(mechanism, "features"):
        return {"status": "rejected",
                "reasons": reasons + ["mechanism_features requires mechanism.features(df, params)."],
                "mechanism_adequacy": adq}
    if mode == "mechanism_residual" and not hasattr(mechanism, "base_predict"):
        return {"status": "rejected",
                "reasons": reasons + ["mechanism_residual requires mechanism.base_predict(df, params)."],
                "mechanism_adequacy": adq}
    if mode == "hidden_parameter_physics" and not hasattr(mechanism, "base_predict"):
        return {"status": "rejected",
                "reasons": reasons + ["hidden_parameter_physics requires mechanism.base_predict(df, params)."],
                "mechanism_adequacy": adq}
    if mode == "hidden_parameter_physics" and not _is_latent_model_family(model_family):
        return {"status": "rejected",
                "reasons": reasons + [
                    "hidden_parameter_physics requires one latent m1_eff architecture: "
                    f"{sorted(_LATENT_ARCHS)} or an LLM latent-physics model id."
                ],
                "mechanism_adequacy": adq}

    need = _MODEL_MIN_SAMPLES.get(str(model_family).lower())
    effective_n = int(high_fidelity_count or n_samples) if mode == "multi_fidelity_base_residual" else int(n_samples)
    if need is not None and effective_n < need:
        return {"status": "rejected",
                "reasons": reasons + [f"model '{model_family}' needs >= {need} samples, only {effective_n}."],
                "mechanism_adequacy": adq}
    return {"status": status, "reasons": reasons or ["inputs present."], "mechanism_adequacy": adq}


def build_pair_menu(mechanisms, models, *, columns, n_samples, fusion_modes=None,
                    fusion_specs=None, fidelity_values=None, high_fidelity_count: int | None = None) -> dict[str, Any]:
    """Cartesian product {mechanism} x {model} x {fusion_spec}, screened for feasibility.

    Returns {admitted: [...], rejected: [...]} where each entry carries the
    mechanism id, model family, fusion mode, and the screen verdict/reasons.
    """
    raw_specs = normalize_fusion_specs(fusion_specs if fusion_specs is not None else fusion_modes)
    specs, fusion_duplicates = _dedupe_fusion_specs_for_execution(raw_specs, columns)
    admitted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for mech in mechanisms:
        mech_id = getattr(mech, "id", None) if mech is not None else None
        for mdl in models:
            for fspec in specs:
                mode = builtin_fusion_mode(fspec)
                scr = screen_pair(
                    mech, mdl, columns=columns, n_samples=n_samples, fusion_spec=fspec,
                    fidelity_values=fidelity_values, high_fidelity_count=high_fidelity_count,
                )
                entry = {"mechanism": mech_id, "model_family": mdl,
                         "fusion_mode": mode, "fusion_id": fspec.get("id"), "fusion_spec": fspec,
                         "status": scr["status"], "reasons": scr["reasons"],
                         "mechanism_adequacy": scr["mechanism_adequacy"]}
                (rejected if scr["status"] == "rejected" else admitted).append(entry)
    return {"admitted": admitted, "rejected": rejected,
            "fusion_modes": [builtin_fusion_mode(spec) for spec in specs],
            "fusion_specs": specs,
            "fusion_specs_before_execution_dedupe": len(raw_specs),
            "fusion_specs_after_execution_dedupe": len(specs),
            "fusion_spec_duplicates": fusion_duplicates,
            "n_pairs": len(admitted) + len(rejected),
            "n_admitted": len(admitted), "n_rejected": len(rejected)}


def run_joint_search(
    candidates: list[Candidate],
    data_path: str | Path,
    *,
    anchor_path: str | Path | None = None,
    n_splits: int = 5,
    random_state: int = 42,
    min_anchor_r2: float = 0.0,
    oof_tol_frac: float = 0.02,
) -> dict[str, Any]:
    """Score every candidate the same way, rank by OOF RMSE, pick a champion.

    Champion selection is GUARDRAIL-AWARE: a candidate whose anchor R2 falls below
    `min_anchor_r2` is disqualified (synthetic-high / real-bad is not trusted). The
    champion is drawn from the viable-and-beats-baseline pool; on near-ties in OOF
    RMSE (within `oof_tol_frac`) the smaller OOF-anchor generalization gap wins.
    """
    df, meta = load_yield_dataframe(data_path)
    feature_columns = _feature_columns(meta, df)
    y = pd.to_numeric(df[TARGET_COLUMN], errors="coerce").to_numpy(float)
    dataset_mf_masks = _fidelity_masks(df)
    if dataset_mf_masks is not None:
        eval_n = int(dataset_mf_masks["high_count"])
    else:
        eval_n = int(len(df))
    n_splits = min(max(2, int(n_splits)), max(2, eval_n))

    anchor_df = None
    if anchor_path and Path(anchor_path).exists():
        try:
            anchor_df, _ = load_yield_dataframe(anchor_path)
        except Exception:
            anchor_df = None

    rows = [
        evaluate_candidate(c, df, feature_columns, y,
                           n_splits=n_splits, random_state=random_state, anchor_df=anchor_df)
        for c in candidates
    ]
    evaluated = [r for r in rows if r.get("status") == "evaluated"]

    if dataset_mf_masks is not None:
        high_idx = np.flatnonzero(dataset_mf_masks["high_mask"])
        baseline_df = df.iloc[high_idx].reset_index(drop=True)
        baseline_summary = _evaluate_baseline_summary(
            baseline_df,
            feature_columns,
            n_splits=n_splits,
            random_state=random_state,
        )
        evaluation_protocol = {
            "type": "high_fidelity_oof",
            "metric_scope": "high_fidelity_rows_only",
            "ordinary_candidates_train_on": "high_fidelity_train_fold_only",
            "multi_fidelity_candidates_train_on": "all_low_fidelity_rows_plus_high_fidelity_train_fold",
            "n_low_fidelity": int(dataset_mf_masks["low_count"]),
            "n_high_fidelity": int(dataset_mf_masks["high_count"]),
            "fidelity_column": str(dataset_mf_masks["column"]),
        }
    else:
        baseline_summary = _evaluate_baseline_summary(
            df,
            feature_columns,
            n_splits=n_splits,
            random_state=random_state,
        )
        evaluation_protocol = {
            "type": "standard_oof",
            "metric_scope": "all_rows",
        }
    best_baseline = baseline_summary.get("best_baseline") if isinstance(baseline_summary, dict) else None
    best_baseline_rmse = _safe_float((best_baseline or {}).get("oof_rmse", (best_baseline or {}).get("mean_rmse")))

    def oof_rmse(r):
        return _safe_float((r.get("oof_metrics") or {}).get("rmse"), float("inf"))

    def _r2(metrics):
        v = _safe_float((metrics or {}).get("r2"), None)
        return v

    have_anchor = anchor_df is not None and len(anchor_df) > 0
    baseline_improvement_frac = _safe_float(os.environ.get("YIELD_BASELINE_IMPROVEMENT_FRAC"), 0.005)
    baseline_improvement_abs = _safe_float(os.environ.get("YIELD_BASELINE_IMPROVEMENT_ABS"), 1e-9)
    baseline_improvement_threshold = None
    if best_baseline_rmse is not None:
        baseline_improvement_threshold = max(
            float(baseline_improvement_abs or 0.0),
            abs(float(best_baseline_rmse)) * float(baseline_improvement_frac or 0.0),
        )

    # annotate every candidate with the guardrail verdict + generalization gap
    for r in evaluated:
        a_r2 = _r2(r.get("anchor_metrics"))
        o_r2 = _r2(r.get("oof_metrics"))
        if best_baseline_rmse is None or baseline_improvement_threshold is None:
            r["baseline_status"] = "unknown"
            r["baseline_rmse_delta"] = None
            r["baseline_improvement_threshold"] = None
            r["beats_baseline"] = False
            r["ties_baseline"] = False
        else:
            delta = float(best_baseline_rmse) - float(oof_rmse(r))
            r["baseline_rmse_delta"] = delta
            r["baseline_improvement_threshold"] = float(baseline_improvement_threshold)
            if delta > baseline_improvement_threshold:
                r["baseline_status"] = "beats"
            elif abs(delta) <= baseline_improvement_threshold:
                r["baseline_status"] = "ties"
            else:
                r["baseline_status"] = "loses"
            r["beats_baseline"] = r["baseline_status"] == "beats"
            r["ties_baseline"] = r["baseline_status"] == "ties"
        r["oof_anchor_gap_r2"] = None if (a_r2 is None or o_r2 is None) else float(o_r2 - a_r2)
        if have_anchor and a_r2 is not None:
            r["viable"] = bool(a_r2 >= min_anchor_r2)
            r["viability_reason"] = (
                f"anchor R2 {a_r2:.3f} >= {min_anchor_r2:.3f}" if r["viable"]
                else f"anchor R2 {a_r2:.3f} < {min_anchor_r2:.3f} -> unreliable, disqualified")
        else:
            r["viable"] = True
            r["viability_reason"] = "no anchor; OOF-only (guardrail not enforced)"

    ranked = sorted(evaluated, key=oof_rmse)
    slowest = sorted(
        (
            {"name": r.get("name"), "elapsed_seconds": _safe_float(r.get("elapsed_seconds"), 0.0)}
            for r in rows
        ),
        key=lambda x: x.get("elapsed_seconds") or 0.0,
        reverse=True,
    )[:10]
    timing = {
        "candidate_eval_seconds_total": float(sum(_safe_float(r.get("elapsed_seconds"), 0.0) or 0.0 for r in rows)),
        "candidate_eval_seconds_mean": (
            float(np.mean([_safe_float(r.get("elapsed_seconds"), 0.0) or 0.0 for r in rows]))
            if rows else 0.0
        ),
        "slowest_candidates": slowest,
    }

    # champion pool: viable AND significantly beats baseline; degrade gracefully
    # but keep "ties baseline" distinct from true improvement.
    pool = [r for r in ranked if r["viable"] and r["beats_baseline"]]
    pool_basis = "viable_and_beats_baseline"
    if not pool:
        pool = [r for r in ranked if r["viable"] and r.get("ties_baseline")]
        pool_basis = "viable_and_ties_baseline"
    if not pool:
        pool = [r for r in ranked if r["viable"]]
        pool_basis = "viable_only"
    if not pool:
        pool = ranked
        pool_basis = "no_candidate_passed_guardrail"

    best_oof_row = ranked[0] if ranked else None
    champion = pool[0] if pool else None
    near_tie_candidates: list[dict[str, Any]] = []
    # near-tie on OOF -> prefer better generalization (smaller gap, then higher anchor R2)
    if champion is not None and have_anchor:
        lead = oof_rmse(champion)
        near = [r for r in pool if oof_rmse(r) <= lead * (1.0 + oof_tol_frac)]
        near_tie_candidates = near

        def gen_key(r):
            gap = r.get("oof_anchor_gap_r2")
            a_r2 = _r2(r.get("anchor_metrics"))
            return (gap if gap is not None else 1e9, -(a_r2 if a_r2 is not None else -1e9))

        champion = min(near, key=gen_key)

    def _public_row(r: dict[str, Any] | None) -> dict[str, Any] | None:
        if r is None:
            return None
        return {
            "name": r.get("name"),
            "oof_rmse": oof_rmse(r),
            "oof_r2": _r2(r.get("oof_metrics")),
            "anchor_r2": _r2(r.get("anchor_metrics")),
            "oof_anchor_gap_r2": r.get("oof_anchor_gap_r2"),
            "beats_baseline": r.get("beats_baseline"),
            "ties_baseline": r.get("ties_baseline"),
            "baseline_status": r.get("baseline_status"),
            "baseline_rmse_delta": r.get("baseline_rmse_delta"),
            "baseline_improvement_threshold": r.get("baseline_improvement_threshold"),
            "viable": r.get("viable"),
            "viability_reason": r.get("viability_reason"),
        }

    def _research_score(r: dict[str, Any]) -> tuple[Any, ...]:
        spec = r.get("spec") or {}
        has_mech = bool(spec.get("mechanism"))
        origin = str(spec.get("model_origin") or "")
        mode = str(spec.get("fusion_mode") or "")
        # This is a reporting champion, not the deployed/performance champion.
        # It prefers interpretable or generated routes that remain close to the
        # best OOF candidate.
        return (
            0 if (has_mech or origin == "llm_model" or mode != "raw_ml") else 1,
            0 if has_mech else 1,
            0 if origin == "llm_model" else 1,
            _FUSION_PRIORITY.get(mode, 99),
            oof_rmse(r),
            str(r.get("name") or ""),
        )

    research_candidates: list[dict[str, Any]] = []
    research_champion = None
    if ranked:
        lead_rmse = oof_rmse(best_oof_row)
        research_candidates = [
            r for r in ranked
            if r.get("viable") and oof_rmse(r) <= lead_rmse * (1.0 + float(oof_tol_frac))
        ]
        research_champion = min(research_candidates, key=_research_score, default=None)

    selected_differs_from_best_oof = bool(
        best_oof_row is not None and champion is not None and best_oof_row.get("name") != champion.get("name")
    )
    if champion is None:
        selection_explanation = "No evaluated candidate was available, so no champion was selected."
    elif selected_differs_from_best_oof and have_anchor:
        selection_explanation = (
            f"OOF最低的是 {best_oof_row.get('name')}，但它和候选池领先者处在 "
            f"{oof_tol_frac:.1%} 的近似并列范围内；guardrail 在近似并列时优先选择 "
            f"OOF-anchor R2 gap 更小、anchor R2 更高的组合，所以最终冠军是 {champion.get('name')}。"
        )
    elif selected_differs_from_best_oof:
        selection_explanation = (
            f"OOF最低的是 {best_oof_row.get('name')}，但它没有进入当前冠军池 "
            f"({pool_basis})；最终冠军是 {champion.get('name')}。"
        )
    elif have_anchor:
        selection_explanation = (
            f"最终冠军 {champion.get('name')} 同时是当前冠军池里的 OOF 领先候选；"
            "anchor guardrail 没有改变冠军。"
        )
    else:
        selection_explanation = (
            f"最终冠军 {champion.get('name')} 是当前冠军池里的 OOF 领先候选；"
            "本轮没有可用 anchor，所以没有执行 anchor tie-break。"
        )

    champion_selection = {
        "pool_basis": pool_basis,
        "oof_tolerance_frac": float(oof_tol_frac),
        "baseline_improvement_frac": float(baseline_improvement_frac or 0.0),
        "baseline_improvement_abs": float(baseline_improvement_abs or 0.0),
        "baseline_improvement_threshold": baseline_improvement_threshold,
        "anchor_guardrail_used": bool(have_anchor),
        "selected_differs_from_lowest_oof": selected_differs_from_best_oof,
        "lowest_oof_candidate": _public_row(best_oof_row),
        "selected_champion": _public_row(champion),
        "research_champion": _public_row(research_champion),
        "research_near_tie_count": len(research_candidates),
        "research_near_tie_candidates": [_public_row(r) for r in research_candidates[:10]],
        "near_tie_count": len(near_tie_candidates),
        "near_tie_candidates": [_public_row(r) for r in near_tie_candidates[:10]],
        "explanation": selection_explanation,
    }

    return {
        "stage": "joint_search",
        "data_path": str(data_path),
        "anchor_path": str(anchor_path) if anchor_path else "",
        "feature_columns": feature_columns,
        "evaluation_protocol": evaluation_protocol,
        "n_splits": int(n_splits),
        "random_state": int(random_state),
        "n_candidates": len(candidates),
        "n_evaluated": len(evaluated),
        "n_viable": sum(1 for r in evaluated if r.get("viable")),
        "best_fixed_baseline_oof_rmse": best_baseline_rmse,
        "baseline_improvement_threshold": baseline_improvement_threshold,
        "min_anchor_r2": min_anchor_r2,
        "champion_pool_basis": pool_basis,
        "champion_selection": champion_selection,
        "research_champion": _public_row(research_champion),
        "ranked": [
            {"name": r["name"], "spec": r.get("spec"),
             "oof_metrics": r.get("oof_metrics"), "anchor_metrics": r.get("anchor_metrics"),
             "beats_baseline": r.get("beats_baseline"),
             "ties_baseline": r.get("ties_baseline"),
             "baseline_status": r.get("baseline_status"),
             "baseline_rmse_delta": r.get("baseline_rmse_delta"),
             "baseline_improvement_threshold": r.get("baseline_improvement_threshold"),
             "viable": r.get("viable"),
             "viability_reason": r.get("viability_reason"),
             "oof_anchor_gap_r2": r.get("oof_anchor_gap_r2"),
             "evaluation_protocol": r.get("evaluation_protocol"),
             "elapsed_seconds": r.get("elapsed_seconds")}
            for r in ranked
        ],
        "failed": [
            {"name": r["name"], "error": r.get("error"), "elapsed_seconds": r.get("elapsed_seconds")}
            for r in rows if r.get("status") == "failed"
        ],
        "timing": timing,
        "champion": None if champion is None else {
            "name": champion["name"], "spec": champion.get("spec"),
            "oof_metrics": champion.get("oof_metrics"), "anchor_metrics": champion.get("anchor_metrics"),
            "beats_baseline": champion.get("beats_baseline"),
            "ties_baseline": champion.get("ties_baseline"),
            "baseline_status": champion.get("baseline_status"),
            "baseline_rmse_delta": champion.get("baseline_rmse_delta"),
            "baseline_improvement_threshold": champion.get("baseline_improvement_threshold"),
            "viable": champion.get("viable"), "viability_reason": champion.get("viability_reason"),
            "oof_anchor_gap_r2": champion.get("oof_anchor_gap_r2"),
            "evaluation_protocol": champion.get("evaluation_protocol"),
            "elapsed_seconds": champion.get("elapsed_seconds"),
        },
        "_champion_row": champion,   # carries _oof_pred/_anchor_pred for finalize
    }


_FUSION_PRIORITY = {
    "raw_ml": 0,
    "mechanism_features": 1,
    "mechanism_residual": 2,
    "hidden_parameter_physics": 3,
    "multi_fidelity_base_residual": 4,
}

_MODEL_PRIORITY = {
    "ridge": 0,
    "rf": 1,
    "hgb": 2,
    "extratrees": 3,
    "kernel": 4,
    "svr": 5,
    "mlp2_hidden16": 6,
    "mlp3_hidden16": 7,
    "branched_hidden16": 8,
    "mlp2_hidden64": 9,
    "mlp": 10,
    "pinn": 11,
}


def _candidate_hparam_index(candidate: Candidate) -> int:
    match = re.search(r"#(\d+)$", str(candidate.name))
    return int(match.group(1)) if match else 0


def _hparam_key(hparams: Any) -> str:
    if not isinstance(hparams, dict):
        return repr(hparams)
    return repr(sorted((str(k), repr(v)) for k, v in hparams.items()))


def _executor_graph_key(graph: Any, mode: str) -> tuple[Any, ...]:
    norm = normalize_executor_graph(graph or builtin_executor_graph(mode, graph_id=f"{mode}_graph"))
    return tuple(
        (
            str(step.get("op") or ""),
            str(step.get("phase") or ""),
            tuple(str(x) for x in (step.get("inputs") or [])),
            tuple(str(x) for x in (step.get("outputs") or [])),
        )
        for step in norm.get("steps", [])
    )


def _candidate_execution_key(candidate: Candidate) -> tuple[Any, ...]:
    spec = candidate.spec or {}
    mode = str(spec.get("fusion_mode") or "unknown")
    return (
        str(spec.get("mechanism") or "raw"),
        str(spec.get("model_family") or ""),
        str(spec.get("model_origin") or ""),
        _hparam_key(spec.get("hparams") or {}),
        mode,
        _executor_graph_key(spec.get("executor_graph"), mode),
    )


def _candidate_family_priority(family: Any) -> int:
    text = str(family or "").strip().lower()
    if text in _MODEL_PRIORITY:
        return _MODEL_PRIORITY[text]
    if "ridge" in text or "linear" in text:
        return _MODEL_PRIORITY["ridge"]
    if "rf" in text or "forest" in text:
        return _MODEL_PRIORITY["rf"]
    if "hgb" in text or "boost" in text:
        return _MODEL_PRIORITY["hgb"]
    if "kernel" in text or "gpr" in text or "gaussian" in text:
        return _MODEL_PRIORITY["kernel"]
    if "svr" in text or "svm" in text:
        return _MODEL_PRIORITY["svr"]
    if "pinn" in text or "physics" in text:
        return _MODEL_PRIORITY["pinn"]
    if "mlp" in text or "neural" in text:
        return _MODEL_PRIORITY["mlp"]
    return 6


def _candidate_budget_group(candidate: Candidate) -> tuple[str, str, str]:
    spec = candidate.spec or {}
    return (
        str(spec.get("fusion_mode") or "unknown"),
        str(spec.get("mechanism") or "raw"),
        str(spec.get("model_origin") or "fixed_family"),
    )


def _candidate_budget_priority(candidate: Candidate) -> tuple[Any, ...]:
    spec = candidate.spec or {}
    mode = str(spec.get("fusion_mode") or "")
    origin = str(spec.get("model_origin") or "")
    # Keep generated model code visible in budgeted runs, but still prefer the
    # first hyperparameter setting before exploring deeper variants.
    origin_bonus = 0 if origin in {"llm_model", "planned_llm_model"} else 1
    return (
        _candidate_hparam_index(candidate),
        _MODEL_PRIORITY.get(str(spec.get("model_family") or "").lower(),
                            _candidate_family_priority(spec.get("model_family"))),
        origin_bonus,
        _FUSION_PRIORITY.get(mode, 99),
        str(candidate.name),
    )


def apply_candidate_budget(candidates: list[Candidate]) -> tuple[list[Candidate], dict[str, Any]]:
    """De-duplicate and cap runnable candidates while preserving search coverage."""
    cfg = search_budget_config()
    max_candidates = int(cfg.get("max_candidates") or 0)
    original_count = len(candidates)

    name_deduped: list[Candidate] = []
    seen_names: set[str] = set()
    duplicate_names: list[str] = []
    for cand in candidates:
        if cand.name in seen_names:
            duplicate_names.append(cand.name)
            continue
        seen_names.add(cand.name)
        name_deduped.append(cand)

    deduped: list[Candidate] = []
    seen_execution: dict[tuple[Any, ...], str] = {}
    duplicate_execution: list[dict[str, str]] = []
    for cand in name_deduped:
        key = _candidate_execution_key(cand)
        if key in seen_execution:
            duplicate_execution.append({
                "name": cand.name,
                "duplicate_of": seen_execution[key],
                "reason": "same mechanism/model/hparams/fusion execution graph",
            })
            continue
        seen_execution[key] = cand.name
        deduped.append(cand)

    if max_candidates <= 0 or len(deduped) <= max_candidates:
        selected = deduped
        truncated = False
    else:
        groups: dict[tuple[str, str, str], list[Candidate]] = {}
        for cand in deduped:
            groups.setdefault(_candidate_budget_group(cand), []).append(cand)
        for group_candidates in groups.values():
            group_candidates.sort(key=_candidate_budget_priority)

        selected = []
        group_keys = sorted(groups, key=lambda key: (
            _FUSION_PRIORITY.get(key[0], 99),
            key[1],
            0 if key[2] in {"llm_model", "planned_llm_model"} else 1,
        ))
        while len(selected) < max_candidates:
            progressed = False
            for key in group_keys:
                bucket = groups[key]
                if not bucket:
                    continue
                selected.append(bucket.pop(0))
                progressed = True
                if len(selected) >= max_candidates:
                    break
            if not progressed:
                break
        truncated = len(selected) < len(deduped)

    selected_names = {cand.name for cand in selected}
    selected_by_fusion: dict[str, int] = {}
    selected_by_origin: dict[str, int] = {}
    selected_by_mechanism: dict[str, int] = {}
    for cand in selected:
        spec = cand.spec or {}
        mode = str(spec.get("fusion_mode") or "unknown")
        origin = str(spec.get("model_origin") or "fixed_family")
        mech = str(spec.get("mechanism") or "raw")
        selected_by_fusion[mode] = selected_by_fusion.get(mode, 0) + 1
        selected_by_origin[origin] = selected_by_origin.get(origin, 0) + 1
        selected_by_mechanism[mech] = selected_by_mechanism.get(mech, 0) + 1

    dropped = [cand.name for cand in deduped if cand.name not in selected_names]
    audit = {
        "mode": cfg.get("mode"),
        "max_candidates": max_candidates,
        "max_hparams_per_model": cfg.get("max_hparams_per_model"),
        "max_iter": cfg.get("max_iter"),
        "n_estimators": cfg.get("n_estimators"),
        "n_candidates_before_budget": original_count,
        "n_candidates_after_name_dedupe": len(name_deduped),
        "n_candidates_after_execution_dedupe": len(deduped),
        "n_candidates_selected": len(selected),
        "n_candidates_dropped": max(0, original_count - len(selected)),
        "n_candidates_removed_as_duplicates": max(0, original_count - len(deduped)),
        "n_candidates_dropped_by_budget": max(0, len(deduped) - len(selected)),
        "truncated": truncated,
        "duplicate_name_count": len(duplicate_names),
        "duplicate_name_examples": duplicate_names[:10],
        "duplicate_execution_count": len(duplicate_execution),
        "duplicate_execution_examples": duplicate_execution[:10],
        "selected_by_fusion_mode": selected_by_fusion,
        "selected_by_model_origin": selected_by_origin,
        "selected_by_mechanism": selected_by_mechanism,
        "dropped_examples": dropped[:20],
    }
    return selected, audit


# ============================================================================
# Upper layer — free search over {searched mechanism} x {searched model}
# ============================================================================

def run_free_search(
    mechanisms: list[Any],
    models: list[Any],
    data_path: str | Path,
    *,
    candidate_builder: Callable[[dict[str, Any]], Candidate | None],
    fusion_modes: list[str] | tuple[str, ...] | None = None,
    fusion_specs: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None = None,
    anchor_path: str | Path | None = None,
    n_splits: int = 5,
    random_state: int = 42,
    min_anchor_r2: float = 0.0,
) -> dict[str, Any]:
    """The direction-④ upper loop: FREE combination = the cartesian product of
    searched mechanisms x searched models x fusion modes.

      1. build_pair_menu -> cartesian product, feasibility-screened (rejected
         pairs are RECORDED with reasons, not dropped).
      2. `candidate_builder(triple)` turns each admitted
         (mechanism, model, fusion_mode) triple into runnable Candidate objects.
         fusion_mode is now an explicit searched axis, not an implicit fixed
         wiring hidden inside the candidate.
      3. run_joint_search scores every candidate identically and picks a
         guardrail-aware champion.

    `mechanisms` are Mechanism-like objects (need `.id` + `.data_adequacy`);
    `models` are model-family identifiers (strings). `fusion_specs` is the new
    structured searchable interface; `fusion_modes` remains a legacy shorthand.
    """
    run_started = time.perf_counter()
    df, meta = load_yield_dataframe(data_path)
    columns = list(df.columns)
    n_samples = len(df)
    if "data_fidelity" in df.columns:
        fidelity_values = sorted(str(v).strip().lower() for v in df["data_fidelity"].dropna().unique().tolist())
        high_fidelity_count = int((df["data_fidelity"].astype(str).str.strip().str.lower() == "high_fidelity").sum())
    else:
        fidelity_values = []
        high_fidelity_count = None

    menu_started = time.perf_counter()
    menu = build_pair_menu(mechanisms, models, columns=columns, n_samples=n_samples,
                           fusion_modes=fusion_modes, fusion_specs=fusion_specs,
                           fidelity_values=fidelity_values,
                           high_fidelity_count=high_fidelity_count)
    menu_seconds = time.perf_counter() - menu_started

    build_started = time.perf_counter()
    candidates: list[Candidate] = []
    build_errors: list[dict[str, Any]] = []
    for entry in menu["admitted"]:
        try:
            built = candidate_builder(entry)
        except Exception as exc:  # a builder failure drops one pair, never the run
            build_errors.append({"pair": entry, "error": f"{type(exc).__name__}: {exc}"})
            continue
        if built is None:
            continue
        # a builder may return one Candidate or a list (e.g. one per hparam config)
        if isinstance(built, Candidate):
            built = [built]
        candidates.extend([c for c in built if c is not None])
    build_seconds = time.perf_counter() - build_started

    n_built_before_budget = len(candidates)
    candidates, budget_audit = apply_candidate_budget(candidates)

    eval_started = time.perf_counter()
    search = run_joint_search(
        candidates, data_path, anchor_path=anchor_path,
        n_splits=n_splits, random_state=random_state, min_anchor_r2=min_anchor_r2,
    )
    evaluation_seconds = time.perf_counter() - eval_started
    timing = {
        "total_seconds": float(time.perf_counter() - run_started),
        "menu_seconds": float(menu_seconds),
        "build_seconds": float(build_seconds),
        "evaluation_seconds": float(evaluation_seconds),
        "candidate_eval_seconds_total": (search.get("timing") or {}).get("candidate_eval_seconds_total"),
        "slowest_candidates": (search.get("timing") or {}).get("slowest_candidates"),
    }

    return {
        "stage": "free_search",
        "pair_menu": menu,
        "n_built_before_budget": n_built_before_budget,
        "n_built": len(candidates),
        "budget": budget_audit,
        "timing": timing,
        "n_splits": int(n_splits),
        "random_state": int(random_state),
        "build_errors": build_errors,
        "search": search,
        "champion": search.get("champion"),
    }


# ============================================================================
# Production candidate builder — the single canonical (mechanism, model) wiring
# ============================================================================

class PhysicsInformedRegressor:
    """Physics-informed neural regressor (model family 'pinn') — a "complex model"
    beyond the plain sklearn families.

    Core PINN idea for this algebraic-mechanism regression setting: a neural
    backbone corrects a PHYSICS BASE prediction rather than regressing y from
    scratch. When `physics_base` is supplied at fit/predict (the mechanism's
    fold-local base tau), the net learns the RESIDUAL y - base and predicts
    base + residual; without it (raw arm) it regresses directly. Output is clamped
    non-negative (a hard physics constraint: yield stress >= 0).

    torch-free by design: uses an sklearn MLP backbone so it runs in this
    environment (torch 2.2.1 has no cp313 wheel). A torch autograd backend with an
    explicit physics-consistency loss term is the documented next upgrade; the
    fit/predict signature already carries `physics_base` so that swap is local.
    """

    def __init__(self, random_state: int = 0, hidden=(64, 32, 16), alpha: float = 1e-3,
                 max_iter: int = 500):
        self.random_state = int(random_state)
        self.hidden = tuple(hidden)
        self.alpha = float(alpha)
        self.max_iter = int(_cap_training_value("max_iter", max_iter))
        self.net_ = None
        self._residual = False

    def fit(self, X, y, physics_base=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self._residual = physics_base is not None
        target = (y - np.asarray(physics_base, dtype=float)) if self._residual else y
        self.net_ = make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=self.hidden, alpha=self.alpha,
                         max_iter=self.max_iter, early_stopping=True,
                         random_state=self.random_state),
        )
        self.net_.fit(X, target)
        return self

    def predict(self, X, physics_base=None):
        X = np.asarray(X, dtype=float)
        out = np.asarray(self.net_.predict(X), dtype=float)
        if physics_base is not None:
            out = out + np.asarray(physics_base, dtype=float)
        return np.clip(out, 0.0, None)


# per-family hyperparameter grids the search sweeps (P4). Each entry is a dict of
# overrides applied on top of the family defaults; {} = the default config. Kept
# small so {mechanism x model x hparam} stays tractable.
_HPARAM_GRIDS: dict[str, list[dict[str, Any]]] = {
    "rf": [{}, {"n_estimators": 400, "min_samples_leaf": 1}, {"max_depth": 8, "min_samples_leaf": 5}],
    "extratrees": [{}, {"n_estimators": 500, "min_samples_leaf": 1}],
    "hgb": [{}, {"learning_rate": 0.03, "max_iter": 600}, {"max_depth": 3, "learning_rate": 0.1}],
    "svr": [{}, {"C": 100.0}, {"C": 1.0, "gamma": 0.1}],
    "ridge": [{"alpha": 1.0}, {"alpha": 0.1}, {"alpha": 10.0}],
    "mlp": [{}, {"hidden_layer_sizes": (64, 32)}],
    "pinn": [{}, {"hidden": (64, 32)}],
    "kernel": [{}],
    "mlp2_hidden16": [{}],
    "mlp2_hidden64": [{}],
    "mlp3_hidden16": [{}],
    "branched_hidden16": [{}],
}


def hparam_grid(family: str) -> list[dict[str, Any]]:
    grid = list(_HPARAM_GRIDS.get(str(family).lower(), [{}]))
    cap = int(search_budget_config().get("max_hparams_per_model") or 0)
    return grid[:cap] if cap > 0 else grid


def _make_model(family: str, random_state: int, n_samples: int, hparams: dict[str, Any] | None = None):
    fam = str(family).lower()
    hp = dict(hparams or {})
    if fam == "mlp2_hidden16":
        hp.setdefault("hidden_layer_sizes", (16, 16))
        fam = "mlp"
    elif fam == "mlp2_hidden64":
        hp.setdefault("hidden_layer_sizes", (64, 64))
        fam = "mlp"
    elif fam == "mlp3_hidden16":
        hp.setdefault("hidden_layer_sizes", (16, 16, 8))
        fam = "mlp"
    elif fam == "branched_hidden16":
        # Outside the latent-physics graph, use a comparable compact MLP fallback
        # so the family remains runnable in raw/features/residual baselines.
        hp.setdefault("hidden_layer_sizes", (16, 16, 8))
        fam = "mlp"
    if fam == "pinn":
        return _cap_estimator_complexity(PhysicsInformedRegressor(
            random_state=random_state,
            hidden=hp.get("hidden", (64, 32, 16)),
            alpha=hp.get("alpha", 1e-3),
            max_iter=hp.get("max_iter", 500),
        ))
    if fam == "ridge":
        return _cap_estimator_complexity(make_pipeline(StandardScaler(), Ridge(alpha=hp.get("alpha", 1.0))))
    if fam == "rf":
        return _cap_estimator_complexity(RandomForestRegressor(
            n_estimators=hp.get("n_estimators", 200),
            max_depth=hp.get("max_depth", None),
            min_samples_leaf=hp.get("min_samples_leaf", 3),
            random_state=random_state, n_jobs=-1,
        ))
    if fam == "extratrees":
        return _cap_estimator_complexity(ExtraTreesRegressor(
            n_estimators=hp.get("n_estimators", 300),
            max_depth=hp.get("max_depth", None),
            min_samples_leaf=hp.get("min_samples_leaf", 2),
            random_state=random_state, n_jobs=-1,
        ))
    if fam == "svr":
        return _cap_estimator_complexity(make_pipeline(
            StandardScaler(),
            SVR(C=hp.get("C", 10.0), gamma=hp.get("gamma", "scale"),
                epsilon=hp.get("epsilon", 0.01)),
        ))
    if fam == "mlp":
        return _cap_estimator_complexity(make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=hp.get("hidden_layer_sizes", (32, 16)),
                         alpha=hp.get("alpha", 1e-3), max_iter=hp.get("max_iter", 300),
                         early_stopping=True, random_state=random_state),
        ))
    if fam == "kernel":
        kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=1.5) + WhiteKernel(1e-3)
        return _cap_estimator_complexity(make_pipeline(
            StandardScaler(),
            GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True,
                                     random_state=random_state),
        ))
    return _cap_estimator_complexity(HistGradientBoostingRegressor(
        max_iter=hp.get("max_iter", 300),
        learning_rate=hp.get("learning_rate", 0.06),
        max_depth=hp.get("max_depth", 6),
        min_samples_leaf=hp.get("min_samples_leaf", 10),
        l2_regularization=hp.get("l2_regularization", 1.0),
        random_state=random_state,
    ))


def _numeric_encode(df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    if cols is None:
        X = df.select_dtypes(include=[np.number]).copy()
    else:
        X = df.reindex(columns=cols)
    return X.apply(pd.to_numeric, errors="coerce").fillna(0.0)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def _softplus(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -40.0, 40.0)
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def _inverse_softplus(y: float) -> float:
    y = max(float(y), 1e-6)
    if y > 30.0:
        return y
    return float(np.log(np.expm1(y)))


class LatentM1PhysicsLayerRegressor:
    """Small NumPy neural latent model for y = m1_eff(X) * physics_shape(X).

    This implements the Chen-inspired A/B/C/D latent architectures without
    adding torch as a dependency. For the linear scale latent m1_eff, the
    train-fold target is the algebraic physical inversion m1_eff = y / g(X);
    prediction still goes through the physics layer y = m1_eff * g(X). It is
    fold-local and sklearn-like (fit/predict) so the existing OOF harness can
    score it.
    """

    def __init__(
        self,
        arch: str = "mlp2_hidden16",
        *,
        random_state: int = 0,
        latent_backend: str = "auto",
        learning_rate: float = 0.001,
        max_iter: int = 500,
        alpha: float = 1e-4,
    ):
        self.arch = str(arch)
        self.random_state = int(random_state)
        self.latent_backend = str(latent_backend or "auto").strip().lower()
        self.learning_rate = float(learning_rate)
        self.max_iter = int(_cap_training_value("max_iter", max_iter))
        self.alpha = float(alpha)
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self._cols: list[str] | None = None
        self._weights: list[np.ndarray] = []
        self._biases: list[np.ndarray] = []
        self._branches: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._fusion_w: np.ndarray | None = None
        self._fusion_b: np.ndarray | None = None
        self._out_w: np.ndarray | None = None
        self._out_b: np.ndarray | None = None
        self._latent_scale: float = 1.0
        self._latent_target_: np.ndarray | None = None
        self._sk_model = None
        self._sk_target_mean: float = 0.0
        self._sk_target_std: float = 1.0

    @staticmethod
    def _clip_grad(x: np.ndarray, limit: float = 5.0) -> np.ndarray:
        return np.clip(np.nan_to_num(x, nan=0.0, posinf=limit, neginf=-limit), -limit, limit)

    @staticmethod
    def _clip_param(x: np.ndarray, limit: float = 100.0) -> np.ndarray:
        return np.clip(np.nan_to_num(x, nan=0.0, posinf=limit, neginf=-limit), -limit, limit)

    def _snapshot(self) -> dict[str, Any]:
        return {
            "weights": [w.copy() for w in self._weights],
            "biases": [b.copy() for b in self._biases],
            "branches": {k: (idx.copy(), w.copy(), b.copy()) for k, (idx, w, b) in self._branches.items()},
            "fusion_w": None if self._fusion_w is None else self._fusion_w.copy(),
            "fusion_b": None if self._fusion_b is None else self._fusion_b.copy(),
            "out_w": None if self._out_w is None else self._out_w.copy(),
            "out_b": None if self._out_b is None else self._out_b.copy(),
        }

    def _restore(self, state: dict[str, Any]):
        self._weights = [w.copy() for w in state.get("weights", [])]
        self._biases = [b.copy() for b in state.get("biases", [])]
        self._branches = {k: (idx.copy(), w.copy(), b.copy()) for k, (idx, w, b) in state.get("branches", {}).items()}
        self._fusion_w = None if state.get("fusion_w") is None else state["fusion_w"].copy()
        self._fusion_b = None if state.get("fusion_b") is None else state["fusion_b"].copy()
        self._out_w = None if state.get("out_w") is None else state["out_w"].copy()
        self._out_b = None if state.get("out_b") is None else state["out_b"].copy()

    def _standardize_fit(self, X: pd.DataFrame) -> np.ndarray:
        self._cols = list(X.columns)
        arr = X.to_numpy(dtype=float)
        self._mean = np.mean(arr, axis=0)
        self._std = np.std(arr, axis=0)
        self._std = np.where(self._std < 1e-8, 1.0, self._std)
        return (arr - self._mean) / self._std

    def _standardize_predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._cols is None or self._mean is None or self._std is None:
            raise ValueError("LatentM1PhysicsLayerRegressor is not fitted.")
        arr = X.reindex(columns=self._cols).fillna(0.0).to_numpy(dtype=float)
        return (arr - self._mean) / self._std

    def _init_dense(self, n_features: int, output_bias: float):
        rng = np.random.default_rng(self.random_state)
        if self.arch == "mlp2_hidden64":
            sizes = [n_features, 64, 64, 1]
        elif self.arch == "mlp3_hidden16":
            sizes = [n_features, 16, 16, 8, 1]
        else:
            sizes = [n_features, 16, 16, 1]
        self._weights = [
            rng.normal(0.0, np.sqrt(2.0 / max(1, sizes[i])), size=(sizes[i], sizes[i + 1]))
            for i in range(len(sizes) - 1)
        ]
        self._biases = [np.zeros((1, size), dtype=float) for size in sizes[1:]]
        self._weights[-1] = np.zeros_like(self._weights[-1])
        self._biases[-1][0, 0] = output_bias

    @staticmethod
    def _group_indices(cols: list[str]) -> dict[str, np.ndarray]:
        groups = {"thermal": [], "composition": [], "mixing": []}
        for i, col in enumerate(cols):
            c = str(col).lower()
            if any(t in c for t in ("temp", "pressure", "humidity", "jacket", "slurry", "water", "room")):
                groups["thermal"].append(i)
            elif any(t in c for t in ("rpm", "time", "mixing", "shear", "reverse", "forward", "stage", "dose")):
                groups["mixing"].append(i)
            else:
                groups["composition"].append(i)
        return {k: np.asarray(v, dtype=int) for k, v in groups.items() if v}

    def _init_branched(self, cols: list[str], output_bias: float):
        rng = np.random.default_rng(self.random_state)
        self._branches = {}
        for name, idx in self._group_indices(cols).items():
            w = rng.normal(0.0, np.sqrt(2.0 / max(1, len(idx))), size=(len(idx), 16))
            b = np.zeros((1, 16), dtype=float)
            self._branches[name] = (idx, w, b)
        n_concat = 16 * max(1, len(self._branches))
        self._fusion_w = rng.normal(0.0, np.sqrt(2.0 / max(1, n_concat)), size=(n_concat, 16))
        self._fusion_b = np.zeros((1, 16), dtype=float)
        self._out_w = np.zeros((16, 1), dtype=float)
        self._out_b = np.array([[output_bias]], dtype=float)

    def _forward_dense(self, X: np.ndarray) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
        activations = [X]
        preacts: list[np.ndarray] = []
        a = X
        for i, (w, b) in enumerate(zip(self._weights, self._biases)):
            z = a @ w + b
            preacts.append(z)
            a = z if i == len(self._weights) - 1 else np.maximum(z, 0.0)
            activations.append(a)
        return a, activations, preacts

    def _forward_branched(self, X: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        branch_cache: dict[str, Any] = {}
        outs = []
        for name, (idx, w, b) in self._branches.items():
            z = X[:, idx] @ w + b
            a = np.maximum(z, 0.0)
            outs.append(a)
            branch_cache[name] = (idx, z, a)
        concat = np.concatenate(outs, axis=1)
        zf = concat @ self._fusion_w + self._fusion_b
        af = np.maximum(zf, 0.0)
        z = af @ self._out_w + self._out_b
        return z, {"branches": branch_cache, "concat": concat, "zf": zf, "af": af}

    def _latent_loss_grad(self, z: np.ndarray, target: np.ndarray, weights: np.ndarray) -> tuple[float, np.ndarray]:
        latent = _softplus(z) + 1e-8
        resid = (latent[:, 0] - target) / self._latent_scale
        loss = float(np.mean(weights * resid * resid))
        d_latent = (2.0 / len(target)) * weights * resid / self._latent_scale
        dz = (d_latent * _sigmoid(z[:, 0]))[:, None]
        return loss, dz

    def fit(self, X: pd.DataFrame, y, physics_shape):
        yv = np.asarray(y, dtype=float)
        g = np.asarray(physics_shape, dtype=float)
        g = np.where(np.abs(g) < 1e-8, 1e-8, g)
        latent_guess = np.clip(yv / g, 1e-6, None)
        self._latent_target_ = latent_guess.astype(float)
        self._latent_scale = max(float(np.std(latent_guess)), 1.0)
        weights = np.square(g)
        weights = weights / max(float(np.mean(weights)), 1e-8)
        output_bias = _inverse_softplus(float(np.median(latent_guess)))
        lr = self.learning_rate

        backend = self.latent_backend
        if backend == "auto":
            backend = "ridge" if len(X) < 300 else "mlp"
        backend = {
            "linear": "ridge",
            "regularized_linear": "ridge",
            "kernel": "kernel_ridge",
            "rbf": "kernel_ridge",
        }.get(backend, backend)

        if self.arch in _LATENT_ARCHS:
            hidden = {
                "mlp2_hidden64": (64, 64),
                "mlp3_hidden16": (16, 16, 8),
                "branched_hidden16": (48, 16),
            }.get(self.arch, (16, 16))
            self._cols = list(X.columns)
            self._sk_target_mean = float(np.mean(latent_guess))
            self._sk_target_std = max(float(np.std(latent_guess)), 1e-6)
            target_z = (latent_guess - self._sk_target_mean) / self._sk_target_std
            if self.arch == "branched_hidden16":
                groups = self._group_indices(self._cols)
                transformer = ColumnTransformer(
                    [(name, StandardScaler(), idx.tolist()) for name, idx in groups.items()],
                    remainder="drop",
                ) if groups else StandardScaler()
            else:
                transformer = StandardScaler()
            if backend == "ridge":
                latent_estimator = Ridge(alpha=max(self.alpha, 1e-8))
            elif backend == "kernel_ridge":
                gamma = 1.0 / max(1, X.shape[1])
                latent_estimator = KernelRidge(alpha=max(self.alpha, 1e-8), kernel="rbf", gamma=gamma)
            else:
                latent_estimator = MLPRegressor(
                    hidden_layer_sizes=hidden,
                    alpha=self.alpha,
                    learning_rate_init=self.learning_rate,
                    max_iter=self.max_iter,
                    early_stopping=len(X) >= 30,
                    random_state=self.random_state,
                )
            self._sk_model = make_pipeline(transformer, latent_estimator)
            self._sk_model.fit(X.to_numpy(dtype=float), target_z)
            return self

        Xs = self._standardize_fit(X)
        if self.arch == "branched_hidden16":
            self._init_branched(list(X.columns), output_bias)
            best_loss = float("inf")
            best_state = self._snapshot()
            for _ in range(self.max_iter):
                z, cache = self._forward_branched(Xs)
                loss, dz = self._latent_loss_grad(z, latent_guess, weights)
                if np.isfinite(loss) and loss < best_loss:
                    best_loss = loss
                    best_state = self._snapshot()
                af = cache["af"]
                concat = cache["concat"]
                d_out_w = af.T @ dz + self.alpha * self._out_w
                d_out_b = np.sum(dz, axis=0, keepdims=True)
                daf = dz @ self._out_w.T
                dzf = daf * (cache["zf"] > 0.0)
                d_fusion_w = concat.T @ dzf + self.alpha * self._fusion_w
                d_fusion_b = np.sum(dzf, axis=0, keepdims=True)
                dconcat = dzf @ self._fusion_w.T
                self._out_w = self._clip_param(self._out_w - lr * self._clip_grad(d_out_w))
                self._out_b = self._clip_param(self._out_b - lr * self._clip_grad(d_out_b))
                self._fusion_w = self._clip_param(self._fusion_w - lr * self._clip_grad(d_fusion_w))
                self._fusion_b = self._clip_param(self._fusion_b - lr * self._clip_grad(d_fusion_b))
                offset = 0
                for name, (idx, w, b) in list(self._branches.items()):
                    da = dconcat[:, offset:offset + 16]
                    offset += 16
                    _idx, z_branch, _a_branch = cache["branches"][name]
                    dz_branch = da * (z_branch > 0.0)
                    dw = Xs[:, idx].T @ dz_branch + self.alpha * w
                    db = np.sum(dz_branch, axis=0, keepdims=True)
                    self._branches[name] = (
                        idx,
                        self._clip_param(w - lr * self._clip_grad(dw)),
                        self._clip_param(b - lr * self._clip_grad(db)),
                    )
            self._restore(best_state)
        else:
            self._init_dense(Xs.shape[1], output_bias)
            best_loss = float("inf")
            best_state = self._snapshot()
            for _ in range(self.max_iter):
                z, activations, preacts = self._forward_dense(Xs)
                loss, delta = self._latent_loss_grad(z, latent_guess, weights)
                if np.isfinite(loss) and loss < best_loss:
                    best_loss = loss
                    best_state = self._snapshot()
                for layer in reversed(range(len(self._weights))):
                    a_prev = activations[layer]
                    dw = a_prev.T @ delta + self.alpha * self._weights[layer]
                    db = np.sum(delta, axis=0, keepdims=True)
                    if layer > 0:
                        delta = (delta @ self._weights[layer].T) * (preacts[layer - 1] > 0.0)
                    self._weights[layer] = self._clip_param(self._weights[layer] - lr * self._clip_grad(dw))
                    self._biases[layer] = self._clip_param(self._biases[layer] - lr * self._clip_grad(db))
            self._restore(best_state)
        return self

    def predict_latent(self, X: pd.DataFrame) -> np.ndarray:
        if self._sk_model is not None:
            if self._cols is None:
                raise ValueError("LatentM1PhysicsLayerRegressor is not fitted.")
            arr = X.reindex(columns=self._cols).fillna(0.0).to_numpy(dtype=float)
            latent_z = np.asarray(self._sk_model.predict(arr), dtype=float)
            latent = latent_z * self._sk_target_std + self._sk_target_mean
            return np.clip(np.nan_to_num(latent, nan=1.0, posinf=1e6, neginf=1e-8), 1e-8, 1e6)
        Xs = self._standardize_predict(X)
        if self.arch == "branched_hidden16":
            z, _cache = self._forward_branched(Xs)
        else:
            z, _activations, _preacts = self._forward_dense(Xs)
        latent = _softplus(z[:, 0]) + 1e-8
        return np.clip(np.nan_to_num(latent, nan=1.0, posinf=1e6, neginf=1e-8), 1e-8, 1e6).astype(float)

    def predict(self, X: pd.DataFrame, physics_shape) -> np.ndarray:
        g = np.asarray(physics_shape, dtype=float)
        pred = self.predict_latent(X) * g
        return np.nan_to_num(pred, nan=0.0, posinf=1e9, neginf=0.0)


class ExecutorGraphInterpreter:
    """Interpret the executable subset of executor_graph.

    v1 intentionally supports the three graph patterns that already have trusted
    semantics: raw_ml, mechanism_features, and mechanism_residual. The route is
    graph-driven rather than fusion-mode if/else: each fit/predict call walks the
    normalized graph and executes allowed ops against a fold-local context.
    """

    def __init__(
        self,
        graph: dict[str, Any],
        *,
        mechanism,
        model_family,
        random_state: int,
        hparams: dict[str, Any] | None = None,
        model_factory: Callable[[int, dict[str, Any]], Any] | None = None,
    ):
        self.graph = normalize_executor_graph(graph)
        audit = executor_graph_readiness(self.graph)
        if audit["status"] != "executable":
            raise ValueError(
                f"executor_graph '{self.graph.get('id')}' is not executable by v1 interpreter: "
                f"{audit['status']} {audit['reasons']}"
            )
        self.mode = str(audit["mode"])
        self.mechanism = mechanism
        self.model_family = model_family
        self.random_state = int(random_state)
        self.hparams = dict(hparams or {})
        self.model_factory = model_factory
        self.params_: dict[str, Any] | None = None
        self._raw_cols: list[str] | None = None
        self._model_cols: list[str] | None = None
        self.model_ = None
        self.latent_model_ = None
        self.low_fidelity_model_ = None
        self.high_fidelity_residual_model_ = None
        self._mf_fidelity_column = "data_fidelity"
        self._mf_low_label = "low_fidelity"
        self._mf_high_label = "high_fidelity"

    def _new_model(self, n_samples: int):
        if self.model_factory is not None:
            return self.model_factory(self.random_state, self.hparams)
        return _make_model(self.model_family, self.random_state, n_samples, hparams=self.hparams)

    def _model_input(self, feats: pd.DataFrame):
        # LLM-generated estimators may use DataFrame column names to implement
        # task-aware feature grouping. Fixed local models keep the historical
        # numpy path for stability.
        if self.model_factory is not None:
            return feats
        return feats.to_numpy(dtype=float)

    def _select_raw_features(self, X: pd.DataFrame, *, fitting: bool) -> pd.DataFrame:
        raw = _numeric_encode(X) if fitting else _numeric_encode(X, self._raw_cols)
        if fitting:
            self._raw_cols = list(raw.columns)
        return raw

    def _require_mechanism(self, op: str):
        if self.mechanism is None:
            raise ValueError(f"executor_graph op '{op}' requires a mechanism.")

    def _step_fidelity_config(self, step: dict[str, Any]) -> dict[str, str]:
        return {
            "column": str(step.get("fidelity_column") or self._mf_fidelity_column or "data_fidelity"),
            "low": str(step.get("low_fidelity_label") or self._mf_low_label or "low_fidelity").strip().lower(),
            "high": str(step.get("high_fidelity_label") or self._mf_high_label or "high_fidelity").strip().lower(),
        }

    @staticmethod
    def _fidelity_masks_for_X(X: pd.DataFrame, cfg: dict[str, str]) -> tuple[np.ndarray, np.ndarray]:
        column = cfg["column"]
        if column not in X.columns:
            raise ValueError(f"multi_fidelity_base_residual requires fidelity column '{column}'.")
        labels = X[column].astype(str).str.strip().str.lower()
        low_mask = labels.isin({cfg["low"], "lf", "low", "low_fidelity", "synthetic_low_fidelity"}).to_numpy(dtype=bool)
        high_mask = labels.isin({cfg["high"], "hf", "high", "high_fidelity", "real_hf"}).to_numpy(dtype=bool)
        return low_mask, high_mask

    def _new_hf_residual_model(self):
        alpha = float(self.hparams.get("hf_residual_alpha", 1.0))
        return _cap_estimator_complexity(make_pipeline(StandardScaler(), Ridge(alpha=alpha)))

    def _compute_mechanism_features(self, X: pd.DataFrame) -> pd.DataFrame:
        self._require_mechanism("compute_mechanism_features")
        if self.params_ is None:
            raise ValueError("mechanism parameters are not fitted.")
        return self.mechanism.features(X, self.params_)

    def _compute_mechanism_base(self, X: pd.DataFrame) -> np.ndarray:
        self._require_mechanism("compute_mechanism_base")
        if self.params_ is None:
            raise ValueError("mechanism parameters are not fitted.")
        return np.asarray(self.mechanism.base_predict(X, self.params_), dtype=float)

    def _compute_unit_physics_shape(self, X: pd.DataFrame) -> np.ndarray:
        self._require_mechanism("compute_physics_output")
        if self.params_ is None:
            raise ValueError("mechanism parameters are not fitted.")
        if not hasattr(self.mechanism, "base_predict"):
            raise ValueError("end_to_end_physics_layer requires mechanism.base_predict.")
        unit_params = dict(self.params_)
        unit_params["m1"] = 1.0
        shape = np.asarray(self.mechanism.base_predict(X, unit_params), dtype=float)
        return np.where(np.isfinite(shape), shape, 0.0)

    @staticmethod
    def _matrix_from_context(ctx: dict[str, Any], inputs: list[Any] | None, *, cols: list[str] | None = None) -> pd.DataFrame:
        names = [str(x) for x in (inputs or [])]
        frames: list[pd.DataFrame] = []
        if "raw_features" in names and isinstance(ctx.get("raw_features"), pd.DataFrame):
            frames.append(ctx["raw_features"])
        if "mechanism_features" in names and isinstance(ctx.get("mechanism_features"), pd.DataFrame):
            frames.append(ctx["mechanism_features"])
        if not frames and isinstance(ctx.get("raw_features"), pd.DataFrame):
            frames.append(ctx["raw_features"])
        if not frames:
            raise ValueError("executor_graph model step has no feature matrix in context.")
        Xmat = pd.concat(frames, axis=1)
        if cols is not None:
            Xmat = Xmat.reindex(columns=cols).fillna(0.0)
        return Xmat

    @staticmethod
    def _phase_allows(step: dict[str, Any], phase: str) -> bool:
        step_phase = str(step.get("phase") or "").strip().lower()
        if not step_phase:
            # Legacy/minimal graph steps are interpreted in both phases; fit-only
            # and predict-only op constraints were already audited upstream.
            return True
        if step_phase == "fit_predict":
            return True
        return step_phase == phase

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        y = np.asarray(y, dtype=float)
        ctx: dict[str, Any] = {"train_X": X, "X": X, "train_y": y}
        for step in self.graph.get("steps", []):
            if not self._phase_allows(step, "fit"):
                continue
            op = str(step.get("op") or "")
            if op == "select_raw_features":
                ctx["raw_features"] = self._select_raw_features(X, fitting=True)
            elif op == "fit_mechanism_params":
                self._require_mechanism(op)
                self.params_ = self.mechanism.fit_global(X, y)
                ctx["mechanism_params"] = self.params_
            elif op == "compute_mechanism_features":
                ctx["mechanism_features"] = self._compute_mechanism_features(X)
            elif op == "compute_mechanism_base":
                ctx["physics_base"] = self._compute_mechanism_base(X)
            elif op == "fit_latent_physics_model":
                feats = self._matrix_from_context(ctx, step.get("inputs"))
                self._model_cols = list(feats.columns)
                physics_shape = ctx.get("physics_shape")
                if physics_shape is None:
                    physics_shape = self._compute_unit_physics_shape(X)
                    ctx["physics_shape"] = physics_shape
                if self.model_factory is not None:
                    self.latent_model_ = self.model_factory(self.random_state, self.hparams)
                    if not hasattr(self.latent_model_, "fit") or not hasattr(self.latent_model_, "predict_latent"):
                        raise ValueError(
                            "LLM latent-physics model factory must return an object with "
                            "fit(X, y, physics_shape=...) and predict_latent(X)."
                        )
                else:
                    arch = str(step.get("arch") or self.hparams.get("latent_arch") or self.model_family)
                    if arch not in _LATENT_ARCHS:
                        arch = "mlp2_hidden16"
                    self.latent_model_ = LatentM1PhysicsLayerRegressor(
                        arch=arch,
                        random_state=self.random_state,
                        latent_backend=str(step.get("latent_backend", self.hparams.get("latent_backend", "auto"))),
                        learning_rate=float(step.get("learning_rate", self.hparams.get("learning_rate", 0.01))),
                        max_iter=int(step.get("max_iter", self.hparams.get("max_iter", 500))),
                        alpha=float(step.get("alpha", self.hparams.get("alpha", 1e-4))),
                    )
                self.latent_model_.fit(feats, y, physics_shape)
                self.model_ = self.latent_model_
                ctx["latent_physics_model"] = self.latent_model_
            elif op == "fit_low_fidelity_base":
                cfg = self._step_fidelity_config(step)
                self._mf_fidelity_column = cfg["column"]
                self._mf_low_label = cfg["low"]
                self._mf_high_label = cfg["high"]
                feats = self._matrix_from_context(ctx, step.get("inputs"))
                low_mask, high_mask = self._fidelity_masks_for_X(X, cfg)
                if int(low_mask.sum()) < 1 or int(high_mask.sum()) < 1:
                    raise ValueError(
                        "multi_fidelity_base_residual needs at least one LF row and one HF row in the train fold."
                    )
                self._model_cols = list(feats.columns)
                self.low_fidelity_model_ = self._new_model(int(low_mask.sum()))
                self.low_fidelity_model_.fit(
                    self._model_input(feats.iloc[low_mask]),
                    y[low_mask],
                )
                ctx["low_fidelity_model"] = self.low_fidelity_model_
            elif op == "fit_high_fidelity_residual":
                cfg = self._step_fidelity_config(step)
                feats = self._matrix_from_context(ctx, step.get("inputs"), cols=self._model_cols)
                _low_mask, high_mask = self._fidelity_masks_for_X(X, cfg)
                if self.low_fidelity_model_ is None:
                    raise ValueError("fit_high_fidelity_residual requires a fitted low_fidelity_model.")
                if int(high_mask.sum()) < 2:
                    raise ValueError("fit_high_fidelity_residual needs at least two HF rows in the train fold.")
                base_hf = np.asarray(
                    self.low_fidelity_model_.predict(self._model_input(feats.iloc[high_mask])),
                    dtype=float,
                )
                self.high_fidelity_residual_model_ = self._new_hf_residual_model()
                self.high_fidelity_residual_model_.fit(
                    feats.iloc[high_mask].to_numpy(dtype=float),
                    y[high_mask] - base_hf,
                )
                self.model_ = self.high_fidelity_residual_model_
                ctx["high_fidelity_residual_model"] = self.high_fidelity_residual_model_
            elif op in {"fit_model", "fit_residual_model"}:
                feats = self._matrix_from_context(ctx, step.get("inputs"))
                self._model_cols = list(feats.columns)
                self.model_ = self._new_model(len(X))
                if op == "fit_residual_model":
                    base = ctx.get("physics_base")
                    if base is None:
                        raise ValueError("fit_residual_model requires physics_base.")
                    if str(self.model_family).lower() == "pinn":
                        self.model_.fit(feats.to_numpy(dtype=float), y, physics_base=np.asarray(base, dtype=float))
                    else:
                        self.model_.fit(self._model_input(feats), y - np.asarray(base, dtype=float))
                else:
                    self.model_.fit(self._model_input(feats), y)
                ctx["model"] = self.model_
                ctx["residual_model" if op == "fit_residual_model" else "model"] = self.model_
        if self.model_ is None:
            raise ValueError("executor_graph fit completed without fitting a model.")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model_ is None:
            raise ValueError("executor_graph interpreter is not fitted.")
        ctx: dict[str, Any] = {"X": X, "model": self.model_, "residual_model": self.model_}
        for step in self.graph.get("steps", []):
            if not self._phase_allows(step, "predict"):
                continue
            op = str(step.get("op") or "")
            if op == "select_raw_features":
                ctx["raw_features"] = self._select_raw_features(X, fitting=False)
            elif op == "compute_mechanism_features":
                ctx["mechanism_features"] = self._compute_mechanism_features(X)
            elif op == "compute_mechanism_base":
                ctx["physics_base"] = self._compute_mechanism_base(X)
            elif op == "predict_latent_physics_model":
                if self.latent_model_ is None:
                    raise ValueError("predict_latent_physics_model requires a fitted latent physics model.")
                feats = self._matrix_from_context(ctx, step.get("inputs"), cols=self._model_cols)
                ctx["latent_prediction"] = self.latent_model_.predict_latent(feats)
            elif op == "compute_physics_output":
                latent = ctx.get("latent_prediction")
                if latent is None:
                    raise ValueError("compute_physics_output requires latent_prediction.")
                physics_shape = self._compute_unit_physics_shape(X)
                ctx["physics_shape"] = physics_shape
                ctx["physics_output"] = np.nan_to_num(
                    np.asarray(latent, dtype=float) * physics_shape,
                    nan=0.0,
                    posinf=1e9,
                    neginf=0.0,
                )
                ctx["prediction"] = ctx["physics_output"]
            elif op == "predict_model":
                feats = self._matrix_from_context(ctx, step.get("inputs"), cols=self._model_cols)
                if self.mode == "multi_fidelity_base_residual":
                    if self.low_fidelity_model_ is None:
                        raise ValueError("multi_fidelity_base_residual prediction requires a fitted low_fidelity_model.")
                    ctx["physics_base"] = np.asarray(
                        self.low_fidelity_model_.predict(self._model_input(feats)),
                        dtype=float,
                    )
                else:
                    ctx["prediction"] = np.asarray(self.model_.predict(self._model_input(feats)), dtype=float)
            elif op == "predict_correction_model":
                feats = self._matrix_from_context(ctx, step.get("inputs"), cols=self._model_cols)
                if self.high_fidelity_residual_model_ is None:
                    raise ValueError("predict_correction_model requires a fitted high_fidelity_residual_model.")
                ctx["residual_prediction"] = np.asarray(
                    self.high_fidelity_residual_model_.predict(feats.to_numpy(dtype=float)),
                    dtype=float,
                )
            elif op == "predict_residual_model":
                feats = self._matrix_from_context(ctx, step.get("inputs"), cols=self._model_cols)
                base = ctx.get("physics_base")
                if str(self.model_family).lower() == "pinn":
                    if base is None:
                        raise ValueError("PINN residual prediction requires physics_base.")
                    full = np.asarray(
                        self.model_.predict(feats.to_numpy(dtype=float), physics_base=np.asarray(base, dtype=float)),
                        dtype=float,
                    )
                    ctx["residual_prediction"] = full - np.asarray(base, dtype=float)
                else:
                    ctx["residual_prediction"] = np.asarray(
                        self.model_.predict(self._model_input(feats)), dtype=float
                    )
            elif op == "add_predictions":
                base = ctx.get("physics_base")
                residual = ctx.get("residual_prediction")
                if base is None or residual is None:
                    raise ValueError("add_predictions requires physics_base and residual_prediction.")
                ctx["prediction"] = np.asarray(base, dtype=float) + np.asarray(residual, dtype=float)
        if "prediction" not in ctx:
            raise ValueError("executor_graph predict completed without prediction.")
        return np.asarray(ctx["prediction"], dtype=float)


class MechanismModelCandidate:
    """Runnable estimator for one (mechanism, model, executor_graph) triple.

    Supported v1 graph patterns:
      * raw_ml: raw numeric features -> model.
      * mechanism_features: raw numeric features + fold-local mechanism features
        -> model.
      * mechanism_residual: fold-local mechanism base predicts tau_base; model
        learns y - tau_base from raw features; final prediction is tau_base plus
        the learned residual.

    Anti-leakage: the mechanism's params are fit inside fit() against the TRAINING
    fold only (mechanism.fit_global sees train y); features()/predict never see y.
    """

    def __init__(self, mechanism, model_family, random_state: int = 0, nonnegative: bool = True,
                 hparams: dict[str, Any] | None = None,
                 fusion_mode: str = "mechanism_features",
                 model_factory: Callable[[int, dict[str, Any]], Any] | None = None,
                 executor_graph: dict[str, Any] | None = None):
        self.mechanism = mechanism
        self.model_family = model_family
        self.fusion_mode = _normalize_fusion_mode(fusion_mode)
        self.executor_graph = normalize_executor_graph(
            executor_graph or builtin_executor_graph(self.fusion_mode)
        )
        self.random_state = int(random_state)
        self.nonnegative = nonnegative
        self.hparams = dict(hparams or {})
        self.model_factory = model_factory
        self.params_: dict[str, Any] | None = None
        self._raw_cols: list[str] | None = None
        self._cols: list[str] | None = None
        self.model_ = None
        self.interpreter_: ExecutorGraphInterpreter | None = None

    def fit(self, X: pd.DataFrame, y):
        self.interpreter_ = ExecutorGraphInterpreter(
            self.executor_graph,
            mechanism=self.mechanism,
            model_family=self.model_family,
            random_state=self.random_state,
            hparams=self.hparams,
            model_factory=self.model_factory,
        )
        self.interpreter_.fit(X, np.asarray(y, dtype=float))
        self.params_ = self.interpreter_.params_
        self._raw_cols = self.interpreter_._raw_cols
        self._cols = self.interpreter_._model_cols
        self.model_ = self.interpreter_.model_
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.interpreter_ is None:
            raise ValueError("MechanismModelCandidate is not fitted.")
        pred = self.interpreter_.predict(X)
        return np.clip(pred, 0.0, None) if self.nonnegative else pred


def make_candidate_builder(mech_by_id: dict[str, Any], model_by_id: dict[str, Any] | None = None,
                           *, random_state: int = 0):
    """Production candidate_builder for run_free_search.

    Resolves each admitted triple {mechanism: id|None, model_family,
    fusion_mode} into MechanismModelCandidate objects. `mech_by_id` maps
    mechanism id -> Mechanism-like object (seed Mechanism or LLM-built
    DynamicMechanism). A triple naming an unknown mechanism id raises (recorded
    as a build error, never silently downgraded to raw).
    """
    model_by_id = dict(model_by_id or {})

    def builder(entry: dict[str, Any]) -> list[Candidate]:
        mech_id = entry.get("mechanism")
        if mech_id is None:
            mech = None
        elif mech_id in mech_by_id:
            mech = mech_by_id[mech_id]
        else:
            raise KeyError(f"pair names unknown mechanism id '{mech_id}'")
        fam = entry.get("model_family")
        fusion_spec = normalize_fusion_spec(
            entry.get("fusion_spec")
            or entry.get("fusion_id")
            or entry.get("fusion_mode")
            or ("raw_ml" if mech_id is None else "mechanism_features")
        )
        mode = builtin_fusion_mode(fusion_spec)
        fusion_id = str(fusion_spec.get("id") or mode)
        executor_graph = (
            (fusion_spec.get("executor_spec") or {}).get("executor_graph")
            or builtin_executor_graph(mode, graph_id=f"{fusion_id}_graph")
        )
        model_obj = model_by_id.get(str(fam))
        grid = model_obj.hparam_grid() if model_obj is not None else hparam_grid(fam)
        cands: list[Candidate] = []
        for i, hp in enumerate(grid):
            tag = f"#{i}" if len(grid) > 1 else ""
            model_kind = getattr(model_obj, "model_kind", None) if model_obj is not None else None
            if model_obj is not None:
                factory = lambda m=mech, f=fam, h=hp, mo=model_obj, fm=mode, g=executor_graph: MechanismModelCandidate(
                    m, f, random_state=random_state, hparams=h, fusion_mode=fm,
                    model_factory=mo.make_estimator, executor_graph=g)
                origin = "llm_model"
            else:
                factory = lambda m=mech, f=fam, h=hp, fm=mode, g=executor_graph: MechanismModelCandidate(
                    m, f, random_state=random_state, hparams=h, fusion_mode=fm, executor_graph=g)
                origin = "fixed_family"
            cands.append(Candidate(
                f"{mech_id or 'raw'}::{fam}::{fusion_id}{tag}", factory,
                spec={"mechanism": mech_id, "model_family": fam, "hparams": hp,
                      "fusion_mode": mode, "fusion_id": fusion_id, "fusion_spec": fusion_spec,
                      "executor_graph": executor_graph,
                      "model_origin": origin, "model_kind": model_kind,
                      "status": entry.get("status")}))
        return cands
    return builder
