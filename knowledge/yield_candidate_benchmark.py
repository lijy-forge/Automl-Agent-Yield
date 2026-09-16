"""Lightweight candidate benchmarking for yield-stress model selection.

This module does not define the final research model. It runs cheap,
deterministic proxy evaluations for LLM-proposed candidate strategies so the
manager can select with data evidence instead of prompt preference alone.
"""

from __future__ import annotations

import math
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.impute import SimpleImputer
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from knowledge.yield_baselines import run_fixed_yield_baselines
from knowledge.yield_schema import INDEX_COLUMN, TARGET_COLUMN, load_yield_dataframe


LEAKAGE_COLUMNS = {
    TARGET_COLUMN,
    "yield_stress",
    "Tau0_Pa",
    "tau_Pa",
    "target",
    "label",
    "phi_max",
    "phi_max_eff",
    "phi_max_eff_reference",
    "phi_max_reference",
    "m1_true",
    "m1_lf",
    "m1_true_reference",
    "m1_lf_reference",
    "plastic_viscosity",
    "plastic_viscosity_pa_s",
    "source",
    "data_fidelity",
}


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else None


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.maximum(np.abs(y_true), 1e-8)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0.0, None)
    y_true = np.asarray(y_true, dtype=float)
    r2 = None
    if len(y_true) >= 2 and np.nanstd(y_true) > 1e-12:
        r2 = float(r2_score(y_true, y_pred))
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": r2,
        "mape": _mape(y_true, y_pred),
    }


def _candidate_text(item: dict[str, Any], allowed_keys: set[str] | None = None) -> str:
    chunks: list[str] = []
    values = (
        [item.get(key) for key in allowed_keys if key in item]
        if allowed_keys is not None
        else list(item.values())
    )
    for value in values:
        if isinstance(value, (str, int, float, bool)):
            chunks.append(str(value))
        elif isinstance(value, (list, tuple, set)):
            chunks.extend(str(x) for x in value)
        elif isinstance(value, dict):
            chunks.extend(
                str(x)
                for key, x in value.items()
                if allowed_keys is None or key in allowed_keys
            )
    return " ".join(chunks).lower()


def _feature_columns(meta: dict[str, Any], df: pd.DataFrame) -> list[str]:
    columns = list(meta.get("feature_columns") or [])
    if not columns:
        columns = [
            str(col)
            for col in df.columns
            if col not in LEAKAGE_COLUMNS
            and col != INDEX_COLUMN
            and not str(col).endswith("_reference")
        ]
    return [
        str(col)
        for col in columns
        if col in df.columns
        and col not in LEAKAGE_COLUMNS
        and col != INDEX_COLUMN
        and not str(col).endswith("_reference")
    ]


def _clean_feature_list(columns: list[Any], available_columns: list[str]) -> list[str]:
    available = set(str(col) for col in available_columns)
    cleaned: list[str] = []
    for item in columns or []:
        if isinstance(item, dict):
            item = item.get("name") or item.get("column") or item.get("feature")
        col = str(item).strip()
        if (
            col
            and col in available
            and col not in LEAKAGE_COLUMNS
            and col != INDEX_COLUMN
            and not col.endswith("_reference")
            and col not in cleaned
        ):
            cleaned.append(col)
    return cleaned


def _strategy_feature_columns(
    strategy: dict[str, Any],
    model: dict[str, Any] | None,
    feature_columns: list[str],
    anchor_df: pd.DataFrame | None,
) -> tuple[list[str], dict[str, Any]]:
    """Resolve the feature set that a strategy actually asked to evaluate.

    CandidateAgent strategies are allowed to propose different feature subsets.
    The benchmark must honor that contract; otherwise an anchor-compatible
    strategy can be unfairly evaluated with full synthetic-only process fields.
    """
    requested_raw = (
        strategy.get("feature_columns")
        or strategy.get("required_columns")
        or strategy.get("input_columns")
        or strategy.get("features")
        or []
    )
    requested = _clean_feature_list(list(requested_raw) if isinstance(requested_raw, list) else [], feature_columns)
    if not requested and isinstance(model, dict):
        model_raw = model.get("feature_columns") or model.get("required_columns") or []
        requested = _clean_feature_list(list(model_raw) if isinstance(model_raw, list) else [], feature_columns)
    if not requested:
        requested = list(feature_columns)

    text = _candidate_text(
        strategy,
        {
            "id",
            "name",
            "combination_method",
            "required_columns",
            "data_support_reason",
            "implementation_plan",
            "expected_benefit",
            "risks",
        },
    )
    anchor_shared = []
    if anchor_df is not None and len(anchor_df) > 0:
        anchor_shared = [col for col in requested if col in anchor_df.columns]
        if any(token in text for token in ("anchor-compatible", "anchor compatible", "shared-feature", "shared feature", "table 6")):
            requested = anchor_shared or requested

    excluded = [col for col in list(feature_columns) if col not in requested]
    return requested, {
        "strategy_requested_columns": requested,
        "anchor_shared_requested_columns": anchor_shared,
        "excluded_available_columns": excluded,
        "resolution_policy": (
            "Honored strategy.required_columns/feature_columns when provided; "
            "anchor-compatible/shared-feature strategies are restricted to columns present in the anchor dataframe."
        ),
    }


def _fit_packing_params(train_df: pd.DataFrame) -> dict[str, float]:
    phi = pd.to_numeric(train_df.get("phi"), errors="coerce")
    finite_phi = phi[np.isfinite(phi)]
    max_phi = float(finite_phi.max()) if len(finite_phi) else 0.50
    base_phi_m = min(0.72, max(0.56, max_phi + 0.055))

    def stat(name: str, default: float = 0.0) -> tuple[float, float]:
        if name not in train_df.columns:
            return default, 1.0
        values = pd.to_numeric(train_df[name], errors="coerce").to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if len(values) == 0:
            return default, 1.0
        median = float(np.median(values))
        q75, q25 = np.percentile(values, [75, 25])
        scale = float(max(q75 - q25, np.std(values), 1e-6))
        return median, scale

    sp_median, sp_scale = stat("sp_percent", 0.0)
    psd_median, psd_scale = stat("psd_width", 1.0)
    mix_median, mix_scale = stat("mixing_energy", 1.0)
    temp_median, temp_scale = stat("temperature_c", 25.0)
    phi_c = float(max(0.20, min(0.42, np.nanpercentile(finite_phi, 10) - 0.025))) if len(finite_phi) else 0.33
    return {
        "base_phi_m": base_phi_m,
        "phi_c": phi_c,
        "sp_median": sp_median,
        "sp_scale": sp_scale,
        "psd_median": psd_median,
        "psd_scale": psd_scale,
        "mix_median": mix_median,
        "mix_scale": mix_scale,
        "temp_median": temp_median,
        "temp_scale": temp_scale,
    }


def _add_packing_features(features: pd.DataFrame, params: dict[str, float]) -> pd.DataFrame:
    out = features.copy()
    if "phi" not in out.columns:
        return out
    phi = pd.to_numeric(out["phi"], errors="coerce").astype(float)
    phi_m = np.full(len(out), float(params["base_phi_m"]), dtype=float)
    if "sp_percent" in out.columns:
        sp = pd.to_numeric(out["sp_percent"], errors="coerce").fillna(params["sp_median"]).astype(float)
        phi_m += 0.025 * np.tanh((sp - params["sp_median"]) / max(params["sp_scale"], 1e-6))
    if "psd_width" in out.columns:
        psd = pd.to_numeric(out["psd_width"], errors="coerce").fillna(params["psd_median"]).astype(float)
        phi_m += 0.015 * np.tanh((psd - params["psd_median"]) / max(params["psd_scale"], 1e-6))
    if "mixing_energy" in out.columns:
        mix = pd.to_numeric(out["mixing_energy"], errors="coerce").fillna(params["mix_median"]).astype(float)
        phi_m += 0.010 * np.tanh((mix - params["mix_median"]) / max(params["mix_scale"], 1e-6))
    if "temperature_c" in out.columns:
        temp = pd.to_numeric(out["temperature_c"], errors="coerce").fillna(params["temp_median"]).astype(float)
        phi_m -= 0.004 * np.tanh((temp - params["temp_median"]) / max(params["temp_scale"], 1e-6))
    phi_m = np.clip(phi_m, phi.to_numpy(dtype=float) + 0.025, 0.74)
    gap = np.maximum(phi_m - phi.to_numpy(dtype=float), 0.025)
    active_phi = np.maximum(phi.to_numpy(dtype=float) - float(params["phi_c"]), 0.0)
    out["bench_phi_m_eff_proxy"] = phi_m
    out["bench_packing_gap"] = gap
    out["bench_packing_index"] = np.power(active_phi, 2.0) / gap
    out["bench_phi_over_gap"] = phi.to_numpy(dtype=float) / gap
    return out


def _add_sp_features(features: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    out = features.copy()
    if "sp_percent" not in out.columns:
        return out
    train_sp = pd.to_numeric(train_df["sp_percent"], errors="coerce")
    center = float(np.nanmedian(train_sp)) if np.isfinite(train_sp).any() else 0.0
    scale = float(max(np.nanpercentile(train_sp, 75) - np.nanpercentile(train_sp, 25), np.nanstd(train_sp), 1e-6))
    sp = pd.to_numeric(out["sp_percent"], errors="coerce").fillna(center).astype(float)
    out["bench_sp_decay"] = np.exp(-np.maximum(sp, 0.0))
    out["bench_sp_centered"] = (sp - center) / scale
    return out


def _make_raw_features(
    df: pd.DataFrame,
    feature_columns: list[str],
    mode: str,
    train_df_for_params: pd.DataFrame,
) -> pd.DataFrame:
    features = pd.DataFrame(index=df.index)
    for col in feature_columns:
        if col in df.columns:
            features[col] = df[col]
        else:
            features[col] = np.nan
    if mode in {"packing", "packing_sp"}:
        features = _add_packing_features(features, _fit_packing_params(train_df_for_params))
    if mode in {"sp_decay", "packing_sp"}:
        features = _add_sp_features(features, train_df_for_params)
    return features


def _encode_train_valid(train_features: pd.DataFrame, valid_features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    train_encoded = pd.get_dummies(train_features, dummy_na=False)
    valid_encoded = pd.get_dummies(valid_features, dummy_na=False)
    valid_encoded = valid_encoded.reindex(columns=train_encoded.columns, fill_value=0.0)
    train_encoded = train_encoded.apply(pd.to_numeric, errors="coerce")
    valid_encoded = valid_encoded.apply(pd.to_numeric, errors="coerce")
    imputer = SimpleImputer(strategy="median")
    x_train = imputer.fit_transform(train_encoded)
    x_valid = imputer.transform(valid_encoded)
    return x_train.astype(float), x_valid.astype(float), list(train_encoded.columns)


def _make_model(proxy_model: str, random_state: int, n_samples: int) -> Any:
    if proxy_model == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    if proxy_model == "kernel":
        if n_samples <= 260:
            kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=1.5) + WhiteKernel(noise_level=1e-3)
            return make_pipeline(
                StandardScaler(),
                GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True, random_state=random_state),
            )
        return make_pipeline(StandardScaler(), KernelRidge(alpha=1.0, kernel="rbf", gamma=0.5))
    if proxy_model == "mlp":
        return make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(32, 16),
                activation="relu",
                alpha=1e-3,
                learning_rate_init=0.004,
                max_iter=220,
                early_stopping=True,
                random_state=random_state,
            ),
        )
    if proxy_model == "rf":
        return RandomForestRegressor(
            n_estimators=120,
            max_depth=8,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        )
    return HistGradientBoostingRegressor(
        max_iter=100,
        max_leaf_nodes=24,
        learning_rate=0.08,
        l2_regularization=1e-3,
        random_state=random_state,
    )


def _mechanism_kind_from_item(mechanism: dict[str, Any]) -> str:
    audit = mechanism.get("data_capability_audit")
    if isinstance(audit, dict) and audit.get("mechanism_kind"):
        return str(audit.get("mechanism_kind") or "").lower()
    text = _candidate_text(
        mechanism,
        {
            "id",
            "name",
            "paper_or_source",
            "formula_or_relationship",
            "required_columns",
            "role_options",
            "applicability_to_current_data",
        },
    )
    if any(
        token in text
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
    if any(token in text for token in ("lian", "formulation", "superplasticizer", "dispersant", "pce", "sp_percent", "additive")):
        return "lian_formulation_packing"
    if any(token in text for token in ("herschel", "bingham", "casson", "constitutive")):
        return "herschel_bulkley_bingham"
    return "unknown"


def _strategy_includes_yodel(strategy: dict[str, Any], mechanisms_by_id: dict[str, dict[str, Any]]) -> bool:
    for mech_id in strategy.get("mechanism_ids", []) or []:
        mechanism = mechanisms_by_id.get(str(mech_id))
        if isinstance(mechanism, dict) and _mechanism_kind_from_item(mechanism) == "yodel_packing_fmax":
            return True
    text = _candidate_text(
        strategy,
        {"id", "name", "combination_method", "required_columns", "data_support_reason", "implementation_plan"},
    )
    return any(token in text for token in ("yodel", "packing", "fmax", "phi_m", "maximum packing", "jamming"))


def _classify_strategy(
    strategy: dict[str, Any],
    mechanisms_by_id: dict[str, dict[str, Any]],
    models_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    strategy_text = _candidate_text(
        strategy,
        {
            "id",
            "name",
            "model_id",
            "mechanism_ids",
            "combination_method",
            "required_columns",
            "data_support",
            "data_support_reason",
            "implementation_plan",
            "expected_benefit",
            "risks",
        },
    )
    model = models_by_id.get(str(strategy.get("model_id") or ""))
    model_text = (
        _candidate_text(
            model,
            {"id", "name", "family", "modeling_idea", "required_columns", "strengths", "risks"},
        )
        if isinstance(model, dict)
        else ""
    )
    mechanism_text = ""
    mechanism_kinds: list[str] = []
    for mech_id in strategy.get("mechanism_ids", []) or []:
        mech = mechanisms_by_id.get(str(mech_id))
        if isinstance(mech, dict):
            mechanism_kinds.append(_mechanism_kind_from_item(mech))
            mechanism_text += " " + _candidate_text(
                mech,
                {
                    "id",
                    "name",
                    "paper_or_source",
                    "formula_or_relationship",
                    "required_columns",
                    "role_options",
                    "applicability_to_current_data",
                },
            )
    text = " ".join([strategy_text, model_text, mechanism_text])

    uses_packing = "yodel_packing_fmax" in mechanism_kinds or any(
        token in strategy_text + " " + mechanism_text
        for token in (
            "yodel",
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
    )
    uses_hb = any(token in text for token in ("herschel", "bingham", "casson", "constitutive"))
    uses_kernel = any(token in text for token in ("gaussian", "kernel", "kriging", "gpr"))
    uses_mlp = any(token in text for token in ("neural", "pinn", "mlp", "deep"))
    uses_linear = any(token in text for token in ("linear", "ridge", "symbolic", "sparse"))
    uses_rf = "random forest" in text or "random_forest" in text
    uses_boosting = any(token in text for token in ("xgboost", "gradient boost", "gradient_boost", "histgradient", "hist gradient", "boosted tree", "boosting"))
    uses_sp = "lian_formulation_packing" in mechanism_kinds or any(
        token in text for token in ("superplasticizer", "dispersant", "sp_percent", "pce", "additive")
    )

    mode = "base"
    if uses_packing and uses_sp:
        mode = "packing_sp"
    elif uses_packing:
        mode = "packing"
    elif uses_sp:
        mode = "sp_decay"

    if uses_boosting:
        proxy_model = "hgb"
    elif uses_kernel:
        proxy_model = "kernel"
    elif uses_mlp:
        proxy_model = "mlp"
    elif uses_linear:
        proxy_model = "ridge"
    elif uses_rf:
        proxy_model = "rf"
    else:
        proxy_model = "hgb"

    warnings = []
    if uses_hb:
        warnings.append("HB/Bingham/Casson wording detected; this proxy cannot validate strict constitutive loss without shear-rate data.")
    return {
        "feature_mode": mode,
        "proxy_model": proxy_model,
        "mechanism_kinds": mechanism_kinds,
        "uses_packing_proxy": uses_packing,
        "uses_hb_like_text": uses_hb,
        "warnings": warnings,
    }


def _evaluate_proxy(
    df: pd.DataFrame,
    feature_columns: list[str],
    strategy_id: str,
    feature_mode: str,
    proxy_model: str,
    *,
    random_state: int,
    n_splits: int,
    anchor_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    y = pd.to_numeric(df[TARGET_COLUMN], errors="coerce").to_numpy(dtype=float)
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_pred = np.zeros(len(df), dtype=float)
    fold_rows: list[dict[str, Any]] = []
    feature_names: list[str] = []

    for fold_idx, (train_idx, valid_idx) in enumerate(splitter.split(df), start=1):
        train_df = df.iloc[train_idx].copy()
        valid_df = df.iloc[valid_idx].copy()
        train_features = _make_raw_features(train_df, feature_columns, feature_mode, train_df)
        valid_features = _make_raw_features(valid_df, feature_columns, feature_mode, train_df)
        x_train, x_valid, feature_names = _encode_train_valid(train_features, valid_features)
        model = _make_model(proxy_model, random_state + fold_idx, len(train_df))
        model.fit(x_train, y[train_idx])
        pred = np.clip(np.asarray(model.predict(x_valid), dtype=float), 0.0, None)
        oof_pred[valid_idx] = pred
        fold_rows.append({"fold": fold_idx, **_metrics(y[valid_idx], pred)})

    oof = _metrics(y, oof_pred)
    result: dict[str, Any] = {
        "strategy_id": strategy_id,
        "benchmark_status": "evaluated",
        "proxy_model": proxy_model,
        "feature_mode": feature_mode,
        "feature_columns_used": feature_names,
        "primary_evaluation": "5-fold OOF" if n_splits == 5 else f"{n_splits}-fold OOF",
        "oof_metrics": oof,
        "fold_metrics": fold_rows,
    }

    if anchor_df is not None and len(anchor_df) > 0:
        train_features = _make_raw_features(df, feature_columns, feature_mode, df)
        anchor_features = _make_raw_features(anchor_df, feature_columns, feature_mode, df)
        x_train, x_anchor, anchor_feature_names = _encode_train_valid(train_features, anchor_features)
        model = _make_model(proxy_model, random_state + 991, len(df))
        model.fit(x_train, y)
        y_anchor = pd.to_numeric(anchor_df[TARGET_COLUMN], errors="coerce").to_numpy(dtype=float)
        anchor_pred = np.clip(np.asarray(model.predict(x_anchor), dtype=float), 0.0, None)
        result["secondary_evaluation"] = "anchor_validation"
        result["anchor_validation"] = _metrics(y_anchor, anchor_pred)
        result["anchor_feature_columns_used"] = anchor_feature_names
    return result


def _evaluate_baseline_summary(df: pd.DataFrame, feature_columns: list[str], n_splits: int, random_state: int) -> dict[str, Any]:
    try:
        bundle = run_fixed_yield_baselines(
            df,
            feature_columns,
            n_splits=n_splits,
            n_repeats=1,
            random_state=random_state,
            enable_sp_monotonic="sp_percent" in feature_columns,
        )
        rows = list(bundle.get("baseline_results") or [])
        best = min(rows, key=lambda row: float(row.get("oof_rmse", row.get("mean_rmse", float("inf"))))) if rows else None
        return {
            "status": "evaluated",
            "baseline_protocol": bundle.get("baseline_protocol"),
            "best_baseline": best,
            "baseline_results": rows,
        }
    except Exception as exc:
        return {
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "baseline_results": [],
        }


def _anchor_generalization_summary(row: dict[str, Any]) -> dict[str, Any]:
    oof = row.get("oof_metrics") if isinstance(row.get("oof_metrics"), dict) else {}
    anchor = row.get("anchor_validation") if isinstance(row.get("anchor_validation"), dict) else {}
    oof_r2 = _safe_float(oof.get("r2"))
    anchor_r2 = _safe_float(anchor.get("r2"))
    gap = None if oof_r2 is None or anchor_r2 is None else float(anchor_r2 - oof_r2)
    if anchor_r2 is None:
        status = "no_anchor"
        reason = "No anchor validation was available for this benchmark row."
    elif anchor_r2 < 0:
        status = "failed"
        reason = "Anchor R2 is negative; proxy generalization from training data to anchor data failed."
    elif anchor_r2 < 0.3:
        status = "weak"
        reason = "Anchor R2 is positive but weak; treat synthetic-data OOF evidence cautiously."
    elif anchor_r2 < 0.6:
        status = "moderate"
        reason = "Anchor R2 is moderate; report remaining synthetic-to-anchor uncertainty."
    else:
        status = "strong"
        reason = "Anchor validation is reasonably aligned with OOF evidence."
    return {
        "oof_r2": oof_r2,
        "anchor_r2": anchor_r2,
        "anchor_minus_oof_r2": gap,
        "status": status,
        "reason": reason,
    }


def _feature_shift_summary(
    train_df: pd.DataFrame,
    anchor_df: pd.DataFrame | None,
    feature_columns: list[str],
) -> dict[str, Any]:
    if anchor_df is None or len(anchor_df) == 0:
        return {
            "status": "no_anchor",
            "reason": "No anchor dataframe was available for synthetic-anchor distribution audit.",
        }

    shared = [col for col in feature_columns if col in train_df.columns and col in anchor_df.columns]
    train_only = [col for col in feature_columns if col in train_df.columns and col not in anchor_df.columns]
    anchor_only = [
        str(col)
        for col in anchor_df.columns
        if col not in train_df.columns
        and col not in LEAKAGE_COLUMNS
        and col != INDEX_COLUMN
        and not str(col).endswith("_reference")
    ]

    def stats(series: pd.Series) -> dict[str, float | None]:
        values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if len(values) == 0:
            return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
        return {
            "count": int(len(values)),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

    feature_rows: list[dict[str, Any]] = []
    severe_shift = 0
    moderate_shift = 0
    for col in shared:
        train_stats = stats(train_df[col])
        anchor_stats = stats(anchor_df[col])
        if not train_stats["count"] or not anchor_stats["count"]:
            feature_rows.append(
                {
                    "feature": col,
                    "status": "non_numeric_or_missing",
                    "train": train_stats,
                    "anchor": anchor_stats,
                }
            )
            continue
        train_std = float(train_stats["std"] or 0.0)
        anchor_mean = float(anchor_stats["mean"] or 0.0)
        train_mean = float(train_stats["mean"] or 0.0)
        mean_shift_std = abs(anchor_mean - train_mean) / max(train_std, 1e-8)
        train_min = float(train_stats["min"] or 0.0)
        train_max = float(train_stats["max"] or 0.0)
        anchor_min = float(anchor_stats["min"] or 0.0)
        anchor_max = float(anchor_stats["max"] or 0.0)
        range_overlap = max(0.0, min(train_max, anchor_max) - max(train_min, anchor_min))
        train_range = max(train_max - train_min, 1e-8)
        anchor_range = max(anchor_max - anchor_min, 1e-8)
        overlap_ratio = range_overlap / max(train_range, anchor_range)
        status = "ok"
        if mean_shift_std > 2.0 or overlap_ratio < 0.25:
            status = "severe_shift"
            severe_shift += 1
        elif mean_shift_std > 1.0 or overlap_ratio < 0.50:
            status = "moderate_shift"
            moderate_shift += 1
        feature_rows.append(
            {
                "feature": col,
                "status": status,
                "train": train_stats,
                "anchor": anchor_stats,
                "mean_shift_in_train_std": float(mean_shift_std),
                "range_overlap_ratio": float(overlap_ratio),
            }
        )

    stable_shared = [
        str(row.get("feature"))
        for row in feature_rows
        if row.get("status") in {"ok", "moderate_shift"}
    ]
    severe_shared = [
        str(row.get("feature"))
        for row in feature_rows
        if row.get("status") == "severe_shift"
    ]

    target_shift = None
    if TARGET_COLUMN in train_df.columns and TARGET_COLUMN in anchor_df.columns:
        target_shift = {
            "target": TARGET_COLUMN,
            "train": stats(train_df[TARGET_COLUMN]),
            "anchor": stats(anchor_df[TARGET_COLUMN]),
        }

    if severe_shift:
        status = "severe_shift"
        reason = f"{severe_shift} shared feature(s) have severe synthetic-anchor distribution shift."
    elif moderate_shift:
        status = "moderate_shift"
        reason = f"{moderate_shift} shared feature(s) have moderate synthetic-anchor distribution shift."
    else:
        status = "ok"
        reason = "Shared train-anchor feature distributions are broadly aligned by simple summary statistics."

    return {
        "status": status,
        "reason": reason,
        "shared_feature_count": len(shared),
        "train_only_features": train_only,
        "anchor_only_features": anchor_only,
        "recommended_anchor_feature_columns": stable_shared,
        "severe_shift_feature_columns": severe_shared,
        "feature_shift": feature_rows,
        "target_shift": target_shift,
        "recommendations": [
            "If all candidate strategies fail anchor validation, generate anchor-compatible candidates using only shared Table 6-supported features.",
            "Prefer recommended_anchor_feature_columns for recovery candidates; severe_shift_feature_columns need ablation, reweighting, or exclusion.",
            "Consider synthetic sample reweighting or regeneration to match anchor phi/sp_percent/w_b/fa_ratio distributions.",
            "Do not treat synthetic OOF as research success when anchor validation is negative.",
        ],
    }


def _anchor_shift_risk_for_features(feature_columns: list[str], shift_audit: dict[str, Any]) -> dict[str, Any]:
    rows = shift_audit.get("feature_shift") if isinstance(shift_audit, dict) else []
    by_feature = {
        str(row.get("feature")): row
        for row in rows or []
        if isinstance(row, dict) and row.get("feature") is not None
    }
    severe = []
    moderate = []
    constant_anchor = []
    for col in feature_columns:
        row = by_feature.get(str(col))
        if not isinstance(row, dict):
            continue
        status = str(row.get("status") or "")
        if status == "severe_shift":
            severe.append(str(col))
        elif status == "moderate_shift":
            moderate.append(str(col))
        anchor_stats = row.get("anchor") if isinstance(row.get("anchor"), dict) else {}
        train_stats = row.get("train") if isinstance(row.get("train"), dict) else {}
        anchor_std = _safe_float(anchor_stats.get("std"), 0.0) or 0.0
        train_std = _safe_float(train_stats.get("std"), 0.0) or 0.0
        if anchor_std <= 1e-12 and train_std > 1e-8:
            constant_anchor.append(str(col))
    if severe:
        status = "severe_shift"
    elif moderate:
        status = "moderate_shift"
    else:
        status = "ok"
    return {
        "status": status,
        "severe_shift_features": severe,
        "moderate_shift_features": moderate,
        "anchor_constant_but_train_variable_features": constant_anchor,
        "risk_rule": (
            "A strategy with severe-shift features is not viable on weak anchor R2; "
            "it needs moderate/strong anchor evidence or explicit ablation/reweighting."
        ),
    }


def _subsample_for_benchmark(df: pd.DataFrame, max_rows: int, random_state: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    if len(df) <= max_rows:
        return df.reset_index(drop=True), {"subsampled": False, "n_rows": int(len(df))}
    sampled = df.sample(n=max_rows, random_state=random_state).reset_index(drop=True)
    return sampled, {
        "subsampled": True,
        "n_rows": int(len(sampled)),
        "n_original_rows": int(len(df)),
        "policy": "deterministic random subsample for lightweight candidate ranking only",
    }


def _score_results(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    evaluated = [row for row in rows if row.get("benchmark_status") == "evaluated"]
    if not evaluated:
        return rows
    best_rmse = min(float(row["oof_metrics"]["rmse"]) for row in evaluated)
    anchor_rmses = [
        _safe_float((row.get("anchor_validation") or {}).get("rmse"))
        for row in evaluated
        if isinstance(row.get("anchor_validation"), dict)
    ]
    best_anchor_rmse = min([x for x in anchor_rmses if x is not None], default=None)
    for row in rows:
        if row.get("benchmark_status") != "evaluated":
            row["benchmark_score_1_to_10"] = 0.0
            row["audited_selection_score_1_to_10"] = 0.0
            continue
        generalization = _anchor_generalization_summary(row)
        row["generalization_gap"] = generalization
        row["anchor_generalization_status"] = generalization["status"]
        if generalization["status"] == "failed":
            row["anchor_generalization_failure_reason"] = generalization["reason"]
        rmse = float(row["oof_metrics"]["rmse"])
        oof_score = 10.0 if rmse <= 0 else max(0.0, min(10.0, 10.0 * best_rmse / rmse))
        anchor_score = oof_score
        anchor = row.get("anchor_validation") if isinstance(row.get("anchor_validation"), dict) else None
        if anchor and best_anchor_rmse is not None:
            anchor_rmse = _safe_float(anchor.get("rmse"))
            if anchor_rmse is not None and anchor_rmse > 0:
                anchor_score = max(0.0, min(10.0, 10.0 * best_anchor_rmse / anchor_rmse))
            anchor_r2 = _safe_float(anchor.get("r2"))
            if anchor_r2 is not None and anchor_r2 < 0:
                anchor_score = min(anchor_score, 2.0)
        data_score = _safe_float(row.get("deterministic_data_score_1_to_10"), 5.0) or 5.0
        penalty = 0.0
        if row.get("data_support") == "unsupported":
            penalty += 4.0
        if row.get("uses_hb_like_text") and not row.get("has_shear_curve", False):
            penalty += 1.0
        if row.get("anchor_generalization_status") == "failed":
            penalty += 1.5
        elif row.get("anchor_generalization_status") == "weak":
            penalty += 0.5
        shift_risk = row.get("anchor_shift_risk") if isinstance(row.get("anchor_shift_risk"), dict) else {}
        severe_shift_features = shift_risk.get("severe_shift_features") or []
        if severe_shift_features:
            penalty += min(2.5, 0.6 * len(severe_shift_features))
            row["anchor_shift_risk_warning"] = (
                "Strategy uses features with severe synthetic-anchor distribution shift; "
                "weak positive anchor R2 is not enough for viable selection."
            )
        audited_score = 0.35 * data_score + 0.45 * oof_score + 0.20 * anchor_score - penalty
        row["benchmark_score_1_to_10"] = round(float(oof_score), 3)
        row["anchor_score_1_to_10"] = round(float(anchor_score), 3)
        row["audited_selection_score_1_to_10"] = round(float(max(0.0, min(10.0, audited_score))), 3)
    return rows


def run_candidate_benchmark(
    data_path: str | Path,
    candidate_report: dict[str, Any],
    *,
    anchor_path: str | Path | None = None,
    random_state: int = 42,
    max_rows: int = 600,
    n_splits: int = 5,
) -> dict[str, Any]:
    """Run cheap proxy benchmarks for CandidateAgent hybrid strategies."""
    df_full, meta = load_yield_dataframe(data_path)
    df, sample_info = _subsample_for_benchmark(df_full, max_rows=max_rows, random_state=random_state)
    feature_columns = _feature_columns(meta, df)
    n_splits = min(max(2, int(n_splits)), max(2, len(df)))
    anchor_df = None
    if anchor_path and Path(anchor_path).exists():
        try:
            anchor_df, _anchor_meta = load_yield_dataframe(anchor_path)
        except Exception:
            anchor_df = None

    mechanisms = {
        str(item.get("id")): item
        for item in candidate_report.get("candidate_mechanisms", []) or []
        if isinstance(item, dict) and item.get("id") is not None
    }
    models = {
        str(item.get("id")): item
        for item in candidate_report.get("candidate_models", []) or []
        if isinstance(item, dict) and item.get("id") is not None
    }
    strategies = [
        item
        for item in candidate_report.get("candidate_hybrid_strategies", []) or []
        if isinstance(item, dict)
    ][:6]

    rows: list[dict[str, Any]] = []
    shift_audit = _feature_shift_summary(df, anchor_df, feature_columns)
    has_shear_curve = bool(
        ((candidate_report.get("data_capability_audit") or {}).get("flags") or {}).get("has_shear_rate_or_flow_curve")
    )
    for index, strategy in enumerate(strategies):
        strategy_id = str(strategy.get("id") or f"strategy_{index + 1}")
        data_support = str(strategy.get("data_support") or "partial").lower()
        classification = _classify_strategy(strategy, mechanisms, models)
        model = models.get(str(strategy.get("model_id") or ""))
        strategy_feature_columns, feature_resolution = _strategy_feature_columns(
            strategy,
            model if isinstance(model, dict) else None,
            feature_columns,
            anchor_df,
        )
        anchor_shift_risk = _anchor_shift_risk_for_features(strategy_feature_columns, shift_audit)
        row_prefix = {
            "strategy_id": strategy_id,
            "strategy_name": strategy.get("name"),
            "model_id": strategy.get("model_id"),
            "mechanism_ids": strategy.get("mechanism_ids", []),
            "data_support": data_support,
            "deterministic_data_score_1_to_10": _safe_float(
                strategy.get("deterministic_data_score_1_to_10", strategy.get("total_score_1_to_10")),
                5.0,
            ),
            "uses_packing_proxy": bool(classification["uses_packing_proxy"]),
            "uses_hb_like_text": bool(classification["uses_hb_like_text"]),
            "mechanism_kinds": classification.get("mechanism_kinds", []),
            "has_shear_curve": has_shear_curve,
            "proxy_model": classification["proxy_model"],
            "feature_mode": classification["feature_mode"],
            "evaluated_base_feature_columns": strategy_feature_columns,
            "feature_resolution": feature_resolution,
            "anchor_shift_risk": anchor_shift_risk,
            "benchmark_proxy_for_strategy": True,
            "warnings": classification["warnings"],
        }
        if not strategy_feature_columns:
            rows.append(
                {
                    **row_prefix,
                    "benchmark_status": "skipped_no_valid_features",
                    "skip_reason": "Strategy did not resolve to any non-leakage feature columns present in the training data.",
                }
            )
            continue
        if data_support == "unsupported":
            rows.append(
                {
                    **row_prefix,
                    "benchmark_status": "skipped_unsupported_data",
                    "skip_reason": strategy.get("rejection_reason") or strategy.get("data_support_reason") or "Strategy data_support is unsupported.",
                }
            )
            continue
        try:
            evaluated = _evaluate_proxy(
                df,
                strategy_feature_columns,
                strategy_id,
                classification["feature_mode"],
                classification["proxy_model"],
                random_state=random_state + index * 17,
                n_splits=n_splits,
                anchor_df=anchor_df,
            )
            rows.append({**row_prefix, **evaluated})
        except Exception as exc:
            rows.append(
                {
                    **row_prefix,
                    "benchmark_status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    rows = _score_results(rows)
    evaluated_rows = [row for row in rows if row.get("benchmark_status") == "evaluated"]
    has_anchor_validation = anchor_df is not None and len(anchor_df) > 0
    for row in evaluated_rows:
        if not has_anchor_validation:
            row["anchor_viability_status"] = "viable_without_anchor"
            continue
        status = row.get("anchor_generalization_status")
        severe_features = (
            (row.get("anchor_shift_risk") or {}).get("severe_shift_features")
            if isinstance(row.get("anchor_shift_risk"), dict)
            else []
        )
        if status == "failed":
            row["anchor_viability_status"] = "failed_negative_anchor_r2"
        elif status == "weak" and severe_features:
            row["anchor_viability_status"] = "blocked_shift_risky_weak_anchor"
            row["anchor_generalization_failure_reason"] = (
                "Anchor R2 is only weak while the strategy uses severe synthetic-anchor shift features: "
                + ", ".join(str(x) for x in severe_features)
            )
        elif status in {"weak", "moderate", "strong"}:
            row["anchor_viability_status"] = f"viable_{status}_anchor"
        else:
            row["anchor_viability_status"] = "not_viable_unknown_anchor_status"
    viable_rows = [
        row
        for row in evaluated_rows
        if not has_anchor_validation
        or str(row.get("anchor_viability_status", "")).startswith("viable_")
    ]
    failed_anchor_rows = [
        row
        for row in evaluated_rows
        if has_anchor_validation and row.get("anchor_generalization_status") == "failed"
    ]
    shift_risky_weak_rows = [
        row
        for row in evaluated_rows
        if has_anchor_validation and row.get("anchor_viability_status") == "blocked_shift_risky_weak_anchor"
    ]
    no_viable_anchor_candidate = bool(has_anchor_validation and evaluated_rows and not viable_rows)
    selected_pool = viable_rows if viable_rows else ([] if no_viable_anchor_candidate else evaluated_rows)
    selected = max(
        selected_pool,
        key=lambda row: float(row.get("audited_selection_score_1_to_10", 0.0)),
        default=None,
    )
    baseline_summary = _evaluate_baseline_summary(df, feature_columns, n_splits=n_splits, random_state=random_state)
    best_baseline = baseline_summary.get("best_baseline") if isinstance(baseline_summary, dict) else None
    best_baseline_rmse = None
    if isinstance(best_baseline, dict):
        best_baseline_rmse = _safe_float(best_baseline.get("oof_rmse", best_baseline.get("mean_rmse")))

    selected_oof_rmse = None
    if isinstance(selected, dict):
        selected_oof_rmse = _safe_float((selected.get("oof_metrics") or {}).get("rmse"))
    viability_summary = {
        "has_anchor_validation": has_anchor_validation,
        "evaluated_strategy_count": len(evaluated_rows),
        "viable_strategy_count": len(viable_rows),
        "failed_anchor_strategy_count": len(failed_anchor_rows),
        "shift_risky_weak_strategy_count": len(shift_risky_weak_rows),
        "no_viable_anchor_candidate": no_viable_anchor_candidate,
        "viability_rule": (
            "When real anchor validation exists, anchor R2 < 0 disqualifies a strategy from benchmark selection. "
            "0 <= anchor R2 < 0.3 remains weak but viable only for strategies that do not rely on severe synthetic-anchor shift features."
        ),
        "failure_reason": (
            "all_candidates_failed_anchor_validation" if no_viable_anchor_candidate else None
        ),
        "failed_anchor_strategy_ids": [row.get("strategy_id") for row in failed_anchor_rows],
        "shift_risky_weak_strategy_ids": [row.get("strategy_id") for row in shift_risky_weak_rows],
        "nonviable_strategy_ids": [
            row.get("strategy_id")
            for row in evaluated_rows
            if row not in viable_rows
        ],
        "viable_strategy_ids": [row.get("strategy_id") for row in viable_rows],
    }

    return {
        "stage": "candidate_benchmark",
        "status": "no_viable_anchor_candidate" if no_viable_anchor_candidate else "completed",
        "purpose": "lightweight proxy ranking for CandidateAgent strategies; not the final generated training model",
        "data_path": str(data_path),
        "anchor_path": str(anchor_path) if anchor_path else "",
        "schema": meta,
        "sample_info": sample_info,
        "synthetic_anchor_shift_audit": shift_audit,
        "anchor_viability_summary": viability_summary,
        "feature_policy": {
            "feature_columns": feature_columns,
            "leakage_columns_excluded": sorted(LEAKAGE_COLUMNS),
            "packing_features_are_fold_local": True,
        },
        "benchmark_protocol": {
            "primary_evaluation": f"{n_splits}-fold OOF",
            "secondary_evaluation": "anchor_validation" if anchor_df is not None else None,
            "proxy_models": ["ridge", "hgb", "rf", "kernel", "mlp"],
            "selection_score": "0.35*data_capability + 0.45*OOF_rank_score + 0.20*anchor_rank_score - unsupported/HB/generalization penalties",
            "random_state": int(random_state),
        },
        "strategy_benchmarks": rows,
        "fixed_baseline_summary": baseline_summary,
        "selected_strategy_id": selected.get("strategy_id") if isinstance(selected, dict) else None,
        "selected_strategy_status": (
            "viable_anchor_candidate"
            if isinstance(selected, dict) and has_anchor_validation
            else "selected_without_anchor"
            if isinstance(selected, dict)
            else "no_viable_anchor_candidate"
            if no_viable_anchor_candidate
            else "no_evaluated_strategy"
        ),
        "selected_strategy_score_1_to_10": selected.get("audited_selection_score_1_to_10") if isinstance(selected, dict) else None,
        "selected_strategy_oof_rmse": selected_oof_rmse,
        "selected_strategy_anchor_validation": selected.get("anchor_validation") if isinstance(selected, dict) else None,
        "selected_strategy_generalization_gap": selected.get("generalization_gap") if isinstance(selected, dict) else None,
        "selected_strategy_anchor_generalization_status": selected.get("anchor_generalization_status") if isinstance(selected, dict) else None,
        "best_fixed_baseline_oof_rmse": best_baseline_rmse,
        "selected_beats_fixed_baseline_proxy": (
            bool(selected_oof_rmse is not None and best_baseline_rmse is not None and selected_oof_rmse <= best_baseline_rmse)
            if selected_oof_rmse is not None and best_baseline_rmse is not None
            else None
        ),
    }


def apply_benchmark_selection(
    candidate_report: dict[str, Any],
    benchmark_report: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return an updated candidate report plus a selection audit payload."""
    updated = deepcopy(candidate_report)
    rows = [
        row
        for row in benchmark_report.get("strategy_benchmarks", []) or []
        if isinstance(row, dict)
    ]
    row_by_id = {str(row.get("strategy_id")): row for row in rows if row.get("strategy_id") is not None}
    mechanisms = {
        str(item.get("id")): item
        for item in updated.get("candidate_mechanisms", []) or []
        if isinstance(item, dict) and item.get("id") is not None
    }
    selected_before = None
    if isinstance(updated.get("selected_combination"), dict):
        selected_before = updated["selected_combination"].get("strategy_id")
    benchmark_selected = benchmark_report.get("selected_strategy_id")
    no_viable_anchor_candidate = bool(
        (benchmark_report.get("anchor_viability_summary") or {}).get("no_viable_anchor_candidate")
        or benchmark_report.get("status") == "no_viable_anchor_candidate"
    )
    selected_after = None if no_viable_anchor_candidate else (benchmark_selected or selected_before)

    def row_summary(row: dict[str, Any]) -> dict[str, Any]:
        oof = row.get("oof_metrics") if isinstance(row.get("oof_metrics"), dict) else {}
        anchor = row.get("anchor_validation") if isinstance(row.get("anchor_validation"), dict) else {}
        return {
            "strategy_id": row.get("strategy_id"),
            "strategy_name": row.get("strategy_name"),
            "model_id": row.get("model_id"),
            "mechanism_ids": row.get("mechanism_ids", []),
            "benchmark_status": row.get("benchmark_status"),
            "proxy_model": row.get("proxy_model"),
            "feature_mode": row.get("feature_mode"),
            "evaluated_base_feature_columns": row.get("evaluated_base_feature_columns"),
            "feature_columns_used": row.get("feature_columns_used"),
            "anchor_shift_risk": row.get("anchor_shift_risk"),
            "anchor_viability_status": row.get("anchor_viability_status"),
            "oof_rmse": oof.get("rmse"),
            "oof_r2": oof.get("r2"),
            "anchor_rmse": anchor.get("rmse"),
            "anchor_r2": anchor.get("r2"),
            "anchor_mape": anchor.get("mape"),
            "anchor_generalization_status": row.get("anchor_generalization_status"),
            "generalization_gap": row.get("generalization_gap"),
            "audited_selection_score_1_to_10": row.get("audited_selection_score_1_to_10"),
            "skip_reason": row.get("skip_reason"),
            "error": row.get("error"),
        }

    for strategy in updated.get("candidate_hybrid_strategies", []) or []:
        if not isinstance(strategy, dict):
            continue
        sid = str(strategy.get("id") or "")
        bench = row_by_id.get(sid)
        if bench:
            strategy["benchmark_audit"] = {
                "benchmark_status": bench.get("benchmark_status"),
                "proxy_model": bench.get("proxy_model"),
                "feature_mode": bench.get("feature_mode"),
                "evaluated_base_feature_columns": bench.get("evaluated_base_feature_columns"),
                "feature_resolution": bench.get("feature_resolution"),
                "anchor_shift_risk": bench.get("anchor_shift_risk"),
                "anchor_viability_status": bench.get("anchor_viability_status"),
                "feature_columns_used": bench.get("feature_columns_used"),
                "anchor_feature_columns_used": bench.get("anchor_feature_columns_used"),
                "oof_metrics": bench.get("oof_metrics"),
                "anchor_validation": bench.get("anchor_validation"),
                "generalization_gap": bench.get("generalization_gap"),
                "anchor_generalization_status": bench.get("anchor_generalization_status"),
                "anchor_generalization_failure_reason": bench.get("anchor_generalization_failure_reason"),
                "benchmark_score_1_to_10": bench.get("benchmark_score_1_to_10"),
                "audited_selection_score_1_to_10": bench.get("audited_selection_score_1_to_10"),
                "warnings": bench.get("warnings", []),
                "skip_reason": bench.get("skip_reason"),
                "error": bench.get("error"),
            }
        strategy["selected"] = bool(selected_after and sid and sid == str(selected_after))

    selected_strategy = next(
        (
            item
            for item in updated.get("candidate_hybrid_strategies", []) or []
            if isinstance(item, dict) and str(item.get("id")) == str(selected_after)
        ),
        None,
    )
    selected_row = row_by_id.get(str(selected_after))
    if isinstance(selected_strategy, dict):
        updated["selected_combination"] = {
            "strategy_id": selected_strategy.get("id"),
            "model_id": selected_strategy.get("model_id"),
            "mechanism_ids": selected_strategy.get("mechanism_ids", []),
            "uses_mechanism": bool(selected_strategy.get("mechanism_ids")),
            "selection_rationale": (
                "Selected after data-capability audit plus lightweight benchmark proxy. "
                "The proxy benchmark does not fix final code; OperationAgent still implements a full runnable model."
            ),
            "fallback_if_mechanism_not_applicable": selected_strategy.get(
                "fallback_if_mechanism_not_applicable",
                "If the mechanism cannot be implemented leakage-safely, report it rejected and use the closest data-driven candidate.",
            ),
            "evaluation_protocol": "Primary 5-fold OOF plus anchor_validation when available.",
            "benchmark_proxy": {
                "proxy_model": selected_row.get("proxy_model") if isinstance(selected_row, dict) else None,
                "feature_mode": selected_row.get("feature_mode") if isinstance(selected_row, dict) else None,
                "evaluated_base_feature_columns": selected_row.get("evaluated_base_feature_columns") if isinstance(selected_row, dict) else None,
                "feature_columns_used": selected_row.get("feature_columns_used") if isinstance(selected_row, dict) else None,
                "mechanism_kinds": selected_row.get("mechanism_kinds") if isinstance(selected_row, dict) else None,
                "audited_selection_score_1_to_10": selected_row.get("audited_selection_score_1_to_10") if isinstance(selected_row, dict) else None,
                "oof_metrics": selected_row.get("oof_metrics") if isinstance(selected_row, dict) else None,
                "anchor_validation": selected_row.get("anchor_validation") if isinstance(selected_row, dict) else None,
                "generalization_gap": selected_row.get("generalization_gap") if isinstance(selected_row, dict) else None,
                "anchor_generalization_status": selected_row.get("anchor_generalization_status") if isinstance(selected_row, dict) else None,
            },
        }
        updated["selection_audit_note"] = (
            "Selected combination was audited with lightweight candidate benchmarks. "
            f"before={selected_before}, after={selected_after}."
        )
    elif no_viable_anchor_candidate:
        updated["selected_combination"] = {
            "strategy_id": None,
            "selection_status": "no_viable_anchor_candidate",
            "selection_rationale": (
                "No candidate strategy passed the real-anchor viability gate. "
                "OperationAgent should not implement a failed-anchor strategy; CandidateAgent must revise candidates."
            ),
            "fallback_if_mechanism_not_applicable": (
                "Generate anchor-compatible candidates using shared Table 6-supported features or revise synthetic data weighting."
            ),
            "benchmark_proxy": None,
        }
        updated["selection_audit_note"] = (
            "Benchmark did not select a strategy because all evaluated candidates failed anchor validation."
        )

    yodel_audit = updated.get("yodel_packing_audit")
    if isinstance(yodel_audit, dict):
        selected_is_yodel = bool(
            isinstance(selected_strategy, dict)
            and _strategy_includes_yodel(selected_strategy, mechanisms)
        )
        yodel_rows = [
            row
            for row in rows
            if row.get("uses_packing_proxy") is True
            or "yodel_packing_fmax" in (row.get("mechanism_kinds") or [])
        ]
        best_yodel = max(
            yodel_rows,
            key=lambda row: float(row.get("audited_selection_score_1_to_10") or 0.0),
            default=None,
        )
        yodel_audit.update(
            {
                "selected": selected_is_yodel,
                "selected_strategy_id": selected_after if selected_is_yodel else None,
                "selected_strategy_id_after_benchmark": selected_after,
                "selected_strategy_is_yodel_after_benchmark": selected_is_yodel,
                "rejected_by_benchmark": bool(yodel_rows and (not selected_is_yodel or no_viable_anchor_candidate)),
                "benchmark_rejection_evidence": {
                    "best_yodel_strategy_id": best_yodel.get("strategy_id") if isinstance(best_yodel, dict) else None,
                    "best_yodel_audited_score_1_to_10": best_yodel.get("audited_selection_score_1_to_10") if isinstance(best_yodel, dict) else None,
                    "best_yodel_oof_metrics": best_yodel.get("oof_metrics") if isinstance(best_yodel, dict) else None,
                    "best_yodel_anchor_validation": best_yodel.get("anchor_validation") if isinstance(best_yodel, dict) else None,
                    "best_yodel_generalization_gap": best_yodel.get("generalization_gap") if isinstance(best_yodel, dict) else None,
                    "selected_strategy_id": selected_after,
                    "selection_status": "no_viable_anchor_candidate" if no_viable_anchor_candidate else "selected",
                    "selected_strategy_score_1_to_10": selected_row.get("audited_selection_score_1_to_10") if isinstance(selected_row, dict) else None,
                    "selected_strategy_oof_metrics": selected_row.get("oof_metrics") if isinstance(selected_row, dict) else None,
                    "selected_strategy_anchor_validation": selected_row.get("anchor_validation") if isinstance(selected_row, dict) else None,
                    "selected_strategy_generalization_gap": selected_row.get("generalization_gap") if isinstance(selected_row, dict) else None,
                },
                "not_selected_reporting_requirement": (
                    "If selected=false while support_status is supported/partial, mechanism_report.json must include "
                    "rejected_yodel_packing_candidate as a structured object with candidate_id, reason, and benchmark_evidence."
                ),
            }
        )
        updated["yodel_packing_audit"] = yodel_audit

    rejected = []
    for row in rows:
        sid = row.get("strategy_id")
        if sid == selected_after:
            continue
        reason = row.get("skip_reason") or row.get("error")
        if not reason:
            reason = (
                f"audited_score={row.get('audited_selection_score_1_to_10')}, "
                f"selected_score={benchmark_report.get('selected_strategy_score_1_to_10')}"
            )
        rejected.append(
            {
                "strategy_id": sid,
                "strategy_name": row.get("strategy_name"),
                "reason": reason,
                "benchmark_status": row.get("benchmark_status"),
                "proxy_model": row.get("proxy_model"),
                "feature_mode": row.get("feature_mode"),
                "evaluated_base_feature_columns": row.get("evaluated_base_feature_columns"),
                "anchor_shift_risk": row.get("anchor_shift_risk"),
                "anchor_viability_status": row.get("anchor_viability_status"),
                "oof_metrics": row.get("oof_metrics"),
                "anchor_validation": row.get("anchor_validation"),
                "anchor_generalization_status": row.get("anchor_generalization_status"),
                "generalization_gap": row.get("generalization_gap"),
                "audited_selection_score_1_to_10": row.get("audited_selection_score_1_to_10"),
            }
        )

    strategy_summaries = [row_summary(row) for row in rows]

    audit = {
        "stage": "candidate_selection_audit",
        "status": "no_viable_anchor_candidate" if no_viable_anchor_candidate else "completed" if benchmark_selected else "no_evaluated_strategy",
        "benchmark_validated": bool(benchmark_selected and not no_viable_anchor_candidate),
        "anchor_viability_summary": benchmark_report.get("anchor_viability_summary"),
        "synthetic_anchor_shift_audit": benchmark_report.get("synthetic_anchor_shift_audit"),
        "selected_before_benchmark": selected_before,
        "selected_after_benchmark": selected_after,
        "selection_changed_by_benchmark": str(selected_before) != str(selected_after),
        "selection_blocked_by_anchor_viability": no_viable_anchor_candidate,
        "failure_reason": "all_candidates_failed_anchor_validation" if no_viable_anchor_candidate else None,
        "decision_basis": [
            "data capability audit",
            "same-protocol lightweight OOF proxy benchmark",
            "real-anchor viability gate when anchor data exists",
            "fixed baseline comparison as a reference only",
        ],
        "strategy_benchmark_summaries": strategy_summaries,
        "rejected_strategies": rejected,
        "selected_strategy_benchmark": selected_row,
        "benchmark_alignment_requirement": {
            "selected_strategy_id": selected_after,
            "selected_proxy_model": selected_row.get("proxy_model") if isinstance(selected_row, dict) else None,
            "selected_feature_mode": selected_row.get("feature_mode") if isinstance(selected_row, dict) else None,
            "selected_evaluated_base_feature_columns": selected_row.get("evaluated_base_feature_columns") if isinstance(selected_row, dict) else None,
            "selected_feature_columns_used": selected_row.get("feature_columns_used") if isinstance(selected_row, dict) else None,
            "selected_mechanism_kinds": selected_row.get("mechanism_kinds") if isinstance(selected_row, dict) else None,
            "selected_generalization_gap": selected_row.get("generalization_gap") if isinstance(selected_row, dict) else None,
            "selected_anchor_generalization_status": selected_row.get("anchor_generalization_status") if isinstance(selected_row, dict) else None,
            "operation_report_required_fields": [
                "selected_hybrid_strategy",
                "planned_model",
                "actual_model",
                "fallback_reason when planned_model differs from actual_model",
                "benchmark_alignment",
                "implementation_validation",
            ],
        },
        "benchmark_report_summary": {
            "benchmark_status": benchmark_report.get("status"),
            "selected_strategy_id": benchmark_report.get("selected_strategy_id"),
            "selected_strategy_status": benchmark_report.get("selected_strategy_status"),
            "selected_strategy_score_1_to_10": benchmark_report.get("selected_strategy_score_1_to_10"),
            "selected_strategy_oof_rmse": benchmark_report.get("selected_strategy_oof_rmse"),
            "selected_strategy_anchor_validation": benchmark_report.get("selected_strategy_anchor_validation"),
            "selected_strategy_generalization_gap": benchmark_report.get("selected_strategy_generalization_gap"),
            "selected_strategy_anchor_generalization_status": benchmark_report.get("selected_strategy_anchor_generalization_status"),
            "best_fixed_baseline_oof_rmse": benchmark_report.get("best_fixed_baseline_oof_rmse"),
            "selected_beats_fixed_baseline_proxy": benchmark_report.get("selected_beats_fixed_baseline_proxy"),
            "sample_info": benchmark_report.get("sample_info"),
        },
        "note": (
            "This is not a mechanism evidence-card cache and not a fixed model registry. "
            "It is a run-local audit of the current CandidateAgent strategies."
        ),
    }
    updated["candidate_selection_audit"] = audit
    return updated, audit


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        out = float(value)
        return out if math.isfinite(out) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "item"):
        try:
            return _jsonable(value.item())
        except Exception:
            return str(value)
    return value


def _selected_row_from_benchmark(benchmark_report: dict[str, Any]) -> dict[str, Any] | None:
    selected_id = benchmark_report.get("selected_strategy_id")
    for row in benchmark_report.get("strategy_benchmarks", []) or []:
        if isinstance(row, dict) and str(row.get("strategy_id")) == str(selected_id):
            return row
    row = benchmark_report.get("selected_strategy_benchmark")
    return row if isinstance(row, dict) else None


def _fit_final_proxy_model(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    feature_mode: str,
    proxy_model: str,
    random_state: int,
) -> dict[str, Any]:
    raw_features = _make_raw_features(train_df, feature_columns, feature_mode, train_df)
    encoded = pd.get_dummies(raw_features, dummy_na=False).apply(pd.to_numeric, errors="coerce")
    imputer = SimpleImputer(strategy="median")
    x_train = imputer.fit_transform(encoded)
    y = pd.to_numeric(train_df[TARGET_COLUMN], errors="coerce").to_numpy(dtype=float)
    model = _make_model(proxy_model, random_state, len(train_df))
    model.fit(x_train.astype(float), y)
    return {
        "model": model,
        "imputer": imputer,
        "encoded_columns": list(encoded.columns),
        "raw_feature_columns": feature_columns,
        "feature_mode": feature_mode,
        "proxy_model": proxy_model,
        "fit_scope": "full_training_data_only",
    }


def _transform_with_final_bundle(df: pd.DataFrame, train_df_for_params: pd.DataFrame, bundle: dict[str, Any]) -> np.ndarray:
    raw = _make_raw_features(
        df,
        list(bundle.get("raw_feature_columns") or []),
        str(bundle.get("feature_mode") or "base"),
        train_df_for_params,
    )
    encoded = pd.get_dummies(raw, dummy_na=False).apply(pd.to_numeric, errors="coerce")
    encoded = encoded.reindex(columns=list(bundle.get("encoded_columns") or []), fill_value=0.0)
    return bundle["imputer"].transform(encoded).astype(float)


def _build_yield_metrics_payload(
    *,
    feature_columns: list[str],
    result: dict[str, Any],
    base_result: dict[str, Any],
    baseline_summary: dict[str, Any],
    final_encoded_columns: list[str],
    model_label: str,
    feature_mode_label: str,
    candidate_source: str,
    uses_mechanism: bool,
    fallback_reason: str | None,
    anchor_present: bool,
    n_splits: int,
    benchmark_oof_rmse: Any = None,
    benchmark_anchor_r2: Any = None,
    benchmark_generalization_gap: Any = None,
    benchmark_proxy_model: Any = None,
    benchmark_feature_mode: Any = None,
    best_alt_anchor_r2: Any = None,
    engineered_prefixes: tuple[str, ...] = ("bench_",),
    mechanism_descriptor: dict[str, Any] | None = None,
    candidate_extra: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Assemble metrics / mechanism_report / preprocessing_audit payloads.

    Shared by both the deterministic proxy fallback and the LLM-plugin harness so
    the artifact field contract (consumed by verify_yield_run) lives in ONE place.
    """
    baseline_rows = list(baseline_summary.get("baseline_results") or [])
    best_baseline = baseline_summary.get("best_baseline") if isinstance(baseline_summary, dict) else None
    best_baseline_name = str((best_baseline or {}).get("name") or "")
    best_baseline_rmse = _safe_float((best_baseline or {}).get("oof_rmse", (best_baseline or {}).get("mean_rmse")))
    oof = result.get("oof_metrics") or {}
    anchor = result.get("anchor_validation") or {}
    base_oof = base_result.get("oof_metrics") or {}
    selected_rmse = _safe_float(oof.get("rmse"))
    beats = bool(best_baseline_rmse is not None and selected_rmse is not None and selected_rmse <= best_baseline_rmse)
    delta_rmse = None
    relative = None
    if _safe_float(base_oof.get("rmse")) is not None and selected_rmse is not None:
        base_rmse = float(base_oof.get("rmse"))
        delta_rmse = base_rmse - float(selected_rmse)
        relative = delta_rmse / base_rmse if abs(base_rmse) > 1e-12 else None
    mechanism_improves = bool(delta_rmse is not None and delta_rmse > 0)
    anchor_r2 = _safe_float(anchor.get("r2"))

    # Confidence gate (tightened): `high` demands the candidate not only beats the
    # fixed baseline but also (a) is not meaningfully worse on the real anchor than
    # the best alternative arm (reference/benchmark), (b) is not worse than the
    # benchmark-selected proxy on OOF, and (c) actually gains from its mechanism
    # features. Otherwise a synthetic-OOF win with weak anchor is only medium/low.
    ANCHOR_TOL = 0.03
    RMSE_TOL = 0.005
    best_alt = _safe_float(best_alt_anchor_r2)
    bench_rmse = _safe_float(benchmark_oof_rmse)
    anchor_not_worse_than_alt = best_alt is None or (anchor_r2 is not None and anchor_r2 >= best_alt - ANCHOR_TOL)
    not_worse_than_benchmark = bench_rmse is None or (selected_rmse is not None and selected_rmse <= bench_rmse + RMSE_TOL)
    strong_anchor = anchor_r2 is not None and anchor_r2 >= 0.6
    if not beats:
        confidence = "low"
        status = "underperforms_baseline"
    elif strong_anchor and anchor_not_worse_than_alt and not_worse_than_benchmark and mechanism_improves:
        confidence = "high"
        status = "beats_baseline"
    elif strong_anchor:
        confidence = "medium"
        status = "beats_baseline"
    elif anchor_r2 is not None and anchor_r2 >= 0.3:
        confidence = "medium"
        status = "competitive_with_baseline"
    else:
        confidence = "low"
        status = "completed_with_warning"
    confidence_gate = {
        "beats_best_baseline": beats,
        "anchor_r2": anchor_r2,
        "best_alternative_anchor_r2": best_alt,
        "anchor_not_worse_than_alternatives": bool(anchor_not_worse_than_alt),
        "benchmark_proxy_oof_rmse": bench_rmse,
        "not_worse_than_benchmark_proxy": bool(not_worse_than_benchmark),
        "mechanism_improves_metric": mechanism_improves,
        "anchor_tolerance": ANCHOR_TOL,
        "rmse_tolerance": RMSE_TOL,
    }

    proxy_model = result.get("proxy_model")
    feature_mode = result.get("feature_mode")
    strategy_id = result.get("strategy_id")
    is_fallback = fallback_reason is not None

    candidate_row = {
        "name": model_label,
        "strategy_id": strategy_id,
        "fixed_baseline": False,
        "benchmark_selected_proxy": candidate_source == "benchmark_proxy",
        "candidate_source": candidate_source,
        "proxy_model": proxy_model,
        "feature_mode": feature_mode,
        "oof_rmse": oof.get("rmse"),
        "oof_mae": oof.get("mae"),
        "oof_r2": oof.get("r2"),
        "oof_mape": oof.get("mape"),
        "anchor_validation": anchor,
        "uses_mechanism_constraint": bool(uses_mechanism),
        "operation_fallback": is_fallback,
    }
    implementation_validation = {
        "benchmark_selected_strategy": strategy_id,
        "benchmark_oof_rmse": benchmark_oof_rmse,
        "actual_oof_rmse": oof.get("rmse"),
        "benchmark_anchor_r2": benchmark_anchor_r2,
        "actual_anchor_r2": anchor.get("r2"),
        "generalization_gap": result.get("generalization_gap") or benchmark_generalization_gap,
        # aligned_success only when the tightened confidence gate did not drop to
        # low — a synthetic-OOF win with a weak/worse anchor is NOT "aligned".
        "validation_status": (
            "aligned_success"
            if anchor_r2 is not None and anchor_r2 >= 0 and confidence != "low"
            else "completed_with_warning"
        ),
        "confidence_gate": confidence_gate,
        "operation_fallback_reason": fallback_reason,
        "candidate_source": candidate_source,
    }
    candidate_result_block = {
        "strategy_id": strategy_id,
        "candidate_source": candidate_source,
        "model_label": model_label,
        "proxy_model": proxy_model,
        "feature_mode": feature_mode,
        "evaluated_base_feature_columns": feature_columns,
        "oof_metrics": oof,
        "anchor_validation": anchor,
    }
    mechanism_ablation = {
        "without_mechanism": base_oof,
        "with_mechanism": oof,
        "without_mechanism_model": base_result.get("proxy_model"),
        "with_mechanism_model": proxy_model,
        "same_model_protocol": str(base_result.get("proxy_model")) == str(proxy_model),
        "protocol_note": (
            "without_mechanism uses the SAME model family as with_mechanism; the "
            "delta reflects mechanism features only, not a model-family difference."
        ),
        "delta_rmse": delta_rmse,
        "delta_r2": (
            (_safe_float(oof.get("r2")) or 0.0) - (_safe_float(base_oof.get("r2")) or 0.0)
            if _safe_float(oof.get("r2")) is not None and _safe_float(base_oof.get("r2")) is not None
            else None
        ),
        "mechanism_improves_metric": mechanism_improves,
    }
    effect_size = {
        "relative_rmse_improvement": relative,
        "relative_rmse_improvement_percent": None if relative is None else relative * 100.0,
        "claim_strength": (
            "moderate"
            if relative is not None and relative >= 0.05
            else "marginal"
            if relative is not None and relative > 0
            else "no_gain"
        ),
    }
    bm_proxy_model = benchmark_proxy_model if benchmark_proxy_model is not None else proxy_model
    bm_feature_mode = benchmark_feature_mode if benchmark_feature_mode is not None else feature_mode
    benchmark_alignment = {
        "selected_strategy_id": strategy_id,
        "benchmark_proxy_model": bm_proxy_model,
        "benchmark_feature_mode": bm_feature_mode,
        "implemented_model": proxy_model,
        "implemented_feature_mode": feature_mode,
        "implemented_matches_benchmark": bool(
            str(bm_proxy_model) == str(proxy_model) and str(bm_feature_mode) == str(feature_mode)
        ),
        "alignment_status": "exact_proxy_fallback" if candidate_source == "benchmark_proxy" else "plugin_execution",
        "deviation_reason": fallback_reason,
    }
    selected_hybrid_strategy = {
        "strategy_id": strategy_id,
        "proxy_model": proxy_model,
        "feature_mode": feature_mode,
        "candidate_benchmark_selected": True,
        "candidate_source": candidate_source,
    }
    metrics = {
        "selected_model": model_label,
        "planned_model": model_label,
        "actual_model": model_label,
        "candidate_source": candidate_source,
        "fallback_reason": fallback_reason,
        "primary_evaluation": f"{n_splits}-fold OOF",
        "secondary_evaluation": "anchor_validation" if anchor_present else None,
        "evaluation_protocol": "5-fold OOF plus real anchor validation when anchor data exists.",
        "oof_metrics": oof,
        "anchor_validation": anchor,
        "baseline_results": baseline_rows,
        "candidate_models": [candidate_row],
        "model_results": [candidate_row],
        "selection_metric": "oof_rmse",
        "selection_metric_name": "oof_rmse",
        "selection_metric_direction": "lower_is_better",
        "best_baseline": best_baseline_name,
        "best_baseline_metric": best_baseline_rmse,
        "best_research_model": model_label,
        "best_research_metric": selected_rmse,
        "beats_best_baseline": beats,
        "research_model_status": status,
        "research_confidence": confidence,
        "selected_hybrid_strategy": selected_hybrid_strategy,
        "benchmark_alignment": benchmark_alignment,
        "implementation_validation": implementation_validation,
        "benchmark_selected_proxy_result": candidate_result_block,
        "mechanism_ablation": mechanism_ablation,
        "mechanism_effect_size": effect_size,
        "operation_fallback": is_fallback,
    }
    if candidate_extra:
        metrics.update(candidate_extra)

    engineered = [col for col in final_encoded_columns if any(str(col).startswith(p) for p in engineered_prefixes)]
    preprocessing_audit = {
        "raw_feature_columns": feature_columns,
        "final_feature_columns": list(final_encoded_columns),
        "engineered_feature_columns": engineered,
        "train_feature_count": len(final_encoded_columns),
        "anchor_feature_count": len(final_encoded_columns) if anchor_present else None,
        "anchor_aligned_to_train_schema": True if anchor_present else None,
        "fold_local_preprocessing": True,
        "final_model_fit_scope": "fit on full training data only; anchor used only for validation transform/prediction",
        "leakage_columns_excluded": sorted(LEAKAGE_COLUMNS),
        "candidate_source": candidate_source,
    }
    mech = mechanism_descriptor or {
        "name": "benchmark proxy packing/SP feature mode",
        "formula_or_relationship": "bench_packing_index and bench_sp_decay are generated fold-locally from allowed features by CandidateBenchmark.",
        "feature_mode": feature_mode,
        "columns": feature_columns,
        "source_ids": ["candidate_selection_audit"],
    }
    mechanism_report = {
        "mechanism_used": bool(uses_mechanism),
        "mechanism_attempted": bool(uses_mechanism),
        "candidate_source": candidate_source,
        "mechanisms": [mech],
        "planned_model": model_label,
        "actual_model": model_label,
        "fallback_reason": fallback_reason,
        "selected_hybrid_strategy": selected_hybrid_strategy,
        "benchmark_alignment": benchmark_alignment,
        "implementation_validation": implementation_validation,
        "benchmark_selected_proxy_result": candidate_result_block,
        "mechanism_ablation": mechanism_ablation,
        "mechanism_effect_size": effect_size,
        "operation_fallback": is_fallback,
    }
    if candidate_extra:
        for key in ("plugin_candidate_result", "reference_arms", "plugin_error", "benchmark_selected_proxy_result"):
            if key in candidate_extra:
                mechanism_report[key] = candidate_extra[key]
    return metrics, mechanism_report, preprocessing_audit


def _select_inputs(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Restrict a frame to the harness-approved input columns (fills missing with NaN)."""
    return frame.reindex(columns=list(feature_columns))


def _load_and_validate_plugin(source: str):
    """Static + dynamic preflight, then import the plugin. Raises ValueError on any violation."""
    # Imported lazily: yield_plugin_contract imports LEAKAGE_COLUMNS from this module.
    from operation_agent.yield_plugin_contract import (
        load_plugin_from_source,
        validate_plugin_feature_declaration,
        validate_plugin_module,
        validate_plugin_source,
    )

    reasons = validate_plugin_source(source)
    if reasons:
        raise ValueError("plugin_source_invalid: " + "; ".join(reasons))
    module = load_plugin_from_source(source)
    reasons = validate_plugin_module(module)
    if reasons:
        raise ValueError("plugin_module_invalid: " + "; ".join(reasons))
    reasons = validate_plugin_feature_declaration(source, getattr(module, "CANDIDATE_SPEC", {}) or {})
    if reasons:
        raise ValueError("plugin_feature_declaration_mismatch: " + "; ".join(reasons))
    return module


def _evaluate_plugin_candidate(
    df: pd.DataFrame,
    feature_columns: list[str],
    strategy_id: str,
    plugin: Any,
    *,
    random_state: int,
    n_splits: int,
    anchor_df: pd.DataFrame | None = None,
    use_plugin_features: bool = True,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray | None]:
    """Evaluate an LLM plugin candidate; mirrors _evaluate_proxy's result shape.

    Returns (result, oof_predictions, anchor_predictions_or_None). The plugin's
    add_features is called fold-locally (fit on train fold with fit_context=None,
    transform on valid/anchor with the returned state) so the anchor never leaks
    into fitting.

    use_plugin_features=False runs the SAME plugin.make_model() on raw input
    columns only (no add_features). This is the same-model ablation baseline so
    the mechanism-feature gain is not confounded with a model-family difference.
    """
    spec = getattr(plugin, "CANDIDATE_SPEC", {}) or {}
    constraints = spec.get("constraints") or []
    nonneg = "nonnegative_prediction" in constraints
    model_family = str(spec.get("model_family") or "hgb")

    def _features(train_in, other_in):
        # Returns (train_feat, other_feat). Fold-local when using plugin features;
        # raw passthrough otherwise (identical model, mechanism features stripped).
        if not use_plugin_features:
            return train_in, other_in
        tr_feat, state = plugin.add_features(train_in, None)
        other_feat, _ = plugin.add_features(other_in, state)
        return tr_feat, other_feat

    y = pd.to_numeric(df[TARGET_COLUMN], errors="coerce").to_numpy(dtype=float)
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_pred = np.zeros(len(df), dtype=float)
    fold_rows: list[dict[str, Any]] = []
    feature_names: list[str] = []

    for fold_idx, (train_idx, valid_idx) in enumerate(splitter.split(df), start=1):
        tr_in = _select_inputs(df.iloc[train_idx], feature_columns)
        va_in = _select_inputs(df.iloc[valid_idx], feature_columns)
        tr_feat, va_feat = _features(tr_in, va_in)
        x_train, x_valid, feature_names = _encode_train_valid(tr_feat, va_feat)
        model = plugin.make_model(random_state + fold_idx)
        model.fit(x_train, y[train_idx])
        pred = np.asarray(model.predict(x_valid), dtype=float)
        if nonneg:
            pred = np.clip(pred, 0.0, None)
        oof_pred[valid_idx] = np.clip(pred, 0.0, None)
        fold_rows.append({"fold": fold_idx, **_metrics(y[valid_idx], pred)})

    oof = _metrics(y, oof_pred)
    result: dict[str, Any] = {
        "strategy_id": strategy_id,
        "benchmark_status": "evaluated",
        "proxy_model": model_family,
        "feature_mode": (
            f"plugin:{spec.get('candidate_id') or 'candidate'}"
            if use_plugin_features
            else f"plugin_rawonly:{spec.get('candidate_id') or 'candidate'}"
        ),
        "candidate_source": "llm_plugin" if use_plugin_features else "llm_plugin_rawonly_ablation",
        "feature_columns_used": feature_names,
        "primary_evaluation": "5-fold OOF" if n_splits == 5 else f"{n_splits}-fold OOF",
        "oof_metrics": oof,
        "fold_metrics": fold_rows,
    }

    anchor_pred = None
    if anchor_df is not None and len(anchor_df) > 0:
        tr_feat, an_feat = _features(_select_inputs(df, feature_columns), _select_inputs(anchor_df, feature_columns))
        x_train, x_anchor, anchor_names = _encode_train_valid(tr_feat, an_feat)
        model = plugin.make_model(random_state + 991)
        model.fit(x_train, y)
        y_anchor = pd.to_numeric(anchor_df[TARGET_COLUMN], errors="coerce").to_numpy(dtype=float)
        anchor_pred = np.asarray(model.predict(x_anchor), dtype=float)
        if nonneg:
            anchor_pred = np.clip(anchor_pred, 0.0, None)
        anchor_pred = np.clip(anchor_pred, 0.0, None)
        result["secondary_evaluation"] = "anchor_validation"
        result["anchor_validation"] = _metrics(y_anchor, anchor_pred)
        result["anchor_feature_columns_used"] = anchor_names

    gen = _anchor_generalization_summary(result)
    result["generalization_gap"] = gen
    result["anchor_generalization_status"] = gen["status"]
    return result, oof_pred, anchor_pred


def _fit_final_plugin_model(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    plugin: Any,
    random_state: int,
) -> dict[str, Any]:
    spec = getattr(plugin, "CANDIDATE_SPEC", {}) or {}
    tr_feat, state = plugin.add_features(_select_inputs(train_df, feature_columns), None)
    encoded = pd.get_dummies(tr_feat, dummy_na=False).apply(pd.to_numeric, errors="coerce")
    imputer = SimpleImputer(strategy="median")
    x_train = imputer.fit_transform(encoded)
    y = pd.to_numeric(train_df[TARGET_COLUMN], errors="coerce").to_numpy(dtype=float)
    model = plugin.make_model(random_state)
    model.fit(x_train.astype(float), y)
    return {
        "model": model,
        "imputer": imputer,
        "encoded_columns": list(encoded.columns),
        "raw_feature_columns": list(feature_columns),
        "feature_state": state,
        "model_family": str(spec.get("model_family") or "hgb"),
        "candidate_id": spec.get("candidate_id"),
        "constraints": list(spec.get("constraints") or []),
        "fit_scope": "full_training_data_only",
    }


def _transform_with_final_plugin_bundle(df: pd.DataFrame, plugin: Any, bundle: dict[str, Any]) -> np.ndarray:
    # Non-None fit_context (possibly {}) => TRANSFORM mode per the plugin contract.
    feat, _ = plugin.add_features(
        _select_inputs(df, list(bundle.get("raw_feature_columns") or [])),
        dict(bundle.get("feature_state") or {}),
    )
    encoded = pd.get_dummies(feat, dummy_na=False).apply(pd.to_numeric, errors="coerce")
    encoded = encoded.reindex(columns=list(bundle.get("encoded_columns") or []), fill_value=0.0)
    return bundle["imputer"].transform(encoded).astype(float)


def execute_selected_benchmark_proxy_artifacts(
    data_path: str | Path,
    candidate_report: dict[str, Any],
    benchmark_report: dict[str, Any],
    run_dir: str | Path,
    *,
    anchor_path: str | Path | None = None,
    random_state: int | None = None,
    n_splits: int = 5,
    fallback_reason: str = "operation_llm_connection_error",
) -> dict[str, Any]:
    """Execute the benchmark-selected proxy and write standard yield artifacts.

    This is an emergency execution path for infrastructure failures in
    OperationAgent code generation. It does not replace LLM-generated AutoML;
    it makes the manager loop auditable when the selected proxy already passed
    CandidateBenchmark and the coding LLM cannot be reached.
    """
    run_path = Path(run_dir)
    metrics_dir = run_path / "metrics"
    pred_dir = run_path / "predictions"
    model_dir = run_path / "trained_models"
    prep_dir = run_path / "preprocessing"
    log_dir = run_path / "logs"
    for path in (metrics_dir, pred_dir, model_dir, prep_dir, log_dir):
        path.mkdir(parents=True, exist_ok=True)

    df, meta = load_yield_dataframe(data_path)
    selected_row = _selected_row_from_benchmark(benchmark_report)
    if not isinstance(selected_row, dict) or not selected_row.get("strategy_id"):
        raise ValueError("No benchmark-selected strategy row is available for fallback execution.")
    protocol = benchmark_report.get("benchmark_protocol") if isinstance(benchmark_report.get("benchmark_protocol"), dict) else {}
    if random_state is None:
        random_state = int(protocol.get("random_state") or 42)
    n_splits = min(max(2, int(n_splits)), max(2, len(df)))

    feature_columns = _clean_feature_list(
        list(selected_row.get("evaluated_base_feature_columns") or selected_row.get("strategy_requested_columns") or []),
        _feature_columns(meta, df),
    )
    if not feature_columns:
        feature_columns = _feature_columns(meta, df)
    proxy_model = str(selected_row.get("proxy_model") or "hgb")
    feature_mode = str(selected_row.get("feature_mode") or "base")
    strategy_id = str(selected_row.get("strategy_id"))

    anchor_df = None
    if anchor_path and Path(anchor_path).exists():
        anchor_df, _anchor_meta = load_yield_dataframe(anchor_path)

    result = _evaluate_proxy(
        df,
        feature_columns,
        strategy_id,
        feature_mode,
        proxy_model,
        random_state=random_state,
        n_splits=n_splits,
        anchor_df=anchor_df,
    )
    base_result = _evaluate_proxy(
        df,
        feature_columns,
        strategy_id + "_without_mechanism",
        "base",
        proxy_model,
        random_state=random_state,
        n_splits=n_splits,
        anchor_df=anchor_df,
    )

    y = pd.to_numeric(df[TARGET_COLUMN], errors="coerce").to_numpy(dtype=float)
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_pred = np.zeros(len(df), dtype=float)
    for fold_idx, (train_idx, valid_idx) in enumerate(splitter.split(df), start=1):
        train_df = df.iloc[train_idx].copy()
        valid_df = df.iloc[valid_idx].copy()
        train_features = _make_raw_features(train_df, feature_columns, feature_mode, train_df)
        valid_features = _make_raw_features(valid_df, feature_columns, feature_mode, train_df)
        x_train, x_valid, _feature_names = _encode_train_valid(train_features, valid_features)
        model = _make_model(proxy_model, random_state + fold_idx, len(train_df))
        model.fit(x_train, y[train_idx])
        oof_pred[valid_idx] = np.clip(np.asarray(model.predict(x_valid), dtype=float), 0.0, None)

    final_bundle = _fit_final_proxy_model(df, feature_columns, feature_mode, proxy_model, random_state + 991)
    dump(final_bundle, model_dir / "benchmark_selected_proxy.joblib")

    prediction_rows = pd.DataFrame(
        {
            INDEX_COLUMN: df[INDEX_COLUMN].astype(str).to_numpy(),
            "split": "oof",
            "y_true": y,
            "y_pred": np.clip(oof_pred, 0.0, None),
            "strategy_id": strategy_id,
            "model": f"benchmark_proxy_{proxy_model}_{feature_mode}",
        }
    )
    anchor_pred = None
    if anchor_df is not None and len(anchor_df) > 0:
        x_anchor = _transform_with_final_bundle(anchor_df, df, final_bundle)
        anchor_pred = np.clip(np.asarray(final_bundle["model"].predict(x_anchor), dtype=float), 0.0, None)
        anchor_rows = pd.DataFrame(
            {
                INDEX_COLUMN: anchor_df[INDEX_COLUMN].astype(str).to_numpy(),
                "split": "anchor_validation",
                "y_true": pd.to_numeric(anchor_df[TARGET_COLUMN], errors="coerce").to_numpy(dtype=float),
                "y_pred": anchor_pred,
                "strategy_id": strategy_id,
                "model": f"benchmark_proxy_{proxy_model}_{feature_mode}",
            }
        )
        prediction_rows = pd.concat([prediction_rows, anchor_rows], ignore_index=True)
    prediction_rows.to_csv(pred_dir / "yield_predictions.csv", index=False)

    baseline_summary = _evaluate_baseline_summary(df, feature_columns, n_splits=n_splits, random_state=random_state)
    metrics, mechanism_report, preprocessing_audit = _build_yield_metrics_payload(
        feature_columns=feature_columns,
        result=result,
        base_result=base_result,
        baseline_summary=baseline_summary,
        final_encoded_columns=list(final_bundle.get("encoded_columns") or []),
        model_label=f"benchmark_proxy_{proxy_model}_{feature_mode}",
        feature_mode_label=feature_mode,
        candidate_source="benchmark_proxy",
        uses_mechanism=feature_mode != "base",
        fallback_reason=fallback_reason,
        anchor_present=anchor_df is not None,
        n_splits=n_splits,
        benchmark_oof_rmse=(selected_row.get("oof_metrics") or {}).get("rmse"),
        benchmark_anchor_r2=(selected_row.get("anchor_validation") or {}).get("r2"),
        benchmark_generalization_gap=selected_row.get("generalization_gap"),
    )
    (metrics_dir / "metrics.json").write_text(json.dumps(_jsonable(metrics), ensure_ascii=False, indent=2), encoding="utf-8")
    (prep_dir / "preprocessing_audit.json").write_text(
        json.dumps(_jsonable(preprocessing_audit), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (log_dir / "mechanism_report.json").write_text(
        json.dumps(_jsonable(mechanism_report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    synthetic_report_path = log_dir / "synthetic_data_report.json"
    if not synthetic_report_path.exists():
        synthetic_report_path.write_text(
            json.dumps(
                {
                    "synthetic_data_used": False,
                    "note": "Real high-fidelity training data were used; no synthetic training data were generated for this fallback execution.",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    source = (
        "# Local OperationAgent fallback executed benchmark-selected proxy because the coding LLM was unavailable.\n"
        "# See knowledge.yield_candidate_benchmark.execute_selected_benchmark_proxy_artifacts.\n"
    )
    (run_path / "generated_code.py").write_text(source, encoding="utf-8")
    return {
        "rcode": 0,
        "action_result": (
            "OperationAgent fallback executed benchmark-selected proxy. "
            f"strategy_id={strategy_id}, proxy_model={proxy_model}, feature_mode={feature_mode}, "
            f"oof_rmse={(metrics.get('oof_metrics') or {}).get('rmse')}, "
            f"anchor_r2={(metrics.get('anchor_validation') or {}).get('r2')}, reason={fallback_reason}."
        ),
        "code": source,
        "error_logs": [],
        "operation_fallback": True,
        "metrics_path": str(metrics_dir / "metrics.json"),
    }


def _reference_arm_summaries(
    df: pd.DataFrame,
    feature_columns: list[str],
    *,
    random_state: int,
    n_splits: int,
    anchor_df: pd.DataFrame | None,
) -> dict[str, Any]:
    """Fixed deterministic ablation arms (the reference/control plugin family).

    reference_raw / reference_packing / reference_sp / reference_packing_sp use
    the same fold-local packing/SP feature code the reference plugin wraps.
    """
    arm_modes = {
        "reference_raw": "base",
        "reference_packing": "packing",
        "reference_sp": "sp_decay",
        "reference_packing_sp": "packing_sp",
    }
    arms: dict[str, Any] = {}
    for label, mode in arm_modes.items():
        try:
            res = _evaluate_proxy(
                df,
                feature_columns,
                label,
                mode,
                "hgb",
                random_state=random_state,
                n_splits=n_splits,
                anchor_df=anchor_df,
            )
            arms[label] = {
                "feature_mode": mode,
                "oof_metrics": res.get("oof_metrics"),
                "anchor_validation": res.get("anchor_validation"),
            }
        except Exception as exc:  # a broken arm must not kill the run
            arms[label] = {"feature_mode": mode, "error": f"{type(exc).__name__}: {exc}"}
    return arms


def execute_plugin_candidate_artifacts(
    data_path: str | Path,
    candidate_report: dict[str, Any],
    benchmark_report: dict[str, Any],
    run_dir: str | Path,
    plugin_source: str,
    *,
    anchor_path: str | Path | None = None,
    random_state: int | None = None,
    n_splits: int = 5,
    run_reference_arms: bool = True,
    source_origin: str = "llm",
) -> dict[str, Any]:
    """Primary yield execution path: run an LLM-generated candidate plugin inside
    the deterministic harness and write the standard yield artifacts.

    Safety: the plugin source is preflighted (static + dynamic) BEFORE import;
    any runtime failure of the plugin marks the plugin candidate as failed and
    falls back to the deterministic reference packing+SP arm so the whole run
    stays auditable instead of crashing.
    """
    run_path = Path(run_dir)
    metrics_dir = run_path / "metrics"
    pred_dir = run_path / "predictions"
    model_dir = run_path / "trained_models"
    prep_dir = run_path / "preprocessing"
    log_dir = run_path / "logs"
    for path in (metrics_dir, pred_dir, model_dir, prep_dir, log_dir):
        path.mkdir(parents=True, exist_ok=True)

    df, meta = load_yield_dataframe(data_path)
    feature_columns = _feature_columns(meta, df)
    protocol = benchmark_report.get("benchmark_protocol") if isinstance(benchmark_report.get("benchmark_protocol"), dict) else {}
    if random_state is None:
        random_state = int(protocol.get("random_state") or 42)
    n_splits = min(max(2, int(n_splits)), max(2, len(df)))
    strategy_id = str(benchmark_report.get("selected_strategy_id") or "llm_plugin_candidate")

    anchor_df = None
    if anchor_path and Path(anchor_path).exists():
        anchor_df, _anchor_meta = load_yield_dataframe(anchor_path)

    # Deterministic hgb+raw reference (used as the same-model ablation base only
    # when we fall back to the hgb reference arm).
    reference_raw_result = _evaluate_proxy(
        df, feature_columns, strategy_id + "_reference_raw", "base", "hgb",
        random_state=random_state, n_splits=n_splits, anchor_df=anchor_df,
    )
    reference_arms = (
        _reference_arm_summaries(df, feature_columns, random_state=random_state, n_splits=n_splits, anchor_df=anchor_df)
        if run_reference_arms
        else {}
    )

    # #2: the REAL benchmark-selected proxy (e.g. ridge+packing_sp). The numbers in
    # the benchmark report were computed on a lightweight SUBSAMPLE for ranking, so
    # we RE-RUN that exact proxy (same proxy_model/feature_mode/feature-columns) on
    # the FULL training data under the same protocol as the plugin. This makes the
    # plugin-vs-benchmark comparison and the confidence gate apples-to-apples.
    bench_selected_row = _selected_row_from_benchmark(benchmark_report) or {}
    sel_proxy_model = bench_selected_row.get("proxy_model")
    sel_feature_mode = bench_selected_row.get("feature_mode")
    sel_feat_cols = _clean_feature_list(
        list(bench_selected_row.get("evaluated_base_feature_columns") or []), feature_columns
    ) or feature_columns
    benchmark_proxy_full = None
    if sel_proxy_model and sel_feature_mode:
        try:
            benchmark_proxy_full = _evaluate_proxy(
                df, sel_feat_cols,
                str(bench_selected_row.get("strategy_id") or "benchmark_selected_proxy"),
                str(sel_feature_mode), str(sel_proxy_model),
                random_state=random_state, n_splits=n_splits, anchor_df=anchor_df,
            )
        except Exception:
            # Re-eval failed -> fall back to the cached (subsampled) report numbers.
            benchmark_proxy_full = None
    if benchmark_proxy_full is not None:
        benchmark_selected_proxy = {
            "strategy_id": bench_selected_row.get("strategy_id"),
            "proxy_model": sel_proxy_model,
            "feature_mode": sel_feature_mode,
            "evaluated_base_feature_columns": sel_feat_cols,
            "oof_metrics": benchmark_proxy_full.get("oof_metrics"),
            "anchor_validation": benchmark_proxy_full.get("anchor_validation"),
            "generalization_gap": _anchor_generalization_summary(benchmark_proxy_full),
            "evaluated_on": "full_training_data_same_protocol_as_plugin",
            "benchmark_report_subsampled": {
                "oof_metrics": bench_selected_row.get("oof_metrics"),
                "anchor_validation": bench_selected_row.get("anchor_validation"),
                "note": "lightweight subsampled numbers from run_candidate_benchmark, kept for audit only",
            },
            "source": "candidate_benchmark_selected_proxy_reevaluated_full_data",
        }
    else:
        benchmark_selected_proxy = {
            "strategy_id": bench_selected_row.get("strategy_id"),
            "proxy_model": sel_proxy_model,
            "feature_mode": sel_feature_mode,
            "oof_metrics": bench_selected_row.get("oof_metrics"),
            "anchor_validation": bench_selected_row.get("anchor_validation"),
            "generalization_gap": bench_selected_row.get("generalization_gap"),
            "source": "candidate_benchmark_selected_proxy_report_cached",
        }
    bench_oof_rmse = _safe_float((benchmark_selected_proxy.get("oof_metrics") or {}).get("rmse"))
    bench_anchor_r2 = _safe_float((benchmark_selected_proxy.get("anchor_validation") or {}).get("r2"))

    # #4: best anchor R2 among all alternative arms (reference family + benchmark proxy)
    # for the tightened confidence gate.
    alt_anchor_r2s = [
        _safe_float((arm.get("anchor_validation") or {}).get("r2"))
        for arm in reference_arms.values()
        if isinstance(arm, dict)
    ] + [bench_anchor_r2]
    alt_anchor_r2s = [x for x in alt_anchor_r2s if x is not None]
    best_alt_anchor_r2 = max(alt_anchor_r2s) if alt_anchor_r2s else None

    # Preflight + import the plugin (hard contract errors raise to the caller).
    plugin = _load_and_validate_plugin(plugin_source)
    spec = dict(getattr(plugin, "CANDIDATE_SPEC", {}) or {})
    candidate_id = str(spec.get("candidate_id") or "candidate")

    plugin_error: str | None = None
    plugin_result: dict[str, Any] | None = None
    oof_pred = None
    try:
        plugin_result, oof_pred, _anchor_pred = _evaluate_plugin_candidate(
            df, feature_columns, strategy_id, plugin,
            random_state=random_state, n_splits=n_splits, anchor_df=anchor_df,
        )
    except Exception as exc:  # plugin runtime failure -> mark failed, keep run alive
        plugin_error = f"{type(exc).__name__}: {exc}"

    if plugin_result is not None:
        # The plugin executed. Its source_origin decides whether this is the
        # genuine main path (LLM-authored) or a deliberate reference-plugin run
        # (used when no LLM plugin was available).
        result = plugin_result
        uses_mechanism = any(fam not in ("raw",) for fam in (spec.get("feature_families") or []))
        # #1 same-model ablation base: SAME plugin.make_model(), raw features only.
        same_model_base, _sb_oof, _sb_anchor = _evaluate_plugin_candidate(
            df, feature_columns, strategy_id + "_rawonly_ablation", plugin,
            random_state=random_state, n_splits=n_splits, anchor_df=anchor_df,
            use_plugin_features=False,
        )
        final_bundle = _fit_final_plugin_model(df, feature_columns, plugin, random_state + 991)
        dump(final_bundle, model_dir / "benchmark_selected_proxy.joblib")
        final_encoded_columns = list(final_bundle.get("encoded_columns") or [])
        anchor_transform = (lambda frame: _transform_with_final_plugin_bundle(frame, plugin, final_bundle))
        feature_mode_label = f"plugin:{candidate_id}"
        if source_origin == "llm":
            candidate_source = "structured_plugin_harness"
            model_label = f"llm_plugin_{candidate_id}"
            harness_fallback_reason = None
        else:
            candidate_source = "plugin_reference_fallback"
            model_label = f"reference_{candidate_id}"
            harness_fallback_reason = "llm_plugin_unavailable_used_reference_plugin"
            plugin_error = plugin_error or harness_fallback_reason
    else:
        # LLM plugin raised at runtime -> deterministic reference packing+SP arm.
        candidate_source = "plugin_reference_fallback"
        harness_fallback_reason = "plugin_runtime_failure_used_reference_arm"
        model_label = "reference_packing_sp_hgb"
        feature_mode_label = "packing_sp"
        result = _evaluate_proxy(
            df, feature_columns, strategy_id + "_reference_packing_sp", "packing_sp", "hgb",
            random_state=random_state, n_splits=n_splits, anchor_df=anchor_df,
        )
        # #1 same-model ablation base: hgb+raw (same hgb family as the packing_sp arm).
        same_model_base = reference_raw_result
        uses_mechanism = True
        final_bundle = _fit_final_proxy_model(df, feature_columns, "packing_sp", "hgb", random_state + 991)
        dump(final_bundle, model_dir / "benchmark_selected_proxy.joblib")
        final_encoded_columns = list(final_bundle.get("encoded_columns") or [])
        anchor_transform = (lambda frame: _transform_with_final_bundle(frame, df, final_bundle))
        # OOF for the reference arm (fold models) for predictions.csv.
        y_all = pd.to_numeric(df[TARGET_COLUMN], errors="coerce").to_numpy(dtype=float)
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        oof_pred = np.zeros(len(df), dtype=float)
        for fold_idx, (tr_idx, va_idx) in enumerate(splitter.split(df), start=1):
            tr_df, va_df = df.iloc[tr_idx].copy(), df.iloc[va_idx].copy()
            tr_f = _make_raw_features(tr_df, feature_columns, "packing_sp", tr_df)
            va_f = _make_raw_features(va_df, feature_columns, "packing_sp", tr_df)
            xtr, xva, _ = _encode_train_valid(tr_f, va_f)
            m = _make_model("hgb", random_state + fold_idx, len(tr_df))
            m.fit(xtr, y_all[tr_idx])
            oof_pred[va_idx] = np.clip(np.asarray(m.predict(xva), dtype=float), 0.0, None)

    # Predictions (OOF for the selected arm; anchor via the saved full-train bundle).
    y = pd.to_numeric(df[TARGET_COLUMN], errors="coerce").to_numpy(dtype=float)
    oof_true = y
    oof_yhat = np.clip(np.asarray(oof_pred, dtype=float), 0.0, None)
    prediction_rows = pd.DataFrame(
        {
            INDEX_COLUMN: df[INDEX_COLUMN].astype(str).to_numpy(),
            "split": "oof",
            "y_true": oof_true,
            "y_pred": oof_yhat,
            # Compatibility aliases so dashboard / guardrail / paper artifacts
            # do not need to branch on column naming.
            "yield_stress_actual": oof_true,
            "yield_stress_predicted": oof_yhat,
            "strategy_id": strategy_id,
            "model": model_label,
        }
    )
    if anchor_df is not None and len(anchor_df) > 0:
        x_anchor = anchor_transform(anchor_df)
        anchor_pred = np.clip(np.asarray(final_bundle["model"].predict(x_anchor), dtype=float), 0.0, None)
        anchor_true = pd.to_numeric(anchor_df[TARGET_COLUMN], errors="coerce").to_numpy(dtype=float)
        anchor_rows = pd.DataFrame(
            {
                INDEX_COLUMN: anchor_df[INDEX_COLUMN].astype(str).to_numpy(),
                "split": "anchor_validation",
                "y_true": anchor_true,
                "y_pred": anchor_pred,
                "yield_stress_actual": anchor_true,
                "yield_stress_predicted": anchor_pred,
                "strategy_id": strategy_id,
                "model": model_label,
            }
        )
        prediction_rows = pd.concat([prediction_rows, anchor_rows], ignore_index=True)
    prediction_rows.to_csv(pred_dir / "yield_predictions.csv", index=False)

    baseline_summary = _evaluate_baseline_summary(df, feature_columns, n_splits=n_splits, random_state=random_state)

    plugin_candidate_block = None
    if plugin_result is not None:
        plugin_candidate_block = {
            "candidate_id": candidate_id,
            "candidate_spec": spec,
            "oof_metrics": plugin_result.get("oof_metrics"),
            "anchor_validation": plugin_result.get("anchor_validation"),
            "generalization_gap": plugin_result.get("generalization_gap"),
        }
    mechanism_descriptor = {
        "name": f"LLM candidate plugin: {candidate_id}" if plugin_result is not None else "reference packing/SP control arm",
        "candidate_source": candidate_source,
        "feature_families": spec.get("feature_families"),
        "physics_hparams": spec.get("physics_hparams"),
        "model_family": spec.get("model_family"),
        "constraints": spec.get("constraints"),
        "columns": feature_columns,
        "source_ids": ["candidate_selection_audit", "llm_candidate_plugin"],
    }
    candidate_extra = {
        "plugin_candidate_result": plugin_candidate_block,
        "reference_arms": reference_arms,
        "plugin_error": plugin_error,
        # #2: honest benchmark_selected_proxy_result = the REAL benchmark-selected
        # proxy (e.g. ridge+packing_sp), not the hgb reference arm.
        "benchmark_selected_proxy_result": benchmark_selected_proxy,
    }

    metrics, mechanism_report, preprocessing_audit = _build_yield_metrics_payload(
        feature_columns=feature_columns,
        result=result,
        base_result=same_model_base,
        baseline_summary=baseline_summary,
        final_encoded_columns=final_encoded_columns,
        model_label=model_label,
        feature_mode_label=feature_mode_label,
        candidate_source=candidate_source,
        uses_mechanism=uses_mechanism,
        fallback_reason=harness_fallback_reason,
        anchor_present=anchor_df is not None,
        n_splits=n_splits,
        benchmark_oof_rmse=bench_oof_rmse,
        benchmark_anchor_r2=bench_anchor_r2,
        benchmark_generalization_gap=bench_selected_row.get("generalization_gap"),
        benchmark_proxy_model=bench_selected_row.get("proxy_model"),
        benchmark_feature_mode=bench_selected_row.get("feature_mode"),
        best_alt_anchor_r2=best_alt_anchor_r2,
        mechanism_descriptor=mechanism_descriptor,
        candidate_extra=candidate_extra,
    )
    (metrics_dir / "metrics.json").write_text(json.dumps(_jsonable(metrics), ensure_ascii=False, indent=2), encoding="utf-8")
    (prep_dir / "preprocessing_audit.json").write_text(json.dumps(_jsonable(preprocessing_audit), ensure_ascii=False, indent=2), encoding="utf-8")
    (log_dir / "mechanism_report.json").write_text(json.dumps(_jsonable(mechanism_report), ensure_ascii=False, indent=2), encoding="utf-8")

    synthetic_report_path = log_dir / "synthetic_data_report.json"
    if not synthetic_report_path.exists():
        synthetic_report_path.write_text(
            json.dumps(
                {"synthetic_data_used": False, "note": "Plugin harness executed on the provided training data."},
                ensure_ascii=False, indent=2,
            ),
            encoding="utf-8",
        )
    (run_path / "generated_code.py").write_text(str(plugin_source or ""), encoding="utf-8")

    oof_m = metrics.get("oof_metrics") or {}
    anchor_m = metrics.get("anchor_validation") or {}
    return {
        "rcode": 0,
        "action_result": (
            f"OperationAgent plugin harness executed candidate_source={candidate_source}, "
            f"model={model_label}, oof_rmse={oof_m.get('rmse')}, anchor_r2={anchor_m.get('r2')}"
            + (f", plugin_error={plugin_error}" if plugin_error else "")
        ),
        "code": str(plugin_source or ""),
        "error_logs": ([plugin_error] if plugin_error else []),
        "operation_fallback": plugin_error is not None,
        "candidate_source": candidate_source,
        "metrics_path": str(metrics_dir / "metrics.json"),
    }
