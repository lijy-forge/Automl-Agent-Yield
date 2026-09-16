"""Fixed baseline comparisons for yield-stress regression.

Generated yield AutoML code should call this module instead of asking the LLM
to reinvent baseline code each run. The final model can still be freely
generated; these baselines are stable references for comparison.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class BaselineResultBundle(dict):
    """Dict-compatible baseline payload with list-add convenience for LLM code."""

    def as_list(self) -> list[dict[str, Any]]:
        value = self.get("baseline_results", [])
        return value if isinstance(value, list) else []

    def __add__(self, other: Any) -> list[Any]:
        if isinstance(other, list):
            return self.as_list() + other
        return self.as_list() + [other]

    def __radd__(self, other: Any) -> list[Any]:
        if isinstance(other, list):
            return other + self.as_list()
        return [other] + self.as_list()


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.maximum(np.abs(y_true), 1e-8)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def _prepare_features(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    features = df[feature_columns].copy()
    for column in features.columns:
        features[column] = pd.to_numeric(features[column], errors="ignore")
    return pd.get_dummies(features, dummy_na=False)


def _infer_default_feature_columns(df: pd.DataFrame, target_column: str) -> list[str]:
    excluded = {
        "sample_id",
        target_column,
        "yield_stress",
        "Tau0_Pa",
        "tau_Pa",
        "target",
        "label",
        "phi_max",
        "phi_max_eff",
        "phi_max_eff_reference",
        "m1_true",
        "m1_lf",
        "plastic_viscosity",
        "plastic_viscosity_pa_s",
        "source",
        "data_fidelity",
    }
    return [col for col in df.columns if col not in excluded and not str(col).endswith("_reference")]


def _supports_monotonic_hgb() -> bool:
    try:
        HistGradientBoostingRegressor(monotonic_cst=[0])
        return True
    except TypeError:
        return False


def _evaluate_factory(
    name: str,
    factory: Callable[[], RegressorMixin],
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int,
    n_repeats: int,
    random_state: int,
    uses_mechanism_constraint: bool = False,
    notes: str = "",
) -> dict[str, Any]:
    splitter = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    fold_metrics: list[dict[str, float]] = []
    oof_sum = np.zeros(len(y), dtype=float)
    oof_count = np.zeros(len(y), dtype=float)

    for train_idx, val_idx in splitter.split(X):
        model = factory()
        model.fit(X[train_idx], y[train_idx])
        pred = np.asarray(model.predict(X[val_idx]), dtype=float)
        pred = np.clip(pred, 0.0, None)
        oof_sum[val_idx] += pred
        oof_count[val_idx] += 1.0
        fold_metrics.append(
            {
                "rmse": float(np.sqrt(mean_squared_error(y[val_idx], pred))),
                "mae": float(mean_absolute_error(y[val_idx], pred)),
                "r2": float(r2_score(y[val_idx], pred)),
                "mape": _mape(y[val_idx], pred),
            }
        )

    oof_pred = oof_sum / np.maximum(oof_count, 1.0)
    return {
        "name": name,
        "fixed_baseline": True,
        "uses_mechanism_constraint": bool(uses_mechanism_constraint),
        "notes": notes,
        "mean_rmse": float(np.mean([m["rmse"] for m in fold_metrics])),
        "std_rmse": float(np.std([m["rmse"] for m in fold_metrics])),
        "mean_mae": float(np.mean([m["mae"] for m in fold_metrics])),
        "std_mae": float(np.std([m["mae"] for m in fold_metrics])),
        "mean_r2": float(np.mean([m["r2"] for m in fold_metrics])),
        "std_r2": float(np.std([m["r2"] for m in fold_metrics])),
        "mean_mape": float(np.mean([m["mape"] for m in fold_metrics])),
        "std_mape": float(np.std([m["mape"] for m in fold_metrics])),
        "oof_rmse": float(np.sqrt(mean_squared_error(y, oof_pred))),
        "oof_mae": float(mean_absolute_error(y, oof_pred)),
        "oof_r2": float(r2_score(y, oof_pred)),
        "oof_mape": _mape(y, oof_pred),
    }


def run_fixed_yield_baselines(
    df: Any,
    feature_columns: Any = None,
    target_column: str = "yield_stress",
    *,
    cv: Any | None = None,
    enable_sp_monotonic: bool = False,
    n_splits: int = 5,
    n_repeats: int = 1,
    random_state: int = 42,
) -> dict[str, Any]:
    """Evaluate stable baselines under the primary OOF protocol.

    `enable_sp_monotonic` should only be true when `sp_percent` is known to mean
    dispersant/superplasticizer/plasticizer dosage, not curing-agent dosage.

    The preferred call is `run_fixed_yield_baselines(df, feature_columns)`, but
    this function intentionally accepts common generated-code variants too:
    `run_fixed_yield_baselines(path)` and `run_fixed_yield_baselines(X, y)`.
    """

    y_override: np.ndarray | None = None
    schema_feature_columns: list[str] | None = None

    if isinstance(df, (str, os.PathLike, Path)):
        from knowledge.yield_schema import load_yield_dataframe

        df, meta = load_yield_dataframe(df)
        schema_feature_columns = list(meta.get("feature_columns") or [])
    elif not isinstance(df, pd.DataFrame):
        if feature_columns is None:
            raise TypeError(
                "run_fixed_yield_baselines expected a DataFrame/CSV path, or a feature matrix plus y as the second argument."
            )
        y_override = np.asarray(feature_columns, dtype=float).reshape(-1)
        arr = np.asarray(df)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        df = pd.DataFrame(arr, columns=[f"feature_{idx}" for idx in range(arr.shape[1])])
        feature_columns = list(df.columns)

    if feature_columns is not None:
        looks_like_feature_names = (
            isinstance(feature_columns, (list, tuple, pd.Index))
            and all(isinstance(item, str) for item in feature_columns)
        )
        if looks_like_feature_names:
            feature_columns = list(feature_columns)
        elif y_override is None:
            y_override = np.asarray(feature_columns, dtype=float).reshape(-1)
            feature_columns = _infer_default_feature_columns(df, target_column)

    if feature_columns is None:
        if schema_feature_columns:
            feature_columns = schema_feature_columns
        else:
            feature_columns = _infer_default_feature_columns(df, target_column)

    missing_features = [col for col in feature_columns if col not in df.columns]
    if missing_features:
        raise ValueError(f"Feature columns not found in dataframe: {missing_features}")

    if cv is not None:
        if isinstance(cv, int):
            n_splits = int(cv)
            n_repeats = 1
        elif hasattr(cv, "get_n_splits"):
            try:
                n_splits = int(cv.get_n_splits())
                n_repeats = 1
            except Exception:
                pass

    X_df = _prepare_features(df, feature_columns)
    X = X_df.to_numpy(dtype=float)
    if y_override is not None:
        y = y_override
    else:
        y = pd.to_numeric(df[target_column], errors="coerce").to_numpy(dtype=float)
    if len(y) != len(df):
        raise ValueError(f"Target length {len(y)} does not match feature rows {len(df)}.")
    if not np.all(np.isfinite(y)):
        raise ValueError(f"Target column {target_column!r} contains non-finite values.")

    baselines: list[tuple[str, Callable[[], RegressorMixin], bool, str]] = [
        (
            "baseline_mean_dummy",
            lambda: DummyRegressor(strategy="mean"),
            False,
            "No-skill mean predictor.",
        ),
        (
            "baseline_ridge_linear",
            lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
            False,
            "Simple linear baseline with standardized features.",
        ),
        (
            "baseline_random_forest",
            lambda: RandomForestRegressor(
                n_estimators=300,
                max_depth=8,
                min_samples_leaf=2,
                random_state=random_state,
                n_jobs=-1,
            ),
            False,
            "Unconstrained nonlinear tree ensemble baseline.",
        ),
        (
            "baseline_hist_gradient_boosting",
            lambda: HistGradientBoostingRegressor(
                max_iter=300,
                max_depth=6,
                learning_rate=0.08,
                l2_regularization=1e-3,
                random_state=random_state,
            ),
            False,
            "Unconstrained histogram gradient boosting baseline.",
        ),
    ]

    if enable_sp_monotonic and "sp_percent" in X_df.columns and _supports_monotonic_hgb():
        monotonic_cst = [0] * len(X_df.columns)
        monotonic_cst[list(X_df.columns).index("sp_percent")] = -1
        baselines.append(
            (
                "baseline_monotonic_hgb_sp_nonincreasing",
                lambda: HistGradientBoostingRegressor(
                    max_iter=300,
                    max_depth=6,
                    learning_rate=0.08,
                    l2_regularization=1e-3,
                    monotonic_cst=monotonic_cst,
                    random_state=random_state,
                ),
                True,
                "Mechanism-aware baseline: sp_percent constrained non-increasing.",
            )
        )

    results = [
        _evaluate_factory(
            name,
            factory,
            X,
            y,
            n_splits=n_splits,
            n_repeats=n_repeats,
            random_state=random_state,
            uses_mechanism_constraint=uses_mechanism,
            notes=notes,
        )
        for name, factory, uses_mechanism, notes in baselines
    ]
    best = min(results, key=lambda item: item["mean_rmse"])
    protocol = (
        f"{n_splits}-fold OOF"
        if int(n_repeats) == 1
        else f"Repeated {n_splits}-fold OOF ({n_repeats} repeats)"
    )
    return BaselineResultBundle({
        "baseline_protocol": protocol,
        "feature_columns": list(X_df.columns),
        "enable_sp_monotonic": bool(enable_sp_monotonic),
        "baseline_results": results,
        "best_baseline": best["name"],
    })
