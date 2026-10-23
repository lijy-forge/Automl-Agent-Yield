#!/usr/bin/env python3
"""Lightweight sklearn baselines on the prepared new-CSV classmate split."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SPLIT_ROOT = Path("agent_workspace/data/yield_predict_new_csv_classmate_split_20260803")
VARIANT_ROOT = Path("agent_workspace/external_baselines/gate_original_learned")
RUN_SCRIPT = VARIANT_ROOT / "multi_fidelity" / "src" / "experiments" / "run_yield_predict_experiment.py"
OUT_PATH = Path("agent_workspace/runs/yield_pinn_newcsv_20260803/newcsv_sklearn_baselines.json")


def _import_run_module():
    root_text = str(VARIANT_ROOT.resolve())
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    spec = importlib.util.spec_from_file_location("newcsv_baseline_run_module", RUN_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {RUN_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "rmse": _rmse(y_true, y_pred),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _load_features(run_module, path: Path, pressure_ref_kpa: float):
    features, _phi, y = run_module.load_feature_frame(path, pressure_ref_kpa, 0.4)
    return features.to_numpy(dtype=np.float32), y.astype(float)


def main() -> int:
    run_module = _import_run_module()
    lf_dir = SPLIT_ROOT / "low_fidelity"
    hf_dir = SPLIT_ROOT / "high_fidelity"
    lf_train_df = pd.read_csv(lf_dir / "train.csv")
    hf_train_df = pd.read_csv(hf_dir / "train.csv")
    pressure_ref_kpa = float(pd.concat([lf_train_df["internal_pressure_kpa"], hf_train_df["internal_pressure_kpa"]]).median())

    x_lf, y_lf = _load_features(run_module, lf_dir / "train.csv", pressure_ref_kpa)
    x_hf_tr, y_hf_tr = _load_features(run_module, hf_dir / "train.csv", pressure_ref_kpa)
    x_hf_ev, y_hf_ev = _load_features(run_module, hf_dir / "eval.csv", pressure_ref_kpa)
    x_hf_te, y_hf_te = _load_features(run_module, hf_dir / "test.csv", pressure_ref_kpa)

    train_sets = {
        "lf_train_only": (x_lf, y_lf),
        "lf_plus_hf_train_eval": (
            np.vstack([x_lf, x_hf_tr, x_hf_ev]),
            np.concatenate([y_lf, y_hf_tr, y_hf_ev]),
        ),
        "hf_train_eval_only": (
            np.vstack([x_hf_tr, x_hf_ev]),
            np.concatenate([y_hf_tr, y_hf_ev]),
        ),
    }
    models = {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "hgb": HistGradientBoostingRegressor(max_iter=120, learning_rate=0.05, max_leaf_nodes=7, random_state=0),
        "rf": RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=0),
    }

    results: dict[str, dict[str, dict[str, float]]] = {}
    for train_name, (x_train, y_train) in train_sets.items():
        results[train_name] = {}
        for model_name, model in models.items():
            model.fit(x_train, y_train)
            pred = model.predict(x_hf_te)
            results[train_name][model_name] = _metrics(y_hf_te, pred)

    payload = {
        "split_root": str(SPLIT_ROOT),
        "pressure_ref_kpa": pressure_ref_kpa,
        "hf_test_n": int(len(y_hf_te)),
        "results": results,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
