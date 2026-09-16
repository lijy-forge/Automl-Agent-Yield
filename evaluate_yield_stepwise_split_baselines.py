#!/usr/bin/env python3
"""Evaluate lightweight tabular baselines on stepwise LF/HF splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from knowledge.yield_schema import load_yield_dataframe


TARGET = "yield_stress"
DEFAULT_SPLIT_ROOT = Path("agent_workspace/data/yield_stepwise_multifidelity_20260804/splits")
DEFAULT_OUT_DIR = Path("agent_workspace/runs/yield_stepwise_baselines_20260804")


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": _rmse(y_true, y_pred),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _load_xy(path: Path) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    df, meta = load_yield_dataframe(path)
    features = list(meta.get("feature_columns") or [])
    x = df[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = df[TARGET].to_numpy(dtype=float)
    return x, y, features


def _models(seed: int) -> dict[str, Any]:
    return {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "ridge_strong": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "elastic_net": make_pipeline(StandardScaler(), ElasticNet(alpha=0.01, l1_ratio=0.2, max_iter=20000, random_state=seed)),
        "hgb": HistGradientBoostingRegressor(
            max_iter=120,
            learning_rate=0.05,
            max_leaf_nodes=7,
            min_samples_leaf=5,
            l2_regularization=1.0,
            random_state=seed,
        ),
        "rf": RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            max_features=0.7,
            random_state=seed,
        ),
    }


def _align_features(frames: list[pd.DataFrame]) -> list[pd.DataFrame]:
    common = set(frames[0].columns)
    for frame in frames[1:]:
        common &= set(frame.columns)
    ordered = [column for column in frames[0].columns if column in common]
    return [frame[ordered] for frame in frames]


def evaluate_mode(split_dir: Path, seed: int) -> dict[str, Any]:
    hf_dir = split_dir / "high_fidelity"
    lf_dir = split_dir / "low_fidelity"
    x_hf_train, y_hf_train, features = _load_xy(hf_dir / "train.csv")
    x_hf_eval, y_hf_eval, _ = _load_xy(hf_dir / "eval.csv")
    x_hf_test, y_hf_test, _ = _load_xy(hf_dir / "test.csv")
    x_lf_train, y_lf_train, _ = _load_xy(lf_dir / "train.csv")
    x_lf_test, y_lf_test, _ = _load_xy(lf_dir / "test.csv")

    x_lf_train, x_lf_test, x_hf_train, x_hf_eval, x_hf_test = _align_features(
        [x_lf_train, x_lf_test, x_hf_train, x_hf_eval, x_hf_test]
    )
    x_hf_train_eval = pd.concat([x_hf_train, x_hf_eval], ignore_index=True)
    y_hf_train_eval = np.concatenate([y_hf_train, y_hf_eval])
    x_lf_plus_hf = pd.concat([x_lf_train, x_hf_train, x_hf_eval], ignore_index=True)
    y_lf_plus_hf = np.concatenate([y_lf_train, y_hf_train, y_hf_eval])

    train_sets = {
        "lf_train_only": (x_lf_train, y_lf_train),
        "hf_train_eval_only": (x_hf_train_eval, y_hf_train_eval),
        "lf_plus_hf_train_eval": (x_lf_plus_hf, y_lf_plus_hf),
    }
    results: dict[str, dict[str, dict[str, float]]] = {}
    for train_name, (x_train, y_train) in train_sets.items():
        results[train_name] = {}
        for model_name, model in _models(seed).items():
            model.fit(x_train, y_train)
            pred = model.predict(x_hf_test)
            results[train_name][model_name] = _metrics(y_hf_test, pred)

    best_rows = []
    for train_name, model_rows in results.items():
        best_model, best_metrics = min(model_rows.items(), key=lambda item: item[1]["rmse"])
        best_rows.append({"train_source": train_name, "model": best_model, **best_metrics})
    best_overall = min(best_rows, key=lambda row: row["rmse"])

    return {
        "split_dir": str(split_dir),
        "seed": int(seed),
        "n_features": int(x_hf_test.shape[1]),
        "n": {
            "hf_train": int(len(y_hf_train)),
            "hf_eval": int(len(y_hf_eval)),
            "hf_test": int(len(y_hf_test)),
            "lf_train": int(len(y_lf_train)),
            "lf_test": int(len(y_lf_test)),
        },
        "target_summary": {
            "hf_test_min": float(np.min(y_hf_test)),
            "hf_test_max": float(np.max(y_hf_test)),
            "hf_test_mean": float(np.mean(y_hf_test)),
            "hf_test_std": float(np.std(y_hf_test, ddof=1)) if len(y_hf_test) > 1 else 0.0,
        },
        "results": results,
        "best_by_train_source": best_rows,
        "best_overall": best_overall,
        "feature_sample": list(x_hf_test.columns[:25]),
    }


def _write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = ["# Stepwise LF/HF 表格基线结果", ""]
    for mode, report in payload["modes"].items():
        lines.extend([
            f"## {mode}",
            "",
            f"- split: `{report['split_dir']}`",
            f"- features: {report['n_features']}",
            f"- HF train/eval/test: {report['n']['hf_train']} / {report['n']['hf_eval']} / {report['n']['hf_test']}",
            f"- LF train/test: {report['n']['lf_train']} / {report['n']['lf_test']}",
            f"- HF test std: {report['target_summary']['hf_test_std']:.4f}",
            "",
            "| 训练来源 | 最佳模型 | RMSE | MAE | R2 |",
            "|---|---|---:|---:|---:|",
        ])
        for row in report["best_by_train_source"]:
            lines.append(
                f"| {row['train_source']} | {row['model']} | {row['rmse']:.4f} | {row['mae']:.4f} | {row['r2']:.4f} |"
            )
        best = report["best_overall"]
        lines.extend([
            "",
            f"Best overall: `{best['train_source']}::{best['model']}` RMSE={best['rmse']:.4f}, MAE={best['mae']:.4f}, R2={best['r2']:.4f}",
            "",
        ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate tabular baselines on stepwise paired/group-holdout splits.")
    parser.add_argument("--split-root", default=str(DEFAULT_SPLIT_ROOT))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--seed", type=int, default=20260804)
    args = parser.parse_args()

    split_root = Path(args.split_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    modes = {}
    for mode in ["paired", "group_holdout"]:
        split_dir = split_root / f"{mode}_seed_{args.seed}"
        modes[mode] = evaluate_mode(split_dir, int(args.seed))

    payload = {
        "split_root": str(split_root),
        "out_dir": str(out_dir),
        "seed": int(args.seed),
        "modes": modes,
    }
    json_path = out_dir / "stepwise_split_baselines.json"
    md_path = out_dir / "stepwise_split_baselines.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(payload, md_path)
    print(json.dumps({"json": str(json_path), "markdown": str(md_path), "summary": {k: v["best_overall"] for k, v in modes.items()}}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
