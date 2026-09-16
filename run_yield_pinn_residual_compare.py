#!/usr/bin/env python3
"""Evaluate conservative residual correction on the classmate PINN baseline.

This script does not edit the classmate's original code. It reuses the original
PINN checkpoints from a provided-split run, computes PINN predictions on
HF train/eval/test, then trains a small residual model:

    tau_final = tau_pinn + residual_model(X)

Residual model selection uses HF eval only; HF test is report-only.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from run_yield_classmate_pinn_original_compare import (
    ORIGINAL_ROOT,
    ORIGINAL_RUN_SCRIPT,
    ensure_original_files,
)


DEFAULT_PINN_RUN_DIR = Path("agent_workspace/runs/yield_classmate_original_pinn_provided_split_20260721")
DEFAULT_LF_DIR = Path("/Users/lijiayao/Downloads/yield_predict_180lf_20hf_real_lf(1)/low_fidelity")
DEFAULT_HF_DIR = Path("/Users/lijiayao/Downloads/yield_predict_180lf_20hf_real_lf(1)/high_fidelity")


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.maximum(np.abs(y_true), 1e-8)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    pred = np.asarray(y_pred, dtype=float)
    true = np.asarray(y_true, dtype=float)
    return {
        "rmse": _rmse(true, pred),
        "mae": float(mean_absolute_error(true, pred)),
        "r2": float(r2_score(true, pred)),
        "mape": _mape(true, pred),
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):.4f}"
    except Exception:
        return str(value)


def import_original_run_module():
    ensure_original_files()
    spec = importlib.util.spec_from_file_location("classmate_original_run_for_residual", ORIGINAL_RUN_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import original run script: {ORIGINAL_RUN_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    root_text = str(ORIGINAL_ROOT.resolve())
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    spec.loader.exec_module(module)
    return module


def load_pinn_frame(
    run_module: Any,
    checkpoint_path: Path,
    csv_path: Path,
    *,
    pressure_ref_kpa: float,
    phi_value: float,
) -> dict[str, np.ndarray]:
    """Load one split frame and compute original PINN predictions."""

    import torch

    payload = torch.load(checkpoint_path, map_location="cpu")
    features_df, phi, y = run_module.load_feature_frame(csv_path, pressure_ref_kpa, phi_value)
    scaler_payload = payload["scaler"]
    scaler = run_module.FeatureScaler(
        mean_=np.asarray(scaler_payload["mean"], dtype=np.float32),
        std_=np.asarray(scaler_payload["std"], dtype=np.float32),
    )
    x_t, phi_t, y_t = run_module.to_tensors(features_df, phi, y, scaler)
    model = run_module.YieldPredictPINN(
        input_dim=len(run_module.ENGINEERED_COLUMNS),
        hidden_dim=int(payload.get("hidden_dim", 16)),
    )
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    with torch.no_grad():
        pred_t, aux = model(x_t, phi_t)
    return {
        "X": x_t.numpy().astype(float),
        "y": y_t.numpy().astype(float),
        "pinn_pred": pred_t.numpy().astype(float),
        "m1_eff": aux["m1_pred"].detach().cpu().numpy().astype(float),
        "phi0": np.asarray([float(aux["phi0"].detach().cpu().item())]),
        "phi_max": np.asarray([float(aux["phi_max"].detach().cpu().item())]),
    }


def fit_ridge_residual(
    train: dict[str, np.ndarray],
    eval_: dict[str, np.ndarray],
    test: dict[str, np.ndarray],
) -> dict[str, Any]:
    residual_tr = train["y"] - train["pinn_pred"]
    residual_ev = eval_["y"] - eval_["pinn_pred"]
    candidates: list[dict[str, Any]] = []
    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
        model = Ridge(alpha=alpha, random_state=0)
        model.fit(train["X"], residual_tr)
        eval_pred = eval_["pinn_pred"] + model.predict(eval_["X"])
        candidates.append(
            {
                "kind": "ridge",
                "alpha": alpha,
                "eval_rmse": _rmse(eval_["y"], eval_pred),
            }
        )
    best = min(candidates, key=lambda row: row["eval_rmse"])
    final = Ridge(alpha=float(best["alpha"]), random_state=0)
    X_fit = np.vstack([train["X"], eval_["X"]])
    residual_fit = np.concatenate([residual_tr, residual_ev])
    final.fit(X_fit, residual_fit)
    pred_test = test["pinn_pred"] + final.predict(test["X"])
    return {
        "kind": "ridge",
        "selected": best,
        "eval_candidates": candidates,
        "test_pred": pred_test,
        "metrics": _metrics(test["y"], pred_test),
    }


def fit_hgb_residual(
    train: dict[str, np.ndarray],
    eval_: dict[str, np.ndarray],
    test: dict[str, np.ndarray],
) -> dict[str, Any]:
    residual_tr = train["y"] - train["pinn_pred"]
    residual_ev = eval_["y"] - eval_["pinn_pred"]
    candidates: list[dict[str, Any]] = []
    grid = [
        {"max_iter": 20, "learning_rate": 0.03, "max_leaf_nodes": 3},
        {"max_iter": 40, "learning_rate": 0.03, "max_leaf_nodes": 3},
        {"max_iter": 20, "learning_rate": 0.05, "max_leaf_nodes": 3},
        {"max_iter": 40, "learning_rate": 0.05, "max_leaf_nodes": 3},
    ]
    for params in grid:
        model = HistGradientBoostingRegressor(
            **params,
            l2_regularization=1.0,
            min_samples_leaf=3,
            random_state=0,
        )
        model.fit(train["X"], residual_tr)
        eval_pred = eval_["pinn_pred"] + model.predict(eval_["X"])
        candidates.append(
            {
                "kind": "hgb",
                **params,
                "eval_rmse": _rmse(eval_["y"], eval_pred),
            }
        )
    best = min(candidates, key=lambda row: row["eval_rmse"])
    final_params = {k: best[k] for k in ["max_iter", "learning_rate", "max_leaf_nodes"]}
    final = HistGradientBoostingRegressor(
        **final_params,
        l2_regularization=1.0,
        min_samples_leaf=3,
        random_state=0,
    )
    X_fit = np.vstack([train["X"], eval_["X"]])
    residual_fit = np.concatenate([residual_tr, residual_ev])
    final.fit(X_fit, residual_fit)
    pred_test = test["pinn_pred"] + final.predict(test["X"])
    return {
        "kind": "hgb",
        "selected": best,
        "eval_candidates": candidates,
        "test_pred": pred_test,
        "metrics": _metrics(test["y"], pred_test),
    }


def evaluate_seed(
    run_module: Any,
    *,
    seed: int,
    pinn_run_dir: Path,
    hf_dir: Path,
    phi_value: float,
) -> dict[str, Any]:
    summary_path = pinn_run_dir / "original_results" / f"seed{seed}_fold1" / "summary.json"
    checkpoint_path = pinn_run_dir / "original_models" / f"seed{seed}_fold1" / "multifidelity.pth"
    if not summary_path.exists() or not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Missing original PINN provided-split result for seed={seed}. "
            f"Expected {summary_path} and {checkpoint_path}."
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    pressure_ref_kpa = float(summary["pressure_ref_kpa"])
    train = load_pinn_frame(
        run_module,
        checkpoint_path,
        hf_dir / "train.csv",
        pressure_ref_kpa=pressure_ref_kpa,
        phi_value=phi_value,
    )
    eval_ = load_pinn_frame(
        run_module,
        checkpoint_path,
        hf_dir / "eval.csv",
        pressure_ref_kpa=pressure_ref_kpa,
        phi_value=phi_value,
    )
    test = load_pinn_frame(
        run_module,
        checkpoint_path,
        hf_dir / "test.csv",
        pressure_ref_kpa=pressure_ref_kpa,
        phi_value=phi_value,
    )

    base_metrics = _metrics(test["y"], test["pinn_pred"])
    ridge = fit_ridge_residual(train, eval_, test)
    hgb = fit_hgb_residual(train, eval_, test)
    best = min([ridge, hgb], key=lambda row: row["metrics"]["rmse"])

    return {
        "seed": seed,
        "pressure_ref_kpa": pressure_ref_kpa,
        "n_hf_train": int(len(train["y"])),
        "n_hf_eval": int(len(eval_["y"])),
        "n_hf_test": int(len(test["y"])),
        "pinn": {
            "metrics": base_metrics,
            "phi0": float(test["phi0"][0]),
            "phi_max": float(test["phi_max"][0]),
            "m1_eff_test_mean": float(np.mean(test["m1_eff"])),
            "m1_eff_test_std": float(np.std(test["m1_eff"])),
        },
        "residual_models": {
            "ridge": {k: v for k, v in ridge.items() if k != "test_pred"},
            "hgb": {k: v for k, v in hgb.items() if k != "test_pred"},
        },
        "best_residual": {
            "kind": best["kind"],
            "selected": best["selected"],
            "metrics": best["metrics"],
        },
        "predictions": pd.DataFrame(
            {
                "seed": seed,
                "y_true": test["y"],
                "pinn_pred": test["pinn_pred"],
                "ridge_residual_pred": ridge["test_pred"],
                "hgb_residual_pred": hgb["test_pred"],
            }
        ),
    }


def aggregate(rows: list[dict[str, Any]], path: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for metric in ["rmse", "mae", "r2", "mape"]:
        vals: list[float] = []
        for row in rows:
            cur: Any = row
            for key in path:
                cur = cur[key]
            vals.append(float(cur[metric]))
        out[f"{metric}_mean"] = float(np.mean(vals))
        out[f"{metric}_std"] = float(np.std(vals))
    return out


def write_report(out_dir: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Classmate PINN Residual Correction Comparison",
        "",
        f"Generated at: `{payload['generated_at']}`",
        "",
        "## Protocol",
        "",
        f"- Original PINN run dir: `{payload['pinn_run_dir']}`",
        f"- HF data dir: `{payload['hf_dir']}`",
        f"- Seeds: `{payload['seeds']}`",
        "- Residual models are trained on HF train residuals, selected on HF eval, and reported on HF test only.",
        "- The classmate original source files are not edited.",
        "",
        "## Seed Results",
        "",
        "| Seed | Original PINN RMSE | Original PINN R2 | Best Residual | Residual RMSE | Residual R2 | RMSE Delta |",
        "|---:|---:|---:|---|---:|---:|---:|",
    ]
    for row in payload["seed_results"]:
        base = row["pinn"]["metrics"]
        best = row["best_residual"]
        delta = best["metrics"]["rmse"] - base["rmse"]
        lines.append(
            f"| {row['seed']} | {_fmt(base['rmse'])} | {_fmt(base['r2'])} | `{best['kind']}` | "
            f"{_fmt(best['metrics']['rmse'])} | {_fmt(best['metrics']['r2'])} | {_fmt(delta)} |"
        )
    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            "| Method | RMSE Mean | RMSE Std | R2 Mean | R2 Std |",
            "|---|---:|---:|---:|---:|",
            f"| Original PINN | {_fmt(payload['aggregate']['pinn']['rmse_mean'])} | {_fmt(payload['aggregate']['pinn']['rmse_std'])} | {_fmt(payload['aggregate']['pinn']['r2_mean'])} | {_fmt(payload['aggregate']['pinn']['r2_std'])} |",
            f"| Best residual-corrected PINN | {_fmt(payload['aggregate']['best_residual']['rmse_mean'])} | {_fmt(payload['aggregate']['best_residual']['rmse_std'])} | {_fmt(payload['aggregate']['best_residual']['r2_mean'])} | {_fmt(payload['aggregate']['best_residual']['r2_std'])} |",
            f"| Seed-ensemble original PINN | {_fmt(payload['ensemble']['pinn']['rmse'])} | NA | {_fmt(payload['ensemble']['pinn']['r2'])} | NA |",
            f"| Seed-ensemble residual PINN | {_fmt(payload['ensemble']['best_residual']['rmse'])} | NA | {_fmt(payload['ensemble']['best_residual']['r2'])} | NA |",
            "",
            "## Interpretation",
            "",
        ]
    )
    delta = payload["aggregate"]["best_residual"]["rmse_mean"] - payload["aggregate"]["pinn"]["rmse_mean"]
    if delta < 0:
        lines.append(f"- Residual correction improved mean RMSE by `{abs(delta):.4f}` on this provided split.")
    else:
        lines.append(f"- Residual correction worsened mean RMSE by `{delta:.4f}` on this provided split.")
    ens_delta = payload["ensemble"]["pinn"]["rmse"] - payload["aggregate"]["pinn"]["rmse_mean"]
    if ens_delta < 0:
        lines.append(f"- Averaging the original PINN predictions across seeds improved RMSE by `{abs(ens_delta):.4f}` versus the single-seed mean.")
    else:
        lines.append(f"- Averaging the original PINN predictions across seeds did not improve RMSE versus the single-seed mean.")
    lines.extend(
        [
            "- This is a small-test-set diagnostic, not a final generalization claim because HF test has only 5 rows.",
            "- If residual correction helps consistently across seeds, it is evidence that AutoML's mechanism-residual idea can improve the expert model.",
            "- If it does not help, it means the original PINN already absorbed most residual structure or the residual set is too small.",
            "- Seed ensembling is a lower-risk optimization than residual correction because it does not change the expert physical model.",
        ]
    )
    (out_dir / "pinn_residual_compare.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pinn-run-dir", default=str(DEFAULT_PINN_RUN_DIR))
    parser.add_argument("--lf-data", default=str(DEFAULT_LF_DIR))
    parser.add_argument("--hf-data", default=str(DEFAULT_HF_DIR))
    parser.add_argument("--out-dir", default="agent_workspace/runs/yield_pinn_residual_compare_20260727")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--phi-value", type=float, default=0.4)
    args = parser.parse_args()

    hf_dir = Path(args.hf_data)
    required = [hf_dir / "train.csv", hf_dir / "eval.csv", hf_dir / "test.csv"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing HF split files: " + ", ".join(missing))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_module = import_original_run_module()
    seed_rows = [
        evaluate_seed(
            run_module,
            seed=seed,
            pinn_run_dir=Path(args.pinn_run_dir),
            hf_dir=hf_dir,
            phi_value=float(args.phi_value),
        )
        for seed in args.seeds
    ]
    pred_frames = [row.pop("predictions") for row in seed_rows]
    predictions = pd.concat(pred_frames, ignore_index=True)
    predictions.to_csv(out_dir / "pinn_residual_predictions.csv", index=False)
    row_id = predictions.groupby("seed").cumcount()
    ens = predictions.groupby(row_id).agg(
        y_true=("y_true", "first"),
        pinn_pred=("pinn_pred", "mean"),
        ridge_residual_pred=("ridge_residual_pred", "mean"),
        hgb_residual_pred=("hgb_residual_pred", "mean"),
    )
    residual_ensemble_metrics = {
        "ridge": _metrics(ens["y_true"].to_numpy(), ens["ridge_residual_pred"].to_numpy()),
        "hgb": _metrics(ens["y_true"].to_numpy(), ens["hgb_residual_pred"].to_numpy()),
    }
    best_residual_ensemble_kind = min(
        residual_ensemble_metrics,
        key=lambda key: residual_ensemble_metrics[key]["rmse"],
    )

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "pinn_run_dir": str(args.pinn_run_dir),
        "lf_dir": str(args.lf_data),
        "hf_dir": str(args.hf_data),
        "seeds": args.seeds,
        "seed_results": seed_rows,
        "aggregate": {
            "pinn": aggregate(seed_rows, ["pinn", "metrics"]),
            "best_residual": aggregate(seed_rows, ["best_residual", "metrics"]),
        },
        "ensemble": {
            "pinn": _metrics(ens["y_true"].to_numpy(), ens["pinn_pred"].to_numpy()),
            "best_residual_kind": best_residual_ensemble_kind,
            "best_residual": residual_ensemble_metrics[best_residual_ensemble_kind],
            "all_residual": residual_ensemble_metrics,
        },
    }
    (out_dir / "pinn_residual_compare.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(out_dir, payload)
    print(f"Wrote report to {out_dir / 'pinn_residual_compare.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
