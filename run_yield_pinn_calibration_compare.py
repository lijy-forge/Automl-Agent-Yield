#!/usr/bin/env python3
"""Low-dimensional calibration diagnostics for the classmate PINN baseline.

This script tests whether the saved original PINN predictions have a simple
systematic bias that can be corrected without changing the expert model:

    identity: tau_final = tau_pinn
    bias:     tau_final = tau_pinn + b
    affine:   tau_final = a * tau_pinn + b

Calibration type is selected on HF eval only; HF test is report-only.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from run_yield_pinn_residual_compare import (
    DEFAULT_HF_DIR,
    DEFAULT_PINN_RUN_DIR,
    import_original_run_module,
    load_pinn_frame,
)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.maximum(np.abs(y_true), 1e-8)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
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


class Calibrator:
    def __init__(self, kind: str, params: dict[str, Any] | None = None):
        self.kind = kind
        self.params = params or {}
        self.offset_: float = 0.0
        self.scale_: float = 1.0
        self.intercept_: float = 0.0
        self.model_: Any = None

    def fit(self, pred: np.ndarray, y: np.ndarray) -> "Calibrator":
        pred = np.asarray(pred, dtype=float).reshape(-1)
        y = np.asarray(y, dtype=float).reshape(-1)
        if self.kind == "identity":
            return self
        if self.kind == "bias":
            self.offset_ = float(np.mean(y - pred))
            return self
        x = pred.reshape(-1, 1)
        if self.kind == "affine_ols":
            self.model_ = LinearRegression().fit(x, y)
            return self
        if self.kind == "affine_ridge":
            alpha = float(self.params.get("alpha", 1.0))
            self.model_ = Ridge(alpha=alpha).fit(x, y)
            return self
        raise ValueError(f"Unknown calibrator kind: {self.kind}")

    def predict(self, pred: np.ndarray) -> np.ndarray:
        pred = np.asarray(pred, dtype=float).reshape(-1)
        if self.kind == "identity":
            return pred
        if self.kind == "bias":
            return pred + self.offset_
        if self.kind in {"affine_ols", "affine_ridge"}:
            return np.asarray(self.model_.predict(pred.reshape(-1, 1)), dtype=float)
        raise ValueError(f"Unknown calibrator kind: {self.kind}")

    def describe(self) -> dict[str, Any]:
        out = {"kind": self.kind, **self.params}
        if self.kind == "bias":
            out["offset"] = self.offset_
        if self.kind in {"affine_ols", "affine_ridge"} and self.model_ is not None:
            out["coef"] = float(self.model_.coef_[0])
            out["intercept"] = float(self.model_.intercept_)
        return out


def candidate_calibrators() -> list[Calibrator]:
    return [
        Calibrator("identity"),
        Calibrator("bias"),
        Calibrator("affine_ols"),
        Calibrator("affine_ridge", {"alpha": 0.1}),
        Calibrator("affine_ridge", {"alpha": 1.0}),
        Calibrator("affine_ridge", {"alpha": 10.0}),
        Calibrator("affine_ridge", {"alpha": 100.0}),
    ]


def select_and_refit_calibrator(train: dict[str, np.ndarray], eval_: dict[str, np.ndarray]) -> tuple[Calibrator, list[dict[str, Any]]]:
    eval_rows: list[dict[str, Any]] = []
    for cal in candidate_calibrators():
        cal.fit(train["pinn_pred"], train["y"])
        pred_eval = cal.predict(eval_["pinn_pred"])
        eval_rows.append({**cal.describe(), "eval_rmse": _rmse(eval_["y"], pred_eval), "eval_r2": float(r2_score(eval_["y"], pred_eval))})
    best_row = min(eval_rows, key=lambda row: row["eval_rmse"])
    best = Calibrator(best_row["kind"], {k: v for k, v in best_row.items() if k == "alpha"})
    fit_pred = np.concatenate([train["pinn_pred"], eval_["pinn_pred"]])
    fit_y = np.concatenate([train["y"], eval_["y"]])
    best.fit(fit_pred, fit_y)
    return best, eval_rows


def evaluate_seed(
    run_module: Any,
    *,
    seed: int,
    pinn_run_dir: Path,
    hf_dir: Path,
    phi_value: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    summary_path = pinn_run_dir / "original_results" / f"seed{seed}_fold1" / "summary.json"
    checkpoint_path = pinn_run_dir / "original_models" / f"seed{seed}_fold1" / "multifidelity.pth"
    if not summary_path.exists() or not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing original PINN result for seed={seed}: {summary_path}, {checkpoint_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    pressure_ref_kpa = float(summary["pressure_ref_kpa"])
    train = load_pinn_frame(run_module, checkpoint_path, hf_dir / "train.csv", pressure_ref_kpa=pressure_ref_kpa, phi_value=phi_value)
    eval_ = load_pinn_frame(run_module, checkpoint_path, hf_dir / "eval.csv", pressure_ref_kpa=pressure_ref_kpa, phi_value=phi_value)
    test = load_pinn_frame(run_module, checkpoint_path, hf_dir / "test.csv", pressure_ref_kpa=pressure_ref_kpa, phi_value=phi_value)

    cal, eval_rows = select_and_refit_calibrator(train, eval_)
    pred_test_cal = cal.predict(test["pinn_pred"])
    pred_frame = pd.DataFrame(
        {
            "seed": seed,
            "row_id": np.arange(len(test["y"])),
            "y_true": test["y"],
            "pinn_pred": test["pinn_pred"],
            "calibrated_pred": pred_test_cal,
        }
    )
    row = {
        "seed": seed,
        "n_hf_train": int(len(train["y"])),
        "n_hf_eval": int(len(eval_["y"])),
        "n_hf_test": int(len(test["y"])),
        "original": _metrics(test["y"], test["pinn_pred"]),
        "selected_calibrator": cal.describe(),
        "eval_candidates": eval_rows,
        "calibrated": _metrics(test["y"], pred_test_cal),
    }
    row["delta_rmse"] = float(row["calibrated"]["rmse"] - row["original"]["rmse"])
    return row, pred_frame


def aggregate(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for metric in ["rmse", "mae", "r2", "mape"]:
        vals = [float(row[key][metric]) for row in rows]
        out[f"{metric}_mean"] = float(np.mean(vals))
        out[f"{metric}_std"] = float(np.std(vals))
    return out


def ensemble_metrics(predictions: pd.DataFrame) -> dict[str, dict[str, float]]:
    grouped = predictions.groupby("row_id")
    y = grouped["y_true"].first().to_numpy(dtype=float)
    out: dict[str, dict[str, float]] = {}
    for col in ["pinn_pred", "calibrated_pred"]:
        out[col] = _metrics(y, grouped[col].mean().to_numpy(dtype=float))
    return out


def write_report(out_dir: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Classmate PINN Calibration Comparison",
        "",
        f"Generated at: `{payload['generated_at']}`",
        "",
        "## Protocol",
        "",
        f"- Original PINN run dir: `{payload['pinn_run_dir']}`",
        f"- HF data dir: `{payload['hf_dir']}`",
        f"- Seeds: `{payload['seeds']}`",
        "- Calibration type is selected on HF eval; HF test is report-only.",
        "- Calibration is low-dimensional and does not edit or retrain the expert PINN.",
        "",
        "## Seed Results",
        "",
        "| Seed | Original RMSE | Original R2 | Selected calibration | Calibrated RMSE | Calibrated R2 | RMSE Delta |",
        "|---:|---:|---:|---|---:|---:|---:|",
    ]
    for row in payload["seed_results"]:
        selected = row["selected_calibrator"]
        label = selected["kind"] if "alpha" not in selected else f"{selected['kind']}(alpha={selected['alpha']})"
        lines.append(
            f"| {row['seed']} | {_fmt(row['original']['rmse'])} | {_fmt(row['original']['r2'])} | `{label}` | "
            f"{_fmt(row['calibrated']['rmse'])} | {_fmt(row['calibrated']['r2'])} | {_fmt(row['delta_rmse'])} |"
        )
    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            "| Method | RMSE Mean | RMSE Std | R2 Mean | R2 Std |",
            "|---|---:|---:|---:|---:|",
            f"| Original PINN | {_fmt(payload['aggregate']['original']['rmse_mean'])} | {_fmt(payload['aggregate']['original']['rmse_std'])} | {_fmt(payload['aggregate']['original']['r2_mean'])} | {_fmt(payload['aggregate']['original']['r2_std'])} |",
            f"| Calibrated PINN | {_fmt(payload['aggregate']['calibrated']['rmse_mean'])} | {_fmt(payload['aggregate']['calibrated']['rmse_std'])} | {_fmt(payload['aggregate']['calibrated']['r2_mean'])} | {_fmt(payload['aggregate']['calibrated']['r2_std'])} |",
            f"| Original seed ensemble | {_fmt(payload['ensemble']['pinn_pred']['rmse'])} | NA | {_fmt(payload['ensemble']['pinn_pred']['r2'])} | NA |",
            f"| Calibrated seed ensemble | {_fmt(payload['ensemble']['calibrated_pred']['rmse'])} | NA | {_fmt(payload['ensemble']['calibrated_pred']['r2'])} | NA |",
            "",
            "## Interpretation",
            "",
        ]
    )
    delta = payload["aggregate"]["calibrated"]["rmse_mean"] - payload["aggregate"]["original"]["rmse_mean"]
    if delta < 0:
        lines.append(f"- Calibration improved mean RMSE by `{abs(delta):.4f}`.")
    else:
        lines.append(f"- Calibration worsened mean RMSE by `{delta:.4f}`.")
    ens_delta = payload["ensemble"]["calibrated_pred"]["rmse"] - payload["ensemble"]["pinn_pred"]["rmse"]
    if ens_delta < 0:
        lines.append(f"- Calibration also improved seed-ensemble RMSE by `{abs(ens_delta):.4f}`.")
    else:
        lines.append(f"- Calibration did not improve seed-ensemble RMSE; delta=`{ens_delta:.4f}`.")
    lines.extend(
        [
            "- If calibration helps, the original PINN likely has a simple scale/bias error.",
            "- If calibration does not help, the main issue is not a simple output bias; seed variance and branch contribution are more important.",
            "- HF eval/test are very small, so this is a diagnostic result rather than a final generalization claim.",
        ]
    )
    (out_dir / "pinn_calibration_compare.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pinn-run-dir", default=str(DEFAULT_PINN_RUN_DIR))
    parser.add_argument("--hf-data", default=str(DEFAULT_HF_DIR))
    parser.add_argument("--out-dir", default="agent_workspace/runs/yield_pinn_calibration_compare_20260727")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--phi-value", type=float, default=0.4)
    args = parser.parse_args()

    hf_dir = Path(args.hf_data)
    for name in ["train.csv", "eval.csv", "test.csv"]:
        if not (hf_dir / name).exists():
            raise FileNotFoundError(f"Missing HF split file: {hf_dir / name}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_module = import_original_run_module()
    rows: list[dict[str, Any]] = []
    pred_frames: list[pd.DataFrame] = []
    for seed in args.seeds:
        row, preds = evaluate_seed(
            run_module,
            seed=seed,
            pinn_run_dir=Path(args.pinn_run_dir),
            hf_dir=hf_dir,
            phi_value=float(args.phi_value),
        )
        rows.append(row)
        pred_frames.append(preds)
    predictions = pd.concat(pred_frames, ignore_index=True)
    predictions.to_csv(out_dir / "pinn_calibration_predictions.csv", index=False)
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "pinn_run_dir": str(args.pinn_run_dir),
        "hf_dir": str(args.hf_data),
        "seeds": args.seeds,
        "seed_results": rows,
        "aggregate": {
            "original": aggregate(rows, "original"),
            "calibrated": aggregate(rows, "calibrated"),
        },
        "ensemble": ensemble_metrics(predictions),
    }
    (out_dir / "pinn_calibration_compare.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(out_dir, payload)
    print(f"Wrote report to {out_dir / 'pinn_calibration_compare.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
