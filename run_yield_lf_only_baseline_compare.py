#!/usr/bin/env python3
"""Evaluate a strict LF-only tabular baseline on the provided HF test split.

The baseline is intentionally simple:

    LF train features + LF labels -> StandardScaler + Ridge -> HF test prediction

No HF labels are used for fitting or selection. The goal is to test whether the
low-fidelity table alone is already highly aligned with the provided HF test.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from run_yield_pinn_residual_compare import (
    DEFAULT_HF_DIR,
    DEFAULT_LF_DIR,
    import_original_run_module,
)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.maximum(np.abs(y_true), 1e-8)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": _rmse(y_true, y_pred),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": _mape(y_true, y_pred),
    }


def _fmt(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except Exception:
        return str(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lf-data", default=str(DEFAULT_LF_DIR))
    parser.add_argument("--hf-data", default=str(DEFAULT_HF_DIR))
    parser.add_argument("--out-dir", default="agent_workspace/runs/yield_lf_only_ridge_baseline_20260727")
    parser.add_argument("--phi-value", type=float, default=0.4)
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0])
    args = parser.parse_args()

    lf_dir = Path(args.lf_data)
    hf_dir = Path(args.hf_data)
    lf_train_csv = lf_dir / "train.csv"
    hf_test_csv = hf_dir / "test.csv"
    missing = [str(path) for path in [lf_train_csv, hf_test_csv] if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing split files: " + ", ".join(missing))

    run_module = import_original_run_module()
    lf_raw = pd.read_csv(lf_train_csv)
    pressure_ref_kpa = float(lf_raw["internal_pressure_kpa"].median())
    lf_features, _lf_phi, lf_y = run_module.load_feature_frame(
        lf_train_csv,
        pressure_ref_kpa,
        float(args.phi_value),
    )
    hf_features, _hf_phi, hf_y = run_module.load_feature_frame(
        hf_test_csv,
        pressure_ref_kpa,
        float(args.phi_value),
    )

    rows: list[dict[str, Any]] = []
    pred_frame = pd.DataFrame({"y_true": hf_y})
    for alpha in args.alphas:
        model = make_pipeline(StandardScaler(), Ridge(alpha=float(alpha)))
        model.fit(lf_features, lf_y)
        pred = np.asarray(model.predict(hf_features), dtype=float)
        pred_frame[f"ridge_alpha_{alpha:g}"] = pred
        rows.append(
            {
                "model": "lf_only_standard_scaler_ridge",
                "alpha": float(alpha),
                **_metrics(hf_y, pred),
            }
        )
    best = min(rows, key=lambda row: row["rmse"])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_frame.to_csv(out_dir / "lf_only_ridge_predictions.csv", index=False)
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "protocol": {
            "lf_train": str(lf_train_csv),
            "hf_test": str(hf_test_csv),
            "pressure_ref_kpa": pressure_ref_kpa,
            "pressure_ref_source": "LF train median only",
            "uses_hf_labels_for_training": False,
            "uses_hf_eval_for_selection": False,
            "n_lf_train": int(len(lf_features)),
            "n_hf_test": int(len(hf_features)),
            "feature_dim": int(lf_features.shape[1]),
            "phi_value": float(args.phi_value),
        },
        "ridge_grid": rows,
        "best_by_hf_test_for_diagnostic_only": best,
    }
    (out_dir / "lf_only_ridge_baseline.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# LF-Only Ridge Baseline on Provided HF Test",
        "",
        f"Generated at: `{payload['generated_at']}`",
        "",
        "## Protocol",
        "",
        f"- LF train: `{lf_train_csv}`",
        f"- HF test: `{hf_test_csv}`",
        "- Training labels used: LF train only.",
        "- HF train/eval/test labels are not used for fitting or selection in this diagnostic.",
        f"- Pressure reference: `{pressure_ref_kpa:.3f}` kPa, computed from LF train only.",
        f"- Feature dim: `{lf_features.shape[1]}`",
        f"- phi: `{float(args.phi_value)}`",
        "",
        "## Ridge Grid",
        "",
        "| Model | Alpha | RMSE | MAE | R2 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['model']}` | {_fmt(row['alpha'])} | {_fmt(row['rmse'])} | "
            f"{_fmt(row['mae'])} | {_fmt(row['r2'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- Best diagnostic LF-only Ridge RMSE is `{best['rmse']:.4f}` on the provided HF test.",
            "- This should not be described as calibrating or improving the classmate PINN; it is a simple LF-only tabular baseline.",
            "- A low RMSE here means the LF table and this provided HF test are highly aligned, so apparent stability can be a data/split artifact rather than proof of a stronger physical model.",
        ]
    )
    (out_dir / "lf_only_ridge_baseline.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote report to {out_dir / 'lf_only_ridge_baseline.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
