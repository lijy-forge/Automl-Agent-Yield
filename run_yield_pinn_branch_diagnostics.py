#!/usr/bin/env python3
"""Post-hoc branch diagnostics for the classmate PINN baseline.

This script does not retrain or edit the classmate's original model. It loads
the provided-split PINN checkpoints and evaluates what happens when one branch
input group is replaced by the scaler mean, which is zero in standardized
feature space:

    original prediction
    mask thermal branch inputs
    mask composition branch inputs
    mask history branch inputs

The goal is diagnostic, not causal proof. HF test has only five rows, so the
report should be read as a small-sample signal.
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from run_yield_classmate_pinn_original_compare import (
    ORIGINAL_ROOT,
    ORIGINAL_RUN_SCRIPT,
    ensure_original_files,
)


DEFAULT_PINN_RUN_DIR = Path("agent_workspace/runs/yield_classmate_original_pinn_provided_split_20260721")
DEFAULT_HF_DIR = Path("/Users/lijiayao/Downloads/yield_predict_180lf_20hf_real_lf(1)/high_fidelity")

BRANCHES = {
    "thermal": "温压分支",
    "composition": "组成分支",
    "history": "工艺历史分支",
}


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
    spec = importlib.util.spec_from_file_location("classmate_original_run_for_branch_diag", ORIGINAL_RUN_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import original run script: {ORIGINAL_RUN_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    root_text = str(ORIGINAL_ROOT.resolve())
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    spec.loader.exec_module(module)
    return module


def load_model_and_frame(
    run_module: Any,
    checkpoint_path: Path,
    csv_path: Path,
    *,
    pressure_ref_kpa: float,
    phi_value: float,
) -> dict[str, Any]:
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
    return {
        "model": model,
        "X": x_t,
        "phi": phi_t,
        "y": y_t.numpy().astype(float),
        "feature_names": list(run_module.ENGINEERED_COLUMNS),
    }


def predict_with_mask(model: Any, x: Any, phi: Any, indices: list[int] | None = None) -> dict[str, np.ndarray]:
    import torch

    x_use = x.clone()
    if indices:
        # FeatureScaler standardizes train mean to 0, so this masks a branch to
        # its training/reference mean without creating out-of-range values.
        x_use[:, indices] = 0.0
    with torch.no_grad():
        pred, aux = model(x_use, phi)
    gates = aux["branch_gates"].detach().cpu().numpy().astype(float)
    return {
        "pred": pred.detach().cpu().numpy().astype(float),
        "m1_eff": aux["m1_pred"].detach().cpu().numpy().astype(float),
        "gates": gates,
        "phi0": np.asarray([float(aux["phi0"].detach().cpu().item())]),
        "phi_max": np.asarray([float(aux["phi_max"].detach().cpu().item())]),
    }


def branch_indices(run_module: Any) -> dict[str, list[int]]:
    model_cls = run_module.YieldPredictPINN
    return {
        "thermal": list(model_cls.THERMAL_IDX),
        "composition": list(model_cls.COMPOSITION_IDX),
        "history": list(model_cls.HISTORY_IDX),
    }


def summarize_gates(gates: np.ndarray) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for idx, name in enumerate(["thermal", "composition", "history"]):
        out[name] = {
            "mean": float(np.mean(gates[:, idx])),
            "std": float(np.std(gates[:, idx])),
            "min": float(np.min(gates[:, idx])),
            "max": float(np.max(gates[:, idx])),
        }
    return out


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
        raise FileNotFoundError(
            f"Missing original PINN provided-split result for seed={seed}. "
            f"Expected {summary_path} and {checkpoint_path}."
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    pressure_ref_kpa = float(summary["pressure_ref_kpa"])
    frame = load_model_and_frame(
        run_module,
        checkpoint_path,
        hf_dir / "test.csv",
        pressure_ref_kpa=pressure_ref_kpa,
        phi_value=phi_value,
    )
    model = frame["model"]
    x = frame["X"]
    phi = frame["phi"]
    y = frame["y"]
    idxs = branch_indices(run_module)

    original = predict_with_mask(model, x, phi)
    original_metrics = _metrics(y, original["pred"])
    masked: dict[str, Any] = {}
    pred_frame = pd.DataFrame({"seed": seed, "row_id": np.arange(len(y)), "y_true": y, "original_pred": original["pred"]})
    for branch, indices in idxs.items():
        out = predict_with_mask(model, x, phi, indices)
        metrics = _metrics(y, out["pred"])
        masked[branch] = {
            "metrics": metrics,
            "delta_rmse": float(metrics["rmse"] - original_metrics["rmse"]),
            "delta_r2": float(metrics["r2"] - original_metrics["r2"]),
        }
        pred_frame[f"mask_{branch}_pred"] = out["pred"]

    gate_summary = summarize_gates(original["gates"])
    row = {
        "seed": seed,
        "pressure_ref_kpa": pressure_ref_kpa,
        "n_hf_test": int(len(y)),
        "original": {
            "metrics": original_metrics,
            "gate_summary": gate_summary,
            "phi0": float(original["phi0"][0]),
            "phi_max": float(original["phi_max"][0]),
            "m1_eff_mean": float(np.mean(original["m1_eff"])),
            "m1_eff_std": float(np.std(original["m1_eff"])),
        },
        "masked": masked,
    }
    return row, pred_frame


def aggregate_mask(rows: list[dict[str, Any]], branch: str) -> dict[str, float]:
    deltas = [float(row["masked"][branch]["delta_rmse"]) for row in rows]
    r2_deltas = [float(row["masked"][branch]["delta_r2"]) for row in rows]
    masked_rmse = [float(row["masked"][branch]["metrics"]["rmse"]) for row in rows]
    return {
        "delta_rmse_mean": float(np.mean(deltas)),
        "delta_rmse_std": float(np.std(deltas)),
        "delta_r2_mean": float(np.mean(r2_deltas)),
        "masked_rmse_mean": float(np.mean(masked_rmse)),
    }


def aggregate_original(rows: list[dict[str, Any]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for metric in ["rmse", "mae", "r2", "mape"]:
        vals = [float(row["original"]["metrics"][metric]) for row in rows]
        out[f"{metric}_mean"] = float(np.mean(vals))
        out[f"{metric}_std"] = float(np.std(vals))
    return out


def ensemble_metrics(preds: pd.DataFrame) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    by_row = preds.groupby("row_id")
    y_true = by_row["y_true"].first().to_numpy(dtype=float)
    for col in ["original_pred", "mask_thermal_pred", "mask_composition_pred", "mask_history_pred"]:
        mean_pred = by_row[col].mean().to_numpy(dtype=float)
        rows[col] = _metrics(y_true, mean_pred)
    rows["mask_delta_rmse"] = {
        "thermal": float(rows["mask_thermal_pred"]["rmse"] - rows["original_pred"]["rmse"]),
        "composition": float(rows["mask_composition_pred"]["rmse"] - rows["original_pred"]["rmse"]),
        "history": float(rows["mask_history_pred"]["rmse"] - rows["original_pred"]["rmse"]),
    }
    return rows


def feature_mapping(run_module: Any) -> dict[str, list[str]]:
    names = list(run_module.ENGINEERED_COLUMNS)
    return {branch: [names[idx] for idx in indices] for branch, indices in branch_indices(run_module).items()}


def write_report(out_dir: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Classmate PINN Branch Diagnostics",
        "",
        f"Generated at: `{payload['generated_at']}`",
        "",
        "## Protocol",
        "",
        f"- Original PINN run dir: `{payload['pinn_run_dir']}`",
        f"- HF data dir: `{payload['hf_dir']}`",
        f"- Seeds: `{payload['seeds']}`",
        "- No retraining is performed; this is post-hoc diagnostics on saved original PINN checkpoints.",
        "- Masking means replacing one standardized branch input group with 0, i.e. the scaler reference mean.",
        "- HF test has only 5 rows, so this is a diagnostic signal rather than a final causal claim.",
        "",
        "## Seed Results",
        "",
        "| Seed | Original RMSE | Original R2 | Mask Thermal ΔRMSE | Mask Composition ΔRMSE | Mask History ΔRMSE | Gate thermal | Gate composition | Gate history |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["seed_results"]:
        gates = row["original"]["gate_summary"]
        lines.append(
            f"| {row['seed']} | {_fmt(row['original']['metrics']['rmse'])} | {_fmt(row['original']['metrics']['r2'])} | "
            f"{_fmt(row['masked']['thermal']['delta_rmse'])} | {_fmt(row['masked']['composition']['delta_rmse'])} | "
            f"{_fmt(row['masked']['history']['delta_rmse'])} | {_fmt(gates['thermal']['mean'])} | "
            f"{_fmt(gates['composition']['mean'])} | {_fmt(gates['history']['mean'])} |"
        )
    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            "| Quantity | Value |",
            "|---|---:|",
            f"| Original RMSE mean | {_fmt(payload['aggregate']['original']['rmse_mean'])} |",
            f"| Original R2 mean | {_fmt(payload['aggregate']['original']['r2_mean'])} |",
            f"| Mask thermal ΔRMSE mean | {_fmt(payload['aggregate']['masked']['thermal']['delta_rmse_mean'])} |",
            f"| Mask composition ΔRMSE mean | {_fmt(payload['aggregate']['masked']['composition']['delta_rmse_mean'])} |",
            f"| Mask history ΔRMSE mean | {_fmt(payload['aggregate']['masked']['history']['delta_rmse_mean'])} |",
            "",
            "Positive ΔRMSE means masking that branch made predictions worse, so the branch is useful under this diagnostic. Negative ΔRMSE means masking improved the metric, which can indicate redundancy, noise, or small-test-set instability.",
            "",
            "## Seed Ensemble",
            "",
            "| Prediction | RMSE | R2 |",
            "|---|---:|---:|",
            f"| Original seed-ensemble | {_fmt(payload['ensemble']['original_pred']['rmse'])} | {_fmt(payload['ensemble']['original_pred']['r2'])} |",
            f"| Mask thermal seed-ensemble | {_fmt(payload['ensemble']['mask_thermal_pred']['rmse'])} | {_fmt(payload['ensemble']['mask_thermal_pred']['r2'])} |",
            f"| Mask composition seed-ensemble | {_fmt(payload['ensemble']['mask_composition_pred']['rmse'])} | {_fmt(payload['ensemble']['mask_composition_pred']['r2'])} |",
            f"| Mask history seed-ensemble | {_fmt(payload['ensemble']['mask_history_pred']['rmse'])} | {_fmt(payload['ensemble']['mask_history_pred']['r2'])} |",
            "",
            "## Interpretation",
            "",
        ]
    )

    deltas = payload["aggregate"]["masked"]
    most_positive = max(deltas, key=lambda key: deltas[key]["delta_rmse_mean"])
    most_negative = min(deltas, key=lambda key: deltas[key]["delta_rmse_mean"])
    lines.extend(
        [
            f"- The branch with the largest positive mean ΔRMSE is `{most_positive}` ({BRANCHES[most_positive]}), so it looks most important by masking.",
            f"- The branch with the smallest mean ΔRMSE is `{most_negative}` ({BRANCHES[most_negative]}), so it looks least supported or most redundant by this diagnostic.",
            "- Gate weights are not the same as causal importance; they show how the learned fusion head allocates attention-like weights for each sample.",
            "- If gate weights and masking importance disagree, that is itself useful: it means the gate may not be a direct explanation of predictive contribution.",
        ]
    )
    lines.extend(["", "## Branch Feature Groups", ""])
    for branch, features in payload["feature_mapping"].items():
        lines.append(f"- `{branch}` ({BRANCHES[branch]}): " + ", ".join(f"`{name}`" for name in features))
    (out_dir / "pinn_branch_diagnostics.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pinn-run-dir", default=str(DEFAULT_PINN_RUN_DIR))
    parser.add_argument("--hf-data", default=str(DEFAULT_HF_DIR))
    parser.add_argument("--out-dir", default="agent_workspace/runs/yield_pinn_branch_diagnostics_20260727")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--phi-value", type=float, default=0.4)
    args = parser.parse_args()

    hf_dir = Path(args.hf_data)
    if not (hf_dir / "test.csv").exists():
        raise FileNotFoundError(f"Missing HF test CSV: {hf_dir / 'test.csv'}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_module = import_original_run_module()
    seed_rows: list[dict[str, Any]] = []
    pred_frames: list[pd.DataFrame] = []
    for seed in args.seeds:
        row, preds = evaluate_seed(
            run_module,
            seed=seed,
            pinn_run_dir=Path(args.pinn_run_dir),
            hf_dir=hf_dir,
            phi_value=float(args.phi_value),
        )
        seed_rows.append(row)
        pred_frames.append(preds)
    predictions = pd.concat(pred_frames, ignore_index=True)
    predictions.to_csv(out_dir / "pinn_branch_predictions.csv", index=False)

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "pinn_run_dir": str(args.pinn_run_dir),
        "hf_dir": str(args.hf_data),
        "seeds": args.seeds,
        "feature_mapping": feature_mapping(run_module),
        "seed_results": seed_rows,
        "aggregate": {
            "original": aggregate_original(seed_rows),
            "masked": {branch: aggregate_mask(seed_rows, branch) for branch in BRANCHES},
        },
        "ensemble": ensemble_metrics(predictions),
    }
    (out_dir / "pinn_branch_diagnostics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(out_dir, payload)
    print(f"Wrote report to {out_dir / 'pinn_branch_diagnostics.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
