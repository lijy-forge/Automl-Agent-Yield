#!/usr/bin/env python3
"""Seed-ensemble stability analysis for the classmate PINN baseline."""

from __future__ import annotations

import argparse
import itertools
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from run_yield_pinn_residual_compare import (
    DEFAULT_HF_DIR,
    import_original_run_module,
    load_pinn_frame,
)


DEFAULT_PINN_RUN_DIR = Path("agent_workspace/runs/yield_classmate_original_pinn_provided_split_seed0_9_20260727")


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.maximum(np.abs(y_true), 1e-8)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    return {
        "rmse": _rmse(y, pred),
        "mae": float(mean_absolute_error(y, pred)),
        "r2": float(r2_score(y, pred)),
        "mape": _mape(y, pred),
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):.4f}"
    except Exception:
        return str(value)


def load_seed_prediction(
    run_module: Any,
    *,
    seed: int,
    pinn_run_dir: Path,
    hf_dir: Path,
    phi_value: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    summary_path = pinn_run_dir / "original_results" / f"seed{seed}_fold1" / "summary.json"
    checkpoint_path = pinn_run_dir / "original_models" / f"seed{seed}_fold1" / "multifidelity.pth"
    if not summary_path.exists() or not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing seed={seed} result: {summary_path}, {checkpoint_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    pressure_ref_kpa = float(summary["pressure_ref_kpa"])
    test = load_pinn_frame(
        run_module,
        checkpoint_path,
        hf_dir / "test.csv",
        pressure_ref_kpa=pressure_ref_kpa,
        phi_value=phi_value,
    )
    metrics = _metrics(test["y"], test["pinn_pred"])
    row = pd.DataFrame(
        {
            "seed": seed,
            "row_id": np.arange(len(test["y"])),
            "y_true": test["y"],
            "pinn_pred": test["pinn_pred"],
        }
    )
    meta = {
        "seed": seed,
        "pressure_ref_kpa": pressure_ref_kpa,
        "metrics": metrics,
        "phi0": float(test["phi0"][0]),
        "phi_max": float(test["phi_max"][0]),
        "m1_eff_mean": float(np.mean(test["m1_eff"])),
        "m1_eff_std": float(np.std(test["m1_eff"])),
    }
    return row, meta


def ensemble_for_seeds(predictions: pd.DataFrame, seeds: tuple[int, ...]) -> dict[str, Any]:
    sub = predictions[predictions["seed"].isin(seeds)]
    grouped = sub.groupby("row_id")
    y = grouped["y_true"].first().to_numpy(dtype=float)
    pred = grouped["pinn_pred"].mean().to_numpy(dtype=float)
    return {"seeds": list(seeds), "k": len(seeds), **_metrics(y, pred)}


def summarize_combinations(combo_rows: list[dict[str, Any]]) -> dict[str, Any]:
    rmses = np.asarray([row["rmse"] for row in combo_rows], dtype=float)
    r2s = np.asarray([row["r2"] for row in combo_rows], dtype=float)
    best = min(combo_rows, key=lambda row: row["rmse"])
    worst = max(combo_rows, key=lambda row: row["rmse"])
    return {
        "n_combinations": int(len(combo_rows)),
        "rmse_mean": float(np.mean(rmses)),
        "rmse_std": float(np.std(rmses)),
        "rmse_min": float(np.min(rmses)),
        "rmse_max": float(np.max(rmses)),
        "r2_mean": float(np.mean(r2s)),
        "r2_std": float(np.std(r2s)),
        "r2_min": float(np.min(r2s)),
        "r2_max": float(np.max(r2s)),
        "best": best,
        "worst": worst,
    }


def write_report(out_dir: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Classmate PINN Seed-Ensemble Stability",
        "",
        f"Generated at: `{payload['generated_at']}`",
        "",
        "## Protocol",
        "",
        f"- Original PINN run dir: `{payload['pinn_run_dir']}`",
        f"- HF data dir: `{payload['hf_dir']}`",
        f"- Seeds: `{payload['seeds']}`",
        "- Each checkpoint is the classmate original PINN trained on the same provided LF/HF split.",
        "- Ensemble prediction is the row-wise average of selected seed predictions.",
        "- All seed combinations are enumerated for each ensemble size k.",
        "- HF test has only 5 rows, so this is a stability diagnostic.",
        "",
        "## Single Seed Results",
        "",
        "| Seed | RMSE | R2 | MAE | phi0 | phi_max | m1_eff mean |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["seed_results"]:
        m = row["metrics"]
        lines.append(
            f"| {row['seed']} | {_fmt(m['rmse'])} | {_fmt(m['r2'])} | {_fmt(m['mae'])} | "
            f"{_fmt(row['phi0'])} | {_fmt(row['phi_max'])} | {_fmt(row['m1_eff_mean'])} |"
        )
    lines.extend(
        [
            "",
            "## Ensemble Size Summary",
            "",
            "| k seeds | # combos | RMSE mean | RMSE std | RMSE min | RMSE max | R2 mean | R2 std |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for k in sorted(payload["ensemble_summary"], key=lambda x: int(x)):
        row = payload["ensemble_summary"][k]
        lines.append(
            f"| {k} | {row['n_combinations']} | {_fmt(row['rmse_mean'])} | {_fmt(row['rmse_std'])} | "
            f"{_fmt(row['rmse_min'])} | {_fmt(row['rmse_max'])} | {_fmt(row['r2_mean'])} | {_fmt(row['r2_std'])} |"
        )
    lines.extend(
        [
            "",
            "## Prefix Ensemble",
            "",
            "| Seeds used | RMSE | R2 | MAE |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in payload["prefix_ensemble"]:
        lines.append(
            f"| `{row['seeds']}` | {_fmt(row['rmse'])} | {_fmt(row['r2'])} | {_fmt(row['mae'])} |"
        )
    all_seed = payload["all_seed_ensemble"]
    single = payload["ensemble_summary"]["1"]
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- Single-seed RMSE mean is `{single['rmse_mean']:.4f}` with std `{single['rmse_std']:.4f}`, so the original PINN is seed-sensitive on this split.",
            f"- All 10 seeds ensembled together reach RMSE `{all_seed['rmse']:.4f}` and R2 `{all_seed['r2']:.4f}`.",
            "- If larger k reduces RMSE std and narrows RMSE max-min, seed ensembling is acting as a stabilizer.",
            "- This does not prove external generalization because the HF test has only 5 rows; it is evidence about training variance under the provided split.",
        ]
    )
    (out_dir / "pinn_seed_ensemble_stability.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pinn-run-dir", default=str(DEFAULT_PINN_RUN_DIR))
    parser.add_argument("--hf-data", default=str(DEFAULT_HF_DIR))
    parser.add_argument("--out-dir", default="agent_workspace/runs/yield_pinn_seed_ensemble_stability_20260727")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--phi-value", type=float, default=0.4)
    args = parser.parse_args()

    hf_dir = Path(args.hf_data)
    if not (hf_dir / "test.csv").exists():
        raise FileNotFoundError(f"Missing HF test CSV: {hf_dir / 'test.csv'}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_module = import_original_run_module()

    pred_frames: list[pd.DataFrame] = []
    seed_results: list[dict[str, Any]] = []
    for seed in args.seeds:
        pred, meta = load_seed_prediction(
            run_module,
            seed=seed,
            pinn_run_dir=Path(args.pinn_run_dir),
            hf_dir=hf_dir,
            phi_value=float(args.phi_value),
        )
        pred_frames.append(pred)
        seed_results.append(meta)
    predictions = pd.concat(pred_frames, ignore_index=True)
    predictions.to_csv(out_dir / "pinn_seed_predictions.csv", index=False)

    combo_details: dict[str, list[dict[str, Any]]] = {}
    ensemble_summary: dict[str, dict[str, Any]] = {}
    seeds_tuple = tuple(args.seeds)
    for k in range(1, len(seeds_tuple) + 1):
        rows = [ensemble_for_seeds(predictions, combo) for combo in itertools.combinations(seeds_tuple, k)]
        combo_details[str(k)] = rows
        ensemble_summary[str(k)] = summarize_combinations(rows)

    prefix_rows = [
        ensemble_for_seeds(predictions, tuple(args.seeds[:k]))
        for k in range(1, len(args.seeds) + 1)
    ]
    all_seed = ensemble_for_seeds(predictions, tuple(args.seeds))
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "pinn_run_dir": str(args.pinn_run_dir),
        "hf_dir": str(args.hf_data),
        "seeds": args.seeds,
        "seed_results": seed_results,
        "ensemble_summary": ensemble_summary,
        "prefix_ensemble": prefix_rows,
        "all_seed_ensemble": all_seed,
        "combo_details": combo_details,
    }
    (out_dir / "pinn_seed_ensemble_stability.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(out_dir, payload)
    print(f"Wrote report to {out_dir / 'pinn_seed_ensemble_stability.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
