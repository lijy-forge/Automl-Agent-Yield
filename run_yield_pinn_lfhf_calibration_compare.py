#!/usr/bin/env python3
"""Diagnostic LF/HF baseline layer next to the classmate PINN outputs.

This is not an AutoML executor and it does not edit the classmate source files.
It tests a narrow question:

    Is the low-fidelity table already strong enough that a simple LF-based
    predictor beats the saved classmate PINN predictions on the provided HF
    test split?

Selection uses HF eval only. HF test is report-only.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from run_yield_pinn_residual_compare import (
    DEFAULT_HF_DIR,
    DEFAULT_LF_DIR,
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
    try:
        return f"{float(value):.4f}"
    except Exception:
        return str(value)


def _make_lf_base(kind: str, params: dict[str, Any], seed: int) -> Any:
    if kind == "ridge":
        return Ridge(alpha=float(params["alpha"]))
    if kind == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            random_state=seed,
        )
    raise ValueError(f"Unknown LF base kind: {kind}")


def _lf_base_specs() -> list[dict[str, Any]]:
    return [
        {"kind": "ridge", "alpha": 0.1},
        {"kind": "ridge", "alpha": 1.0},
        {"kind": "ridge", "alpha": 10.0},
        {"kind": "ridge", "alpha": 100.0},
        {"kind": "extra_trees", "n_estimators": 200, "max_depth": 3, "min_samples_leaf": 2},
        {"kind": "extra_trees", "n_estimators": 200, "max_depth": 4, "min_samples_leaf": 2},
    ]


def _fit_prediction_rule(
    *,
    rule: str,
    alpha: float,
    y_train: np.ndarray,
    pinn_train: np.ndarray,
    lf_train: np.ndarray,
) -> dict[str, Any]:
    y_train = np.asarray(y_train, dtype=float)
    pinn_train = np.asarray(pinn_train, dtype=float)
    lf_train = np.asarray(lf_train, dtype=float)
    if rule == "lf_identity":
        return {"rule": rule, "alpha": alpha, "model": None}
    if rule == "lf_bias":
        return {"rule": rule, "alpha": alpha, "bias": float(np.mean(y_train - lf_train))}
    if rule == "lf_affine":
        model = Ridge(alpha=alpha).fit(lf_train.reshape(-1, 1), y_train)
        return {"rule": rule, "alpha": alpha, "model": model}
    if rule == "pinn_lf_stack":
        X = np.column_stack([pinn_train, lf_train, lf_train - pinn_train])
        model = Ridge(alpha=alpha).fit(X, y_train)
        return {"rule": rule, "alpha": alpha, "model": model}
    if rule == "pinn_plus_lfhf_delta":
        X = np.column_stack([lf_train, lf_train - pinn_train])
        model = Ridge(alpha=alpha).fit(X, y_train - pinn_train)
        return {"rule": rule, "alpha": alpha, "model": model}
    raise ValueError(f"Unknown prediction rule: {rule}")


def _apply_prediction_rule(rule_payload: dict[str, Any], pinn_pred: np.ndarray, lf_pred: np.ndarray) -> np.ndarray:
    rule = str(rule_payload["rule"])
    pinn_pred = np.asarray(pinn_pred, dtype=float)
    lf_pred = np.asarray(lf_pred, dtype=float)
    if rule == "lf_identity":
        return lf_pred
    if rule == "lf_bias":
        return lf_pred + float(rule_payload["bias"])
    model = rule_payload["model"]
    if rule == "lf_affine":
        return np.asarray(model.predict(lf_pred.reshape(-1, 1)), dtype=float)
    if rule == "pinn_lf_stack":
        X = np.column_stack([pinn_pred, lf_pred, lf_pred - pinn_pred])
        return np.asarray(model.predict(X), dtype=float)
    if rule == "pinn_plus_lfhf_delta":
        X = np.column_stack([lf_pred, lf_pred - pinn_pred])
        return pinn_pred + np.asarray(model.predict(X), dtype=float)
    raise ValueError(f"Unknown prediction rule: {rule}")


def _candidate_rules() -> list[tuple[str, float]]:
    rules: list[tuple[str, float]] = [
        ("lf_identity", 0.0),
        ("lf_bias", 0.0),
    ]
    for alpha in [0.1, 1.0, 10.0, 100.0]:
        rules.extend(
            [
                ("lf_affine", alpha),
                ("pinn_lf_stack", alpha),
                ("pinn_plus_lfhf_delta", alpha),
            ]
        )
    return rules


def evaluate_seed(
    run_module: Any,
    *,
    seed: int,
    pinn_run_dir: Path,
    lf_dir: Path,
    hf_dir: Path,
    phi_value: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    summary_path = pinn_run_dir / "original_results" / f"seed{seed}_fold1" / "summary.json"
    checkpoint_path = pinn_run_dir / "original_models" / f"seed{seed}_fold1" / "multifidelity.pth"
    if not summary_path.exists() or not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing original PINN result for seed={seed}: {summary_path}, {checkpoint_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    pressure_ref_kpa = float(summary["pressure_ref_kpa"])

    lf_train = load_pinn_frame(run_module, checkpoint_path, lf_dir / "train.csv", pressure_ref_kpa=pressure_ref_kpa, phi_value=phi_value)
    hf_train = load_pinn_frame(run_module, checkpoint_path, hf_dir / "train.csv", pressure_ref_kpa=pressure_ref_kpa, phi_value=phi_value)
    hf_eval = load_pinn_frame(run_module, checkpoint_path, hf_dir / "eval.csv", pressure_ref_kpa=pressure_ref_kpa, phi_value=phi_value)
    hf_test = load_pinn_frame(run_module, checkpoint_path, hf_dir / "test.csv", pressure_ref_kpa=pressure_ref_kpa, phi_value=phi_value)

    eval_rows: list[dict[str, Any]] = []
    for base_spec in _lf_base_specs():
        base = _make_lf_base(str(base_spec["kind"]), base_spec, seed)
        base.fit(lf_train["X"], lf_train["y"])
        lf_pred_train = np.asarray(base.predict(hf_train["X"]), dtype=float)
        lf_pred_eval = np.asarray(base.predict(hf_eval["X"]), dtype=float)
        for rule, alpha in _candidate_rules():
            fitted = _fit_prediction_rule(
                rule=rule,
                alpha=alpha,
                y_train=hf_train["y"],
                pinn_train=hf_train["pinn_pred"],
                lf_train=lf_pred_train,
            )
            pred_eval = _apply_prediction_rule(fitted, hf_eval["pinn_pred"], lf_pred_eval)
            row = {
                "base_spec": base_spec,
                "rule": rule,
                "alpha": alpha,
                "eval_rmse": _rmse(hf_eval["y"], pred_eval),
                "eval_r2": float(r2_score(hf_eval["y"], pred_eval)),
            }
            eval_rows.append(row)

    best_index = int(np.argmin([row["eval_rmse"] for row in eval_rows]))
    best_row = eval_rows[best_index]

    best_base_spec = dict(best_row["base_spec"])
    final_base = _make_lf_base(str(best_base_spec["kind"]), best_base_spec, seed)
    final_base.fit(lf_train["X"], lf_train["y"])
    fit_lf_pred = np.asarray(final_base.predict(np.vstack([hf_train["X"], hf_eval["X"]])), dtype=float)
    test_lf_pred = np.asarray(final_base.predict(hf_test["X"]), dtype=float)
    fit_pinn_pred = np.concatenate([hf_train["pinn_pred"], hf_eval["pinn_pred"]])
    fit_y = np.concatenate([hf_train["y"], hf_eval["y"]])
    final_rule = _fit_prediction_rule(
        rule=str(best_row["rule"]),
        alpha=float(best_row["alpha"]),
        y_train=fit_y,
        pinn_train=fit_pinn_pred,
        lf_train=fit_lf_pred,
    )
    pred_test = _apply_prediction_rule(final_rule, hf_test["pinn_pred"], test_lf_pred)

    pred_frame = pd.DataFrame(
        {
            "seed": seed,
            "row_id": np.arange(len(hf_test["y"])),
            "y_true": hf_test["y"],
            "pinn_pred": hf_test["pinn_pred"],
            "lfhf_calibrated_pred": pred_test,
        }
    )
    row = {
        "seed": seed,
        "n_lf_train": int(len(lf_train["y"])),
        "n_hf_train": int(len(hf_train["y"])),
        "n_hf_eval": int(len(hf_eval["y"])),
        "n_hf_test": int(len(hf_test["y"])),
        "original": _metrics(hf_test["y"], hf_test["pinn_pred"]),
        "selected": best_row,
        "eval_candidates": eval_rows,
        "lfhf_calibrated": _metrics(hf_test["y"], pred_test),
    }
    row["delta_rmse"] = float(row["lfhf_calibrated"]["rmse"] - row["original"]["rmse"])
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
    return {
        "original": _metrics(y, grouped["pinn_pred"].mean().to_numpy(dtype=float)),
        "lfhf_calibrated": _metrics(y, grouped["lfhf_calibrated_pred"].mean().to_numpy(dtype=float)),
    }


def write_report(out_dir: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Classmate PINN Output Plus LF/HF Baseline Diagnostic",
        "",
        f"Generated at: `{payload['generated_at']}`",
        "",
        "## Protocol",
        "",
        f"- Original PINN run dir: `{payload['pinn_run_dir']}`",
        f"- LF data dir: `{payload['lf_dir']}`",
        f"- HF data dir: `{payload['hf_dir']}`",
        f"- Seeds: `{payload['seeds']}`",
        "- The classmate original model is loaded from saved checkpoints and is not edited.",
        "- LF base models are trained only on LF train rows.",
        "- The diagnostic rule is selected on HF eval, then refit on HF train+eval; HF test is report-only.",
        "",
        "## Seed Results",
        "",
        "| Seed | Original RMSE | Original R2 | Selected LF Base | Selected Rule | Diagnostic RMSE | Diagnostic R2 | RMSE Delta |",
        "|---:|---:|---:|---|---|---:|---:|---:|",
    ]
    for row in payload["seed_results"]:
        selected = row["selected"]
        base = selected["base_spec"]
        base_label = base["kind"]
        if base["kind"] == "ridge":
            base_label += f"(alpha={base['alpha']})"
        else:
            base_label += f"(depth={base['max_depth']})"
        rule_label = selected["rule"]
        if float(selected["alpha"]) > 0:
            rule_label += f"(alpha={selected['alpha']})"
        lines.append(
            f"| {row['seed']} | {_fmt(row['original']['rmse'])} | {_fmt(row['original']['r2'])} | "
            f"`{base_label}` | `{rule_label}` | {_fmt(row['lfhf_calibrated']['rmse'])} | "
            f"{_fmt(row['lfhf_calibrated']['r2'])} | {_fmt(row['delta_rmse'])} |"
        )
    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            "| Method | RMSE Mean | RMSE Std | R2 Mean | R2 Std |",
            "|---|---:|---:|---:|---:|",
            f"| Original PINN | {_fmt(payload['aggregate']['original']['rmse_mean'])} | {_fmt(payload['aggregate']['original']['rmse_std'])} | {_fmt(payload['aggregate']['original']['r2_mean'])} | {_fmt(payload['aggregate']['original']['r2_std'])} |",
            f"| PINN output plus LF/HF diagnostic layer | {_fmt(payload['aggregate']['lfhf_calibrated']['rmse_mean'])} | {_fmt(payload['aggregate']['lfhf_calibrated']['rmse_std'])} | {_fmt(payload['aggregate']['lfhf_calibrated']['r2_mean'])} | {_fmt(payload['aggregate']['lfhf_calibrated']['r2_std'])} |",
            f"| Original seed ensemble | {_fmt(payload['ensemble']['original']['rmse'])} | NA | {_fmt(payload['ensemble']['original']['r2'])} | NA |",
            f"| PINN output plus LF/HF diagnostic seed ensemble | {_fmt(payload['ensemble']['lfhf_calibrated']['rmse'])} | NA | {_fmt(payload['ensemble']['lfhf_calibrated']['r2'])} | NA |",
            "",
            "## Interpretation",
            "",
        ]
    )
    delta = payload["aggregate"]["lfhf_calibrated"]["rmse_mean"] - payload["aggregate"]["original"]["rmse_mean"]
    if delta < 0:
        lines.append(f"- The LF/HF diagnostic output is lower in mean RMSE than the original PINN by `{abs(delta):.4f}` on this provided split.")
    else:
        lines.append(f"- The LF/HF diagnostic output is higher in mean RMSE than the original PINN by `{delta:.4f}` on this provided split.")
    ens_delta = payload["ensemble"]["lfhf_calibrated"]["rmse"] - payload["ensemble"]["original"]["rmse"]
    if ens_delta < 0:
        lines.append(f"- The LF/HF diagnostic output is lower in seed-ensemble RMSE by `{abs(ens_delta):.4f}`.")
    else:
        lines.append(f"- The LF/HF diagnostic output is higher in seed-ensemble RMSE by `{ens_delta:.4f}`.")
    lines.extend(
        [
            "- This is a diagnostic layer around saved PINN checkpoints, not a retraining or improvement of the expert model.",
            "- Do not describe this as calibrating or improving the classmate PINN; a separate strict LF-only Ridge baseline already shows that the LF table itself is very strong on this provided HF test.",
            "- HF test has only 5 rows, so RMSE/MAE and seed stability are more important than a single R2.",
        ]
    )
    (out_dir / "pinn_lfhf_calibration_compare.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pinn-run-dir", default=str(DEFAULT_PINN_RUN_DIR))
    parser.add_argument("--lf-data", default=str(DEFAULT_LF_DIR))
    parser.add_argument("--hf-data", default=str(DEFAULT_HF_DIR))
    parser.add_argument("--out-dir", default="agent_workspace/runs/yield_pinn_lfhf_calibration_compare_20260727")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--phi-value", type=float, default=0.4)
    args = parser.parse_args()

    lf_dir = Path(args.lf_data)
    hf_dir = Path(args.hf_data)
    required = [
        lf_dir / "train.csv",
        hf_dir / "train.csv",
        hf_dir / "eval.csv",
        hf_dir / "test.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing split files: " + ", ".join(missing))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_module = import_original_run_module()
    seed_rows: list[dict[str, Any]] = []
    pred_frames: list[pd.DataFrame] = []
    for seed in args.seeds:
        row, pred = evaluate_seed(
            run_module,
            seed=seed,
            pinn_run_dir=Path(args.pinn_run_dir),
            lf_dir=lf_dir,
            hf_dir=hf_dir,
            phi_value=float(args.phi_value),
        )
        seed_rows.append(row)
        pred_frames.append(pred)

    predictions = pd.concat(pred_frames, ignore_index=True)
    predictions.to_csv(out_dir / "pinn_lfhf_calibration_predictions.csv", index=False)
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "pinn_run_dir": str(args.pinn_run_dir),
        "lf_dir": str(lf_dir),
        "hf_dir": str(hf_dir),
        "seeds": args.seeds,
        "seed_results": seed_rows,
        "aggregate": {
            "original": aggregate(seed_rows, "original"),
            "lfhf_calibrated": aggregate(seed_rows, "lfhf_calibrated"),
        },
        "ensemble": ensemble_metrics(predictions),
    }
    (out_dir / "pinn_lfhf_calibration_compare.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(out_dir, payload)
    print(f"Wrote report to {out_dir / 'pinn_lfhf_calibration_compare.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
