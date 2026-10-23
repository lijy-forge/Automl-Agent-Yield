#!/usr/bin/env python3
"""Run the classmate's original PINN pipeline on the current yield dataset.

This script keeps the original training code path intact and only adds a schema
adapter/splitter around it:
- target `yield_stress` -> `tau_y_final_pa`
- current stage columns -> the original script's sequential stage schema
- one CSV -> fold-local LF/HF train/eval/test directories

The current dataset has no distinct low-fidelity table, so LF pretraining is
fed with train-fold data only as a mechanical directory adapter. No new data or
synthetic low-fidelity labels are generated here. The outer test fold is never
included in LF/HF training or early stopping.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split


DEFAULT_DATA_PATH = "agent_workspace/data/generated_yield_stress/generated_yield_stress_data_engineered.csv"
DOWNLOAD_RUN_SCRIPT = Path("/Users/lijiayao/Downloads/run_yield_predict_experiment(1).py")
DOWNLOAD_MODEL_SCRIPT = Path("/Users/lijiayao/Downloads/pinn_yield_predict(1).py")
ORIGINAL_ROOT = Path("agent_workspace/external_baselines/classmate_yield_pinn_original")
ORIGINAL_RUN_SCRIPT = ORIGINAL_ROOT / "multi_fidelity" / "src" / "experiments" / "run_yield_predict_experiment.py"
ORIGINAL_MODEL_SCRIPT = ORIGINAL_ROOT / "multi_fidelity" / "src" / "model" / "pinn_yield_predict.py"

FORWARD_STAGE_SOURCE = [1, 2, 3, 4, 6, 7, 9, 10, 11]
REVERSE_STAGE_SOURCE = [4, 10]

AUTO_RUNS = {
    0: "agent_workspace/runs/yield_multiseed_normal_seed0_20260720",
    1: "agent_workspace/runs/yield_multiseed_normal_seed1_20260720",
    2: "agent_workspace/runs/yield_multiseed_normal_seed2_20260720",
}


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.maximum(np.abs(y_true), 1e-8)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0.0, None)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": _mape(y_true, y_pred),
    }


def ensure_original_files() -> None:
    if not DOWNLOAD_RUN_SCRIPT.exists():
        raise FileNotFoundError(f"Missing original run script: {DOWNLOAD_RUN_SCRIPT}")
    if not DOWNLOAD_MODEL_SCRIPT.exists():
        raise FileNotFoundError(f"Missing original model script: {DOWNLOAD_MODEL_SCRIPT}")
    ORIGINAL_RUN_SCRIPT.parent.mkdir(parents=True, exist_ok=True)
    ORIGINAL_MODEL_SCRIPT.parent.mkdir(parents=True, exist_ok=True)
    for dst in (ORIGINAL_RUN_SCRIPT, ORIGINAL_MODEL_SCRIPT):
        if dst.exists():
            dst.chmod(0o644)
    shutil.copy2(DOWNLOAD_RUN_SCRIPT, ORIGINAL_RUN_SCRIPT)
    shutil.copy2(DOWNLOAD_MODEL_SCRIPT, ORIGINAL_MODEL_SCRIPT)
    ORIGINAL_RUN_SCRIPT.chmod(0o644)
    ORIGINAL_MODEL_SCRIPT.chmod(0o644)
    for path in [
        ORIGINAL_ROOT / "multi_fidelity" / "__init__.py",
        ORIGINAL_ROOT / "multi_fidelity" / "src" / "__init__.py",
        ORIGINAL_ROOT / "multi_fidelity" / "src" / "model" / "__init__.py",
        ORIGINAL_ROOT / "multi_fidelity" / "src" / "experiments" / "__init__.py",
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=True)


def adapt_to_original_schema(df: pd.DataFrame) -> pd.DataFrame:
    required = [
        "slurry_temp_c",
        "water_tank_temp_c",
        "jacket_temp_c",
        "internal_pressure_kpa",
        "coarse_ap_mass_kg",
        "fine_ap_mass_kg",
        "rdx_mass_kg",
        "yield_stress",
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Current dataset is missing required columns for original PINN adapter: {missing}")

    out = pd.DataFrame(index=df.index)
    for col in required[:-1]:
        out[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    for target_idx, source_stage in enumerate(FORWARD_STAGE_SOURCE, start=1):
        out[f"forward_rpm_{target_idx}"] = pd.to_numeric(
            df.get(f"forward_rpm_{source_stage}", 0.0), errors="coerce"
        ).fillna(0.0)
        out[f"forward_time_{target_idx}_min"] = pd.to_numeric(
            df.get(f"forward_time_{source_stage}_min", 0.0), errors="coerce"
        ).fillna(0.0)

    for target_idx, source_stage in enumerate(REVERSE_STAGE_SOURCE, start=1):
        out[f"reverse_rpm_{target_idx}"] = pd.to_numeric(
            df.get(f"reverse_rpm_{source_stage}", 0.0), errors="coerce"
        ).fillna(0.0)
        out[f"reverse_time_{target_idx}_min"] = pd.to_numeric(
            df.get(f"reverse_time_{source_stage}_min", 0.0), errors="coerce"
        ).fillna(0.0)

    out["tau_y_final_pa"] = pd.to_numeric(df["yield_stress"], errors="coerce")
    if "sample_id" in df.columns:
        out["sample_id"] = df["sample_id"].astype(str)
    return out.dropna(subset=["tau_y_final_pa"]).reset_index(drop=True)


def prepare_fold_data(df: pd.DataFrame, *, seed: int, fold: int, train_idx: np.ndarray, test_idx: np.ndarray, out_dir: Path) -> dict[str, Any]:
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    hf_train_df, hf_eval_df = train_test_split(train_df, test_size=0.20, random_state=seed * 100 + fold)
    hf_train_df = hf_train_df.reset_index(drop=True)
    hf_eval_df = hf_eval_df.reset_index(drop=True)

    # No synthetic LF is created here. This is only a mechanical directory
    # adapter so the original script can run without changing its training code.
    lf_train_df = hf_train_df.copy()
    lf_test_df = hf_eval_df.copy()

    lf_dir = out_dir / "low_fidelity"
    hf_dir = out_dir / "high_fidelity"
    lf_dir.mkdir(parents=True, exist_ok=True)
    hf_dir.mkdir(parents=True, exist_ok=True)

    lf_train_df.to_csv(lf_dir / "train.csv", index=False)
    lf_test_df.to_csv(lf_dir / "test.csv", index=False)
    hf_train_df.to_csv(hf_dir / "train.csv", index=False)
    hf_eval_df.to_csv(hf_dir / "eval.csv", index=False)
    test_df.to_csv(hf_dir / "test.csv", index=False)

    return {
        "lf_dir": lf_dir,
        "hf_dir": hf_dir,
        "n_lf_train": len(lf_train_df),
        "n_lf_test": len(lf_test_df),
        "n_hf_train": len(hf_train_df),
        "n_hf_eval": len(hf_eval_df),
        "n_hf_test": len(test_df),
    }


def import_original_run_module():
    spec = importlib.util.spec_from_file_location("classmate_original_run", ORIGINAL_RUN_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import original run script: {ORIGINAL_RUN_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(ORIGINAL_ROOT.resolve()))
    spec.loader.exec_module(module)
    return module


def compute_original_predictions(
    run_module: Any,
    checkpoint_path: Path,
    hf_test_csv: Path,
    phi_value: float,
    *,
    pressure_ref_kpa: float,
) -> tuple[np.ndarray, np.ndarray]:
    import torch

    payload = torch.load(checkpoint_path, map_location="cpu")
    pressure_ref = float(pressure_ref_kpa)
    features_df, phi, y = run_module.load_feature_frame(hf_test_csv, pressure_ref, phi_value)
    scaler_payload = payload["scaler"]
    scaler = run_module.FeatureScaler(
        mean_=np.asarray(scaler_payload["mean"], dtype=np.float32),
        std_=np.asarray(scaler_payload["std"], dtype=np.float32),
    )
    x, phi_t, y_t = run_module.to_tensors(features_df, phi, y, scaler)
    model = run_module.YieldPredictPINN(input_dim=len(run_module.ENGINEERED_COLUMNS), hidden_dim=int(payload.get("hidden_dim", 16)))
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    with torch.no_grad():
        pred, _ = model(x, phi_t)
    return y_t.numpy().astype(float), pred.numpy().astype(float)


def run_original_fold(
    *,
    seed: int,
    fold: int,
    fold_data: dict[str, Any],
    out_dir: Path,
    phi_value: float,
    hidden_dim: int,
    lf_max_epochs: int,
    hf_max_epochs: int,
    lf_patience: int,
    hf_patience: int,
) -> dict[str, Any]:
    result_dir = out_dir / "original_results" / f"seed{seed}_fold{fold}"
    models_dir = out_dir / "original_models" / f"seed{seed}_fold{fold}"
    cmd = [
        sys.executable,
        str(ORIGINAL_RUN_SCRIPT.resolve()),
        "--lf-data",
        str(Path(fold_data["lf_dir"]).resolve()),
        "--hf-data",
        str(Path(fold_data["hf_dir"]).resolve()),
        "--out-dir",
        str(result_dir.resolve()),
        "--models-dir",
        str(models_dir.resolve()),
        "--phi-value",
        str(phi_value),
        "--hidden-dim",
        str(hidden_dim),
        "--seed",
        str(seed),
        "--lf-max-epochs",
        str(lf_max_epochs),
        "--hf-max-epochs",
        str(hf_max_epochs),
        "--lf-patience",
        str(lf_patience),
        "--hf-patience",
        str(hf_patience),
    ]
    proc = subprocess.run(cmd, cwd=str(Path.cwd()), text=True, capture_output=True)
    summary_path = result_dir / "summary.json"
    row: dict[str, Any] = {
        "seed": seed,
        "fold": fold,
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
        **{k: v for k, v in fold_data.items() if k.startswith("n_")},
    }
    if proc.returncode != 0:
        row["status"] = "failed"
        return row
    row["status"] = "ok"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    checkpoint = models_dir / "multifidelity.pth"
    run_module = import_original_run_module()
    y_true, y_pred = compute_original_predictions(
        run_module,
        checkpoint,
        Path(fold_data["hf_dir"]) / "test.csv",
        phi_value,
        pressure_ref_kpa=float(summary.get("pressure_ref_kpa")),
    )
    row["summary_metrics"] = summary.get("metrics", {})
    row["metrics"] = _metrics(y_true, y_pred)
    row["y_true"] = y_true.tolist()
    row["y_pred"] = y_pred.tolist()
    return row


def load_automl_seed(seed: int) -> dict[str, Any] | None:
    run_dir = AUTO_RUNS.get(seed)
    if not run_dir:
        return None
    path = Path(run_dir) / "metrics" / "free_search_report.json"
    if not path.exists():
        return None
    report = json.loads(path.read_text(encoding="utf-8"))
    champion = report.get("champion") or {}
    metrics = champion.get("oof_metrics") or {}
    return {
        "seed": seed,
        "run_dir": run_dir,
        "champion": champion.get("name"),
        "rmse": metrics.get("rmse"),
        "mae": metrics.get("mae"),
        "r2": metrics.get("r2"),
        "mape": metrics.get("mape"),
    }


def write_markdown(out_dir: Path, payload: dict[str, Any]) -> None:
    provided_split = bool(payload["protocol"].get("provided_split"))
    lines = [
        "# AutoML vs Classmate Original PINN Pipeline",
        "",
        f"Generated at: `{payload['generated_at']}`",
        "",
        "## Protocol",
        "",
        f"- Data: `{payload['data_path']}`",
        f"- Seeds: `{payload['protocol']['seeds']}`",
        "- Original PINN script is copied from Downloads and run as a subprocess.",
        "- Only schema/split adapter is added; the original model/training functions are not rewritten.",
    ]
    if provided_split:
        lines.extend(
            [
                "- Provided LF/HF split is used directly; no KFold split adapter is applied.",
                f"- LF data: `{payload['protocol'].get('lf_data')}`",
                f"- HF data: `{payload['protocol'].get('hf_data')}`",
                f"- `small_sample_mode` expected from HF train size: `{payload['protocol'].get('small_sample_mode_expected')}`",
            ]
        )
    else:
        lines.extend(
            [
                f"- KFold splits: `{payload['protocol']['n_splits']}`",
                "- Because no separate LF CSV is present in this repo, LF pretraining is fed with outer-train-fold data only as a mechanical adapter. No new synthetic LF data are generated.",
            ]
        )
    lines.extend(
        [
            "",
            "## Seed Results",
            "",
            "| Seed | AutoML RMSE | Original PINN RMSE | Delta PINN-AutoML | AutoML R2 | Original PINN R2 |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["seed_results"]:
        auto = row.get("automl") or {}
        pinn = row.get("original_pinn") or {}
        delta = None
        if auto.get("rmse") is not None and pinn.get("rmse") is not None:
            delta = float(pinn["rmse"]) - float(auto["rmse"])
        lines.append(
            f"| {row['seed']} | {_fmt(auto.get('rmse'))} | {_fmt(pinn.get('rmse'))} | {_fmt(delta)} | "
            f"{_fmt(auto.get('r2'))} | {_fmt(pinn.get('r2'))} |"
        )
    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            "| Method | RMSE Mean | RMSE Std | MAE Mean | R2 Mean |",
            "|---|---:|---:|---:|---:|",
            f"| AutoML champion | {_fmt(payload['aggregate']['automl']['rmse_mean'])} | {_fmt(payload['aggregate']['automl']['rmse_std'])} | {_fmt(payload['aggregate']['automl']['mae_mean'])} | {_fmt(payload['aggregate']['automl']['r2_mean'])} |",
            f"| Original PINN pipeline | {_fmt(payload['aggregate']['original_pinn']['rmse_mean'])} | {_fmt(payload['aggregate']['original_pinn']['rmse_std'])} | {_fmt(payload['aggregate']['original_pinn']['mae_mean'])} | {_fmt(payload['aggregate']['original_pinn']['r2_mean'])} |",
            "",
            "## Notes",
            "",
            "- This is closer to the original classmate pipeline than the earlier structure-adapted PINN baseline.",
            (
                "- This run uses the provided LF/HF split, so it is the closest current reproduction of the classmate script."
                if provided_split
                else "- It is still not exactly her original experimental split unless her LF/HF train/eval/test CSVs are provided."
            ),
            "- The current dataset has constant `phi=0.4`, which weakens packing-physics layers.",
        ]
    )
    (out_dir / "automl_vs_classmate_original_pinn.md").write_text("\n".join(lines), encoding="utf-8")


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):.4f}"
    except Exception:
        return str(value)


def aggregate_seed_metrics(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    metrics = [row[key] for row in rows if row.get(key)]
    out: dict[str, Any] = {}
    for metric in ["rmse", "mae", "r2", "mape"]:
        vals = [float(m[metric]) for m in metrics if m.get(metric) is not None]
        out[f"{metric}_mean"] = float(np.mean(vals)) if vals else None
        out[f"{metric}_std"] = float(np.std(vals)) if vals else None
    return out


def run_provided_original_split(args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    """Run the classmate script on already-split LF/HF directories."""

    if not args.lf_data or not args.hf_data:
        raise ValueError("--provided-split requires --lf-data and --hf-data")
    lf_dir = Path(args.lf_data)
    hf_dir = Path(args.hf_data)
    required = [
        lf_dir / "train.csv",
        lf_dir / "test.csv",
        hf_dir / "train.csv",
        hf_dir / "eval.csv",
        hf_dir / "test.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Provided split is missing required files: " + ", ".join(missing))

    lf_train = pd.read_csv(lf_dir / "train.csv")
    lf_test = pd.read_csv(lf_dir / "test.csv")
    hf_train = pd.read_csv(hf_dir / "train.csv")
    hf_eval = pd.read_csv(hf_dir / "eval.csv")
    hf_test = pd.read_csv(hf_dir / "test.csv")
    fold_data = {
        "lf_dir": lf_dir,
        "hf_dir": hf_dir,
        "n_lf_train": len(lf_train),
        "n_lf_test": len(lf_test),
        "n_hf_train": len(hf_train),
        "n_hf_eval": len(hf_eval),
        "n_hf_test": len(hf_test),
    }

    seed_results: list[dict[str, Any]] = []
    for seed in args.seeds:
        row = run_original_fold(
            seed=seed,
            fold=1,
            fold_data=fold_data,
            out_dir=out_dir,
            phi_value=args.phi_value,
            hidden_dim=args.hidden_dim,
            lf_max_epochs=args.lf_max_epochs,
            hf_max_epochs=args.hf_max_epochs,
            lf_patience=args.lf_patience,
            hf_patience=args.hf_patience,
        )
        if row.get("status") != "ok":
            raise RuntimeError(f"Original PINN failed on provided split seed={seed}: {row.get('stderr_tail')}")
        seed_results.append(
            {
                "seed": seed,
                "automl": load_automl_seed(seed),
                "original_pinn": row["metrics"],
                "folds": [{k: v for k, v in row.items() if k not in {"y_true", "y_pred"}}],
            }
        )

    return {
        "seed_results": seed_results,
        "protocol_extra": {
            "provided_split": True,
            "lf_data": str(lf_dir),
            "hf_data": str(hf_dir),
            "small_sample_mode_expected": bool(len(hf_train) <= 60),
            "note": "Uses classmate-provided LF/HF split directories directly; no KFold split adapter is applied.",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DEFAULT_DATA_PATH)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--out-dir", default="agent_workspace/runs/yield_classmate_original_pinn_compare_20260721")
    parser.add_argument("--provided-split", action="store_true", help="Use already split LF/HF directories instead of building KFold splits.")
    parser.add_argument("--lf-data", default=None, help="Directory containing low_fidelity train.csv/test.csv.")
    parser.add_argument("--hf-data", default=None, help="Directory containing high_fidelity train.csv/eval.csv/test.csv.")
    parser.add_argument("--phi-value", type=float, default=0.4)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--lf-max-epochs", type=int, default=800)
    parser.add_argument("--hf-max-epochs", type=int, default=1200)
    parser.add_argument("--lf-patience", type=int, default=80)
    parser.add_argument("--hf-patience", type=int, default=180)
    args = parser.parse_args()

    ensure_original_files()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    protocol_extra: dict[str, Any] = {"provided_split": False}
    if args.provided_split:
        provided = run_provided_original_split(args, out_dir)
        seed_results = provided["seed_results"]
        protocol_extra = provided["protocol_extra"]
    else:
        raw = pd.read_csv(args.data) if str(args.data).lower().endswith(".csv") else pd.read_excel(args.data)
        df = adapt_to_original_schema(raw)

        seed_results = []
        for seed in args.seeds:
            splitter = KFold(n_splits=args.n_splits, shuffle=True, random_state=seed)
            oof_true: list[float] = []
            oof_pred: list[float] = []
            fold_rows: list[dict[str, Any]] = []
            for fold, (train_idx, test_idx) in enumerate(splitter.split(df), start=1):
                fold_root = out_dir / "fold_data" / f"seed{seed}_fold{fold}"
                fold_data = prepare_fold_data(
                    df,
                    seed=seed,
                    fold=fold,
                    train_idx=train_idx,
                    test_idx=test_idx,
                    out_dir=fold_root,
                )
                row = run_original_fold(
                    seed=seed,
                    fold=fold,
                    fold_data=fold_data,
                    out_dir=out_dir,
                    phi_value=args.phi_value,
                    hidden_dim=args.hidden_dim,
                    lf_max_epochs=args.lf_max_epochs,
                    hf_max_epochs=args.hf_max_epochs,
                    lf_patience=args.lf_patience,
                    hf_patience=args.hf_patience,
                )
                fold_rows.append({k: v for k, v in row.items() if k not in {"y_true", "y_pred"}})
                if row.get("status") != "ok":
                    raise RuntimeError(f"Original PINN failed for seed={seed}, fold={fold}: {row.get('stderr_tail')}")
                oof_true.extend(row["y_true"])
                oof_pred.extend(row["y_pred"])

            true_arr = np.asarray(oof_true, dtype=float)
            pred_arr = np.asarray(oof_pred, dtype=float)
            pinn_metrics = _metrics(true_arr, pred_arr)
            auto = load_automl_seed(seed)
            seed_results.append(
                {
                    "seed": seed,
                    "automl": auto,
                    "original_pinn": pinn_metrics,
                    "folds": fold_rows,
                }
            )

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "data_path": str(args.data),
        "protocol": {
            "seeds": list(args.seeds),
            "n_splits": int(args.n_splits),
            "phi_value": float(args.phi_value),
            "hidden_dim": int(args.hidden_dim),
            "lf_max_epochs": int(args.lf_max_epochs),
            "hf_max_epochs": int(args.hf_max_epochs),
            "schema_adapter": {
                "target": "yield_stress -> tau_y_final_pa",
                "forward_stage_map": dict(enumerate(FORWARD_STAGE_SOURCE, start=1)),
                "reverse_stage_map": dict(enumerate(REVERSE_STAGE_SOURCE, start=1)),
            },
            "lf_note": "No distinct LF table exists in current repo path; LF directories are populated from each outer train fold only. No synthetic LF data are generated.",
            **protocol_extra,
        },
        "seed_results": seed_results,
        "aggregate": {
            "automl": aggregate_seed_metrics(seed_results, "automl"),
            "original_pinn": aggregate_seed_metrics(seed_results, "original_pinn"),
        },
    }
    (out_dir / "automl_vs_classmate_original_pinn.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_markdown(out_dir, payload)
    print(f"Wrote {out_dir / 'automl_vs_classmate_original_pinn.json'}")
    print(f"Wrote {out_dir / 'automl_vs_classmate_original_pinn.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
