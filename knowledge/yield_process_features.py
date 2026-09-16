"""Process-physics feature engineering for high-solid-content yield data.

This module adds deterministic, leakage-free features to the generated process
dataset. The features are "physics proxies": they summarize composition,
mixing history, and thermal/pressure state without looking at the target.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from knowledge.yield_baselines import run_fixed_yield_baselines
from knowledge.yield_schema import INDEX_COLUMN, TARGET_COLUMN, load_yield_dataframe


PHYS_PREFIX = "phys_"


def _num(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce").fillna(default).astype(float)


def _safe_div(num: pd.Series | np.ndarray, den: pd.Series | np.ndarray, eps: float = 1e-9) -> pd.Series:
    num_s = pd.Series(num)
    den_s = pd.Series(den)
    out = np.divide(
        num_s.to_numpy(dtype=float),
        den_s.to_numpy(dtype=float),
        out=np.zeros(len(num_s), dtype=float),
        where=np.abs(den_s.to_numpy(dtype=float)) > eps,
    )
    return pd.Series(out, index=num_s.index, dtype=float)


def _rpm_time_pairs(df: pd.DataFrame, direction: str) -> list[tuple[int, str, str]]:
    pairs: list[tuple[int, str, str]] = []
    pattern = re.compile(rf"^{re.escape(direction)}_rpm_(\d+)$")
    for column in df.columns:
        match = pattern.match(str(column))
        if not match:
            continue
        stage = int(match.group(1))
        time_col = f"{direction}_time_{stage}_min"
        if time_col in df.columns:
            pairs.append((stage, str(column), time_col))
    return sorted(pairs, key=lambda item: item[0])


def _time_sum(df: pd.DataFrame, pairs: list[tuple[int, str, str]]) -> pd.Series:
    total = pd.Series(0.0, index=df.index, dtype=float)
    for _stage, _rpm_col, time_col in pairs:
        total = total + _num(df, time_col).clip(lower=0.0)
    return total


def _rpm_time_integral(df: pd.DataFrame, pairs: list[tuple[int, str, str]]) -> pd.Series:
    total = pd.Series(0.0, index=df.index, dtype=float)
    for _stage, rpm_col, time_col in pairs:
        rpm = _num(df, rpm_col).clip(lower=0.0)
        time = _num(df, time_col).clip(lower=0.0)
        total = total + rpm * time
    return total


def _weighted_rpm_std(
    df: pd.DataFrame,
    pairs: list[tuple[int, str, str]],
    weighted_mean: pd.Series,
    total_time: pd.Series,
) -> pd.Series:
    var = pd.Series(0.0, index=df.index, dtype=float)
    for _stage, rpm_col, time_col in pairs:
        rpm = _num(df, rpm_col).clip(lower=0.0)
        time = _num(df, time_col).clip(lower=0.0)
        var = var + time * np.square(rpm - weighted_mean)
    return np.sqrt(_safe_div(var, total_time).clip(lower=0.0))


def add_generated_process_physics_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Add deterministic process-physics proxy features.

    The target is not used. All derived columns are prefixed with ``phys_`` so
    downstream audits can separate raw process variables from engineered ones.
    """

    out = df.copy()
    original_columns = list(out.columns)

    rdx = _num(out, "rdx_mass_kg")
    coarse_ap = _num(out, "coarse_ap_mass_kg")
    fine_ap = _num(out, "fine_ap_mass_kg")
    ap_total = coarse_ap + fine_ap
    solid_total = rdx + ap_total

    out["phys_total_solid_mass_kg"] = solid_total
    out["phys_total_ap_mass_kg"] = ap_total
    out["phys_rdx_mass_fraction"] = _safe_div(rdx, solid_total)
    out["phys_ap_mass_fraction"] = _safe_div(ap_total, solid_total)
    out["phys_coarse_ap_fraction_of_ap"] = _safe_div(coarse_ap, ap_total)
    out["phys_fine_ap_fraction_of_ap"] = _safe_div(fine_ap, ap_total)
    out["phys_coarse_ap_mass_fraction"] = _safe_div(coarse_ap, solid_total)
    out["phys_fine_ap_mass_fraction"] = _safe_div(fine_ap, solid_total)
    out["phys_ap_to_rdx_mass_ratio"] = _safe_div(ap_total, rdx)
    out["phys_coarse_to_fine_ap_ratio"] = _safe_div(coarse_ap, fine_ap)
    out["phys_bimodal_ap_balance"] = 1.0 - _safe_div((coarse_ap - fine_ap).abs(), ap_total)

    forward_pairs = _rpm_time_pairs(out, "forward")
    reverse_pairs = _rpm_time_pairs(out, "reverse")
    all_pairs = forward_pairs + reverse_pairs

    forward_time = _time_sum(out, forward_pairs)
    reverse_time = _time_sum(out, reverse_pairs)
    total_time = forward_time + reverse_time
    forward_integral = _rpm_time_integral(out, forward_pairs)
    reverse_integral = _rpm_time_integral(out, reverse_pairs)
    total_integral = forward_integral + reverse_integral

    out["phys_forward_time_total_min"] = forward_time
    out["phys_reverse_time_total_min"] = reverse_time
    out["phys_mixing_time_total_min"] = total_time
    out["phys_forward_rpm_time_integral"] = forward_integral
    out["phys_reverse_rpm_time_integral"] = reverse_integral
    out["phys_mixing_rpm_time_integral"] = total_integral
    out["phys_forward_weighted_rpm"] = _safe_div(forward_integral, forward_time)
    out["phys_reverse_weighted_rpm"] = _safe_div(reverse_integral, reverse_time)
    out["phys_mixing_weighted_rpm"] = _safe_div(total_integral, total_time)
    out["phys_reverse_time_fraction"] = _safe_div(reverse_time, total_time)
    out["phys_reverse_shear_fraction"] = _safe_div(reverse_integral, total_integral)
    out["phys_mixing_rpm_std"] = _weighted_rpm_std(out, all_pairs, out["phys_mixing_weighted_rpm"], total_time)

    late_pairs = [pair for pair in all_pairs if pair[0] >= 9]
    late_time = _time_sum(out, late_pairs)
    late_integral = _rpm_time_integral(out, late_pairs)
    out["phys_late_stage_time_fraction"] = _safe_div(late_time, total_time)
    out["phys_late_stage_shear_fraction"] = _safe_div(late_integral, total_integral)

    out["phys_specific_mixing_dose"] = _safe_div(total_integral, solid_total)
    out["phys_mass_weighted_mixing_dose"] = total_integral * solid_total

    slurry_temp = _num(out, "slurry_temp_c")
    room_temp = _num(out, "room_temp_c")
    jacket_temp = _num(out, "jacket_temp_c")
    water_temp = _num(out, "water_tank_temp_c")
    pressure = _num(out, "internal_pressure_kpa")
    humidity = _num(out, "room_humidity_pct")
    phi = _num(out, "phi")

    out["phys_slurry_minus_room_temp_c"] = slurry_temp - room_temp
    out["phys_jacket_minus_room_temp_c"] = jacket_temp - room_temp
    out["phys_water_minus_slurry_temp_c"] = water_temp - slurry_temp
    out["phys_water_minus_room_temp_c"] = water_temp - room_temp
    out["phys_pressure_deviation_kpa"] = pressure - 101.325
    out["phys_pressure_phi_interaction"] = pressure * phi
    out["phys_slurry_temp_mixing_dose"] = slurry_temp * total_integral
    out["phys_room_humidity_temp_index"] = humidity * room_temp

    engineered_columns = [col for col in out.columns if str(col).startswith(PHYS_PREFIX)]
    out[engineered_columns] = out[engineered_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    meta = {
        "feature_layer": "generated_process_physics_v1",
        "raw_columns": original_columns,
        "engineered_feature_columns": engineered_columns,
        "n_engineered_features": len(engineered_columns),
        "feature_groups": {
            "composition": [
                col for col in engineered_columns
                if any(token in col for token in ("mass", "ap_", "rdx", "bimodal"))
            ],
            "mixing_history": [
                col for col in engineered_columns
                if any(token in col for token in ("mixing", "rpm", "shear", "time", "dose", "stage"))
            ],
            "thermal_pressure": [
                col for col in engineered_columns
                if any(token in col for token in ("temp", "pressure", "humidity"))
            ],
        },
        "assumptions": [
            "Engineered features are deterministic functions of measured or generated process inputs only.",
            "mixing_dose and shear_fraction are process proxies based on rpm*time, not calibrated mechanical energy.",
            "No feature uses yield_stress or any fitted target-dependent quantity.",
        ],
    }
    return out, meta


def _baseline_snapshot(csv_path: Path, random_states: list[int]) -> dict[str, Any]:
    df, meta = load_yield_dataframe(csv_path)
    feature_columns = list(meta.get("feature_columns") or [])
    rows: list[dict[str, Any]] = []
    for seed in random_states:
        bundle = run_fixed_yield_baselines(
            df,
            feature_columns,
            n_splits=5,
            n_repeats=1,
            random_state=int(seed),
        )
        best = min(
            list(bundle.get("baseline_results") or []),
            key=lambda item: float(item.get("oof_rmse", item.get("mean_rmse", float("inf")))),
        )
        rows.append({
            "seed": int(seed),
            "best_baseline": best.get("name"),
            "oof_rmse": float(best.get("oof_rmse")),
            "oof_mae": float(best.get("oof_mae")),
            "oof_r2": float(best.get("oof_r2")),
            "oof_mape": float(best.get("oof_mape")),
        })
    return {
        "csv_path": str(csv_path),
        "schema": meta.get("source_schema"),
        "n_rows": int(len(df)),
        "n_features": int(len(feature_columns)),
        "feature_columns": feature_columns,
        "seeds": rows,
        "mean_best_oof_rmse": float(np.mean([row["oof_rmse"] for row in rows])),
        "std_best_oof_rmse": float(np.std([row["oof_rmse"] for row in rows])),
        "mean_best_oof_r2": float(np.mean([row["oof_r2"] for row in rows])),
        "std_best_oof_r2": float(np.std([row["oof_r2"] for row in rows])),
    }


def prepare_generated_process_feature_dataset(
    input_csv: str | Path,
    out_dir: str | Path,
    *,
    evaluate_baselines: bool = False,
    random_states: list[int] | None = None,
) -> dict[str, Any]:
    input_csv = Path(input_csv)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_df, raw_meta = load_yield_dataframe(input_csv)
    engineered_df, feature_meta = add_generated_process_physics_features(raw_df)

    out_csv = out_dir / "generated_yield_stress_data_engineered.csv"
    report_path = out_dir / "generated_yield_stress_feature_report.json"
    engineered_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    engineered_loaded, engineered_meta = load_yield_dataframe(out_csv)
    target = pd.to_numeric(engineered_loaded[TARGET_COLUMN], errors="coerce")
    report: dict[str, Any] = {
        "input_csv": str(input_csv),
        "output_csv": str(out_csv),
        "source_schema": engineered_meta.get("source_schema"),
        "n_rows": int(len(engineered_loaded)),
        "raw_feature_count": int(len(raw_meta.get("feature_columns") or [])),
        "engineered_feature_count": int(len(feature_meta["engineered_feature_columns"])),
        "final_feature_count": int(len(engineered_meta.get("feature_columns") or [])),
        "target_summary": {
            "min": float(np.nanmin(target)),
            "max": float(np.nanmax(target)),
            "mean": float(np.nanmean(target)),
        },
        **feature_meta,
    }

    if evaluate_baselines:
        seeds = random_states or [0, 1, 2]
        report["baseline_raw_vs_engineered"] = {
            "raw": _baseline_snapshot(input_csv, seeds),
            "engineered": _baseline_snapshot(out_csv, seeds),
            "notes": [
                "This is a quick OOF sanity check on augmented data, not proof of external generalization.",
                "Use it to verify that the feature layer is runnable and worth carrying into AutoML ablations.",
            ],
        }

    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"csv_path": str(out_csv), "report_path": str(report_path), "report": report}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Add process-physics features to generated yield data.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--out-dir", default="agent_workspace/data/generated_yield_stress")
    parser.add_argument("--evaluate-baselines", action="store_true")
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    args = parser.parse_args(argv)

    result = prepare_generated_process_feature_dataset(
        args.input_csv,
        args.out_dir,
        evaluate_baselines=bool(args.evaluate_baselines),
        random_states=list(args.seeds),
    )
    print(json.dumps({"csv_path": result["csv_path"], "report_path": result["report_path"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
