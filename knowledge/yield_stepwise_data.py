"""Schema and process features for 12-step yield-stress LF/HF CSVs."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from knowledge.yield_schema import INDEX_COLUMN, TARGET_COLUMN


STEP_COUNT = 12
STEP_FIELDS = {
    "药浆温度": "slurry_temp_c",
    "夹套温度": "jacket_temp_c",
    "水箱温度": "water_tank_temp_c",
    "静置时长": "rest_time_min",
    "非静置时长": "active_time_min",
    "静置转速": "rest_rpm",
    "非静置转速": "active_rpm",
    "锅内压力": "pressure_kpa",
    "扭矩均值": "torque_mean",
    "扭矩斜率": "torque_slope",
}
BASE_COLUMN_MAP = {
    "RDX料斗下料量设定值": "rdx_mass_kg",
    "粗AP下料至锅内物料重量": "coarse_ap_mass_kg",
    "细AP下料至锅内物料重量": "fine_ap_mass_kg",
    "屈服应力": TARGET_COLUMN,
}
PROC_PREFIX = "proc_"


def read_stepwise_source_table(path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = Path(path)
    errors: list[str] = []
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            return pd.read_csv(path, encoding=encoding), {"reader": "csv", "encoding": encoding}
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc}")
    raise ValueError(f"Could not read CSV {path}; tried utf-8/gb18030/gbk: {errors[:2]}")


def _numeric(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
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


def _base_hf_id(raw_id: str) -> str:
    text = str(raw_id)
    match = re.match(r"^(batch_\d+_HF_\d+)", text)
    if match:
        return match.group(1)
    match = re.match(r"^(batch_\d+)_LF_01_\d+$", text)
    if match:
        return match.group(1)
    match = re.match(r"^(batch_\d+)", text)
    if match:
        return match.group(1)
    return text


def normalize_stepwise_yield_dataframe(
    df: pd.DataFrame,
    *,
    data_fidelity: str,
    source: str,
    is_augmented: bool,
    sample_prefix: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Normalize 12-step Chinese process CSVs without collapsing step structure."""

    required = ["批次ID", *BASE_COLUMN_MAP.keys()]
    for step in range(1, STEP_COUNT + 1):
        for src_suffix in STEP_FIELDS:
            required.append(f"工步{step}_{src_suffix}")
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Stepwise yield dataset is missing required columns: {missing[:20]}")

    columns: dict[str, Any] = {}
    columns[INDEX_COLUMN] = [f"{sample_prefix}_{i:04d}" for i in range(1, len(df) + 1)]
    raw_batch_id = df["批次ID"].astype(str)
    columns["raw_batch_id"] = raw_batch_id
    columns["base_hf_id"] = raw_batch_id.map(_base_hf_id)
    for src, dst in BASE_COLUMN_MAP.items():
        columns[dst] = _numeric(df, src)

    for step in range(1, STEP_COUNT + 1):
        step_prefix = f"step_{step:02d}"
        for src_suffix, dst_suffix in STEP_FIELDS.items():
            columns[f"{step_prefix}_{dst_suffix}"] = _numeric(df, f"工步{step}_{src_suffix}")

    columns["phi"] = 0.4
    columns["data_fidelity"] = data_fidelity
    columns["source"] = source
    columns["is_augmented"] = bool(is_augmented)
    out = pd.DataFrame(columns, index=df.index)

    stepwise_columns = [column for column in out.columns if column.startswith("step_")]
    feature_columns = [
        column
        for column in out.columns
        if column not in {INDEX_COLUMN, TARGET_COLUMN, "raw_batch_id", "base_hf_id", "data_fidelity", "source", "is_augmented"}
        and pd.api.types.is_numeric_dtype(out[column])
    ]
    meta = {
        "source_schema": "yield_stepwise_process_20260804",
        "raw_columns": list(df.columns),
        "step_count": STEP_COUNT,
        "step_fields": STEP_FIELDS,
        "base_column_map": BASE_COLUMN_MAP,
        "stepwise_columns": stepwise_columns,
        "feature_columns": feature_columns,
        "target_column": TARGET_COLUMN,
        "target_unit": "Pa",
        "data_fidelity": data_fidelity,
        "is_augmented": bool(is_augmented),
        "id_columns": ["sample_id", "raw_batch_id", "base_hf_id"],
        "assumptions": [
            "phi is fixed to 0.4 because the raw stepwise files do not contain measured solid volume fraction.",
            "raw_batch_id and base_hf_id are metadata for grouped LF/HF evaluation and must not be used as model inputs.",
            "The 12 step_* columns preserve process order; they are not collapsed into the older forward/reverse schema.",
        ],
    }
    return out, meta


def add_stepwise_process_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Add deterministic leakage-free features from the 12-step process table."""

    out = df.copy()
    steps = list(range(1, STEP_COUNT + 1))

    rdx = _numeric(out, "rdx_mass_kg")
    coarse_ap = _numeric(out, "coarse_ap_mass_kg")
    fine_ap = _numeric(out, "fine_ap_mass_kg")
    ap_total = coarse_ap + fine_ap
    solid_total = rdx + ap_total

    out["proc_total_solid_mass_kg"] = solid_total
    out["proc_total_ap_mass_kg"] = ap_total
    out["proc_rdx_mass_fraction"] = _safe_div(rdx, solid_total)
    out["proc_ap_mass_fraction"] = _safe_div(ap_total, solid_total)
    out["proc_coarse_ap_fraction_of_ap"] = _safe_div(coarse_ap, ap_total)
    out["proc_fine_ap_fraction_of_ap"] = _safe_div(fine_ap, ap_total)
    out["proc_coarse_to_fine_ap_ratio"] = _safe_div(coarse_ap, fine_ap)
    out["proc_bimodal_ap_balance"] = 1.0 - _safe_div((coarse_ap - fine_ap).abs(), ap_total)

    rest_time = pd.Series(0.0, index=out.index)
    active_time = pd.Series(0.0, index=out.index)
    rest_rpm_time = pd.Series(0.0, index=out.index)
    active_rpm_time = pd.Series(0.0, index=out.index)
    active_rpm_sq_time = pd.Series(0.0, index=out.index)
    temp_time_integral = pd.Series(0.0, index=out.index)
    torque_time_integral = pd.Series(0.0, index=out.index)
    torque_slope_time_integral = pd.Series(0.0, index=out.index)
    pressure_time_integral = pd.Series(0.0, index=out.index)

    slurry_cols: list[str] = []
    jacket_cols: list[str] = []
    water_cols: list[str] = []
    pressure_cols: list[str] = []
    torque_mean_cols: list[str] = []
    torque_slope_cols: list[str] = []
    active_rpm_cols: list[str] = []

    for step in steps:
        prefix = f"step_{step:02d}"
        rest_t = _numeric(out, f"{prefix}_rest_time_min").clip(lower=0.0)
        active_t = _numeric(out, f"{prefix}_active_time_min").clip(lower=0.0)
        rest_rpm = _numeric(out, f"{prefix}_rest_rpm").clip(lower=0.0)
        active_rpm = _numeric(out, f"{prefix}_active_rpm").clip(lower=0.0)
        slurry = _numeric(out, f"{prefix}_slurry_temp_c")
        torque = _numeric(out, f"{prefix}_torque_mean")
        slope = _numeric(out, f"{prefix}_torque_slope")
        pressure = _numeric(out, f"{prefix}_pressure_kpa")
        total_t = rest_t + active_t

        rest_time = rest_time + rest_t
        active_time = active_time + active_t
        rest_rpm_time = rest_rpm_time + rest_rpm * rest_t
        active_rpm_time = active_rpm_time + active_rpm * active_t
        active_rpm_sq_time = active_rpm_sq_time + (active_rpm / 1000.0) ** 2 * active_t
        temp_time_integral = temp_time_integral + slurry * total_t
        torque_time_integral = torque_time_integral + torque * total_t
        torque_slope_time_integral = torque_slope_time_integral + slope * total_t
        pressure_time_integral = pressure_time_integral + pressure * total_t

        slurry_cols.append(f"{prefix}_slurry_temp_c")
        jacket_cols.append(f"{prefix}_jacket_temp_c")
        water_cols.append(f"{prefix}_water_tank_temp_c")
        pressure_cols.append(f"{prefix}_pressure_kpa")
        torque_mean_cols.append(f"{prefix}_torque_mean")
        torque_slope_cols.append(f"{prefix}_torque_slope")
        active_rpm_cols.append(f"{prefix}_active_rpm")

    total_time = rest_time + active_time
    total_rpm_time = rest_rpm_time + active_rpm_time

    out["proc_total_rest_time_min"] = rest_time
    out["proc_total_active_time_min"] = active_time
    out["proc_total_process_time_min"] = total_time
    out["proc_active_time_fraction"] = _safe_div(active_time, total_time)
    out["proc_total_rest_rpm_time"] = rest_rpm_time
    out["proc_total_active_rpm_time"] = active_rpm_time
    out["proc_total_rpm_time"] = total_rpm_time
    out["proc_active_weighted_rpm"] = _safe_div(active_rpm_time, active_time)
    out["proc_rest_weighted_rpm"] = _safe_div(rest_rpm_time, rest_time)
    out["proc_active_rpm_time_per_solid"] = _safe_div(active_rpm_time, solid_total)
    out["proc_active_mixing_intensity_2"] = active_rpm_sq_time

    slurry_df = out[slurry_cols].astype(float)
    jacket_df = out[jacket_cols].astype(float)
    water_df = out[water_cols].astype(float)
    pressure_df = out[pressure_cols].astype(float)
    torque_df = out[torque_mean_cols].astype(float)
    slope_df = out[torque_slope_cols].astype(float)
    active_rpm_df = out[active_rpm_cols].astype(float)

    out["proc_slurry_temp_start_c"] = slurry_df.iloc[:, 0]
    out["proc_slurry_temp_end_c"] = slurry_df.iloc[:, -1]
    out["proc_slurry_temp_rise_c"] = slurry_df.iloc[:, -1] - slurry_df.iloc[:, 0]
    out["proc_slurry_temp_max_c"] = slurry_df.max(axis=1)
    out["proc_slurry_temp_min_c"] = slurry_df.min(axis=1)
    out["proc_slurry_temp_range_c"] = slurry_df.max(axis=1) - slurry_df.min(axis=1)
    out["proc_slurry_temp_time_weighted_mean_c"] = _safe_div(temp_time_integral, total_time)
    out["proc_jacket_temp_mean_c"] = jacket_df.mean(axis=1)
    out["proc_water_tank_temp_mean_c"] = water_df.mean(axis=1)
    out["proc_jacket_slurry_delta_mean_c"] = jacket_df.mean(axis=1) - slurry_df.mean(axis=1)
    out["proc_water_slurry_delta_mean_c"] = water_df.mean(axis=1) - slurry_df.mean(axis=1)
    out["proc_pressure_mean_kpa"] = pressure_df.mean(axis=1)
    out["proc_pressure_std_kpa"] = pressure_df.std(axis=1).fillna(0.0)
    out["proc_pressure_time_weighted_mean_kpa"] = _safe_div(pressure_time_integral, total_time)
    out["proc_pressure_range_kpa"] = pressure_df.max(axis=1) - pressure_df.min(axis=1)

    early_slice = slice(0, 4)
    mid_slice = slice(4, 8)
    late_slice = slice(8, 12)
    out["proc_torque_mean_avg"] = torque_df.mean(axis=1)
    out["proc_torque_mean_max"] = torque_df.max(axis=1)
    out["proc_torque_mean_min"] = torque_df.min(axis=1)
    out["proc_torque_mean_range"] = torque_df.max(axis=1) - torque_df.min(axis=1)
    out["proc_torque_time_weighted_mean"] = _safe_div(torque_time_integral, total_time)
    out["proc_torque_early_mean"] = torque_df.iloc[:, early_slice].mean(axis=1)
    out["proc_torque_mid_mean"] = torque_df.iloc[:, mid_slice].mean(axis=1)
    out["proc_torque_late_mean"] = torque_df.iloc[:, late_slice].mean(axis=1)
    out["proc_torque_late_minus_early"] = out["proc_torque_late_mean"] - out["proc_torque_early_mean"]
    out["proc_torque_late_to_early_ratio"] = _safe_div(out["proc_torque_late_mean"], out["proc_torque_early_mean"])
    out["proc_torque_slope_max"] = slope_df.max(axis=1)
    out["proc_torque_slope_min"] = slope_df.min(axis=1)
    out["proc_torque_slope_mean"] = slope_df.mean(axis=1)
    out["proc_torque_slope_abs_mean"] = slope_df.abs().mean(axis=1)
    out["proc_torque_slope_time_integral"] = torque_slope_time_integral
    out["proc_positive_torque_slope_sum"] = slope_df.clip(lower=0.0).sum(axis=1)
    out["proc_negative_torque_slope_sum"] = slope_df.clip(upper=0.0).sum(axis=1)
    out["proc_active_rpm_mean"] = active_rpm_df.mean(axis=1)
    out["proc_active_rpm_max"] = active_rpm_df.max(axis=1)
    out["proc_active_rpm_range"] = active_rpm_df.max(axis=1) - active_rpm_df.min(axis=1)
    out["proc_temp_torque_interaction"] = out["proc_slurry_temp_time_weighted_mean_c"] * out["proc_torque_time_weighted_mean"]
    out["proc_shear_torque_interaction"] = _safe_div(out["proc_total_active_rpm_time"] * out["proc_torque_time_weighted_mean"], solid_total)

    engineered_columns = [col for col in out.columns if col.startswith(PROC_PREFIX)]
    out[engineered_columns] = out[engineered_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    meta = {
        "feature_layer": "yield_stepwise_process_features_v1",
        "n_engineered_features": len(engineered_columns),
        "engineered_feature_columns": engineered_columns,
        "feature_groups": {
            "composition": [col for col in engineered_columns if any(token in col for token in ("mass", "ap_", "rdx", "bimodal", "solid"))],
            "time_speed": [col for col in engineered_columns if any(token in col for token in ("time", "rpm", "mixing", "active", "rest"))],
            "thermal_pressure": [col for col in engineered_columns if any(token in col for token in ("temp", "pressure", "slurry", "jacket", "water"))],
            "torque_response": [col for col in engineered_columns if "torque" in col or "slope" in col],
        },
        "assumptions": [
            "All proc_* features are deterministic functions of process inputs only.",
            "Torque features are process-response signals; they are treated as measured inputs, not target leakage.",
            "Feature engineering preserves the original 12 step_* columns for sequence models.",
        ],
    }
    return out, meta


def _target_summary(df: pd.DataFrame) -> dict[str, float]:
    target = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    return {
        "min": float(np.nanmin(target)),
        "max": float(np.nanmax(target)),
        "mean": float(np.nanmean(target)),
        "std": float(np.nanstd(target, ddof=1)) if len(target) > 1 else 0.0,
    }


def prepare_stepwise_multifidelity_dataset(
    high_fidelity_path: str | Path,
    low_fidelity_path: str | Path,
    out_dir: str | Path,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hf_raw, hf_reader = read_stepwise_source_table(high_fidelity_path)
    lf_raw, lf_reader = read_stepwise_source_table(low_fidelity_path)
    hf_norm, hf_meta = normalize_stepwise_yield_dataframe(
        hf_raw,
        data_fidelity="high_fidelity",
        source="yield_stepwise_high_fidelity_csv",
        is_augmented=False,
        sample_prefix="hf_step",
    )
    lf_norm, lf_meta = normalize_stepwise_yield_dataframe(
        lf_raw,
        data_fidelity="low_fidelity",
        source="yield_stepwise_low_fidelity_csv",
        is_augmented=True,
        sample_prefix="lf_step",
    )

    hf_feat, feature_meta = add_stepwise_process_features(hf_norm)
    lf_feat, _ = add_stepwise_process_features(lf_norm)
    combined = pd.concat([hf_feat, lf_feat], ignore_index=True)

    hf_csv = out_dir / "high_fidelity_stepwise_normalized.csv"
    lf_csv = out_dir / "low_fidelity_stepwise_normalized.csv"
    combined_csv = out_dir / "multifidelity_stepwise_normalized.csv"
    hf_feat.to_csv(hf_csv, index=False, encoding="utf-8-sig")
    lf_feat.to_csv(lf_csv, index=False, encoding="utf-8-sig")
    combined.to_csv(combined_csv, index=False, encoding="utf-8-sig")

    report = {
        "input_high_fidelity_path": str(high_fidelity_path),
        "input_low_fidelity_path": str(low_fidelity_path),
        "high_fidelity_csv": str(hf_csv),
        "low_fidelity_csv": str(lf_csv),
        "combined_csv": str(combined_csv),
        "reader": {"high_fidelity": hf_reader, "low_fidelity": lf_reader},
        "n_high_fidelity": int(len(hf_feat)),
        "n_low_fidelity": int(len(lf_feat)),
        "n_total": int(len(combined)),
        "n_output_columns": int(len(combined.columns)),
        "n_stepwise_columns": int(len(hf_meta["stepwise_columns"])),
        "n_engineered_features": int(feature_meta["n_engineered_features"]),
        "target_summary": {
            "high_fidelity": _target_summary(hf_feat),
            "low_fidelity": _target_summary(lf_feat),
            "combined": _target_summary(combined),
        },
        "id_summary": {
            "hf_unique_raw_batch_id": int(hf_feat["raw_batch_id"].nunique()),
            "lf_unique_raw_batch_id": int(lf_feat["raw_batch_id"].nunique()),
            "hf_unique_base_hf_id": int(hf_feat["base_hf_id"].nunique()),
            "lf_unique_base_hf_id": int(lf_feat["base_hf_id"].nunique()),
        },
        "schema": {
            "source_schema": hf_meta["source_schema"],
            "base_column_map": BASE_COLUMN_MAP,
            "step_fields": STEP_FIELDS,
            "step_count": STEP_COUNT,
            "metadata_columns": ["sample_id", "raw_batch_id", "base_hf_id", "data_fidelity", "source", "is_augmented"],
            "target_column": TARGET_COLUMN,
            "stepwise_columns": hf_meta["stepwise_columns"],
            "engineered_feature_columns": feature_meta["engineered_feature_columns"],
        },
        "feature_groups": feature_meta["feature_groups"],
        "assumptions": [
            *hf_meta["assumptions"],
            *feature_meta["assumptions"],
            "The output keeps both raw 12-step columns and proc_* aggregate features.",
            "Downstream evaluations should use base_hf_id for grouped LF/HF leakage audits.",
        ],
    }
    report_path = out_dir / "stepwise_schema_feature_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "high_fidelity_csv": str(hf_csv),
        "low_fidelity_csv": str(lf_csv),
        "combined_csv": str(combined_csv),
        "report_path": str(report_path),
        "report": report,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Normalize 12-step LF/HF yield CSVs and add process features.")
    parser.add_argument("--input-hf", required=True)
    parser.add_argument("--input-lf", required=True)
    parser.add_argument("--out-dir", default="agent_workspace/data/yield_stepwise_multifidelity_20260804")
    args = parser.parse_args(argv)
    result = prepare_stepwise_multifidelity_dataset(args.input_hf, args.input_lf, args.out_dir)
    print(json.dumps({
        "high_fidelity_csv": result["high_fidelity_csv"],
        "low_fidelity_csv": result["low_fidelity_csv"],
        "combined_csv": result["combined_csv"],
        "report_path": result["report_path"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
