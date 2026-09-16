"""Schema adapter for yield-stress prediction datasets.

The real industrial dataset is not available yet, so this adapter supports the
temporary Lian 2025 and Zhou 1999 CSV layouts plus a future generic layout.
Generated training code should import this module instead of hard-coding column
names from any one temporary dataset.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_LIAN_DATA_PATH = (
    "/Users/lijiayao/my-project/Yield-value-prediction/"
    "data/lian2025/high_fidelity/all_400.csv"
)
DEFAULT_LIAN_TEST_PATH = (
    "/Users/lijiayao/my-project/Yield-value-prediction/"
    "data/lian2025/high_fidelity/table6.csv"
)

TARGET_COLUMN = "yield_stress"
INDEX_COLUMN = "sample_id"

GENERATED_PROCESS_COLUMN_MAP = {
    "夹套温度": "jacket_temp_c",
    "工房温度": "room_temp_c",
    "工房湿度": "room_humidity_pct",
    "水箱温度": "water_tank_temp_c",
    "药浆温度": "slurry_temp_c",
    "锅内压力": "internal_pressure_kpa",
    "RDX料斗下料量设定值": "rdx_mass_kg",
    "粗AP料斗下料量设定值": "coarse_ap_mass_kg",
    "细AP料斗下料量设定值": "fine_ap_mass_kg",
    "屈服应力": TARGET_COLUMN,
}

for _idx in (1, 2, 3, 4, 6, 7, 9, 10, 11):
    GENERATED_PROCESS_COLUMN_MAP[f"正转转速设定值_{_idx}"] = f"forward_rpm_{_idx}"
    GENERATED_PROCESS_COLUMN_MAP[f"正转时间设定值_{_idx}"] = f"forward_time_{_idx}_min"
for _idx in (4, 10):
    GENERATED_PROCESS_COLUMN_MAP[f"反转转速设定值_{_idx}"] = f"reverse_rpm_{_idx}"
    GENERATED_PROCESS_COLUMN_MAP[f"反转时间设定值_{_idx}"] = f"reverse_time_{_idx}_min"

LEAKAGE_COLUMNS = {
    "yield_stress",
    "Tau0_Pa",
    "tau_Pa",
    "target",
    "label",
    "phi_max",
    "phi_max_eff",
    "phi_max_eff_reference",
    "m1_true",
    "m1_lf",
    "plastic_viscosity",
    "plastic_viscosity_pa_s",
}


def _numeric(series: pd.Series, default: float | None = np.nan) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce")
    if default is not None:
        out = out.fillna(default)
    return out.astype(float)


def _ensure_id(df: pd.DataFrame) -> pd.Series:
    if INDEX_COLUMN in df.columns:
        return df[INDEX_COLUMN]
    if "Source" in df.columns:
        return df["Source"].astype(str) + "_" + pd.Series(np.arange(len(df))).astype(str)
    return pd.Series(np.arange(1, len(df) + 1), index=df.index)


def _from_lian(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = pd.DataFrame(index=df.index)
    out[INDEX_COLUMN] = _ensure_id(df)
    out["phi"] = _numeric(df["Phi"])
    out["sp_percent"] = _numeric(df["SP_percent"])
    out[TARGET_COLUMN] = _numeric(df["Tau0_Pa"])
    if "phi_max" in df.columns:
        out["phi_max_reference"] = _numeric(df["phi_max"])
    meta = {
        "source_schema": "lian2025",
        "feature_columns": ["phi", "sp_percent"],
        "target_column": TARGET_COLUMN,
        "target_unit": "Pa",
        "leakage_columns": sorted(LEAKAGE_COLUMNS),
        "notes": [
            "Temporary Lian 2025 cement-paste dataset.",
            "Original target column Tau0_Pa is normalized to yield_stress.",
            "SP_percent is a temporary Lian cement-paste additive column and is treated as superplasticizer/dispersant/plasticizer dosage in this dataset, not as a curing-agent or hardener column.",
            "phi_max is treated as an auxiliary reference column, not a model input.",
        ],
    }
    return out, meta


def _from_lian_table6_full(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Normalize the richer Materials 18, 2983 Table 6 layout.

    The public high-fidelity CSV used earlier only contained Phi/SP/Tau0. The
    table extracted from the paper has formulation columns too. Keep measured
    formulation fields as features, but do not include plastic viscosity or
    reference/proxy columns.
    """

    out = pd.DataFrame(index=df.index)
    out[INDEX_COLUMN] = _ensure_id(df)
    out["phi"] = _numeric(df["phi"] if "phi" in df.columns else df["phi_percent"] / 100.0)
    out["sp_percent"] = _numeric(df["sp_percent"])
    out[TARGET_COLUMN] = _numeric(df[TARGET_COLUMN] if TARGET_COLUMN in df.columns else df["Tau0_Pa"])

    measured_candidates = [
        "Vp_L",
        "Vw_L",
        "cement_kg_m3",
        "fly_ash_kg_m3",
        "water_kg_m3",
        "sp_kg_m3",
        "d50_um",
        "psd_width",
        "specific_surface_m2kg",
        "temperature_c",
        "mixing_time_min",
        "mixing_speed_rpm",
        "mixing_energy",
        "rest_time_min",
        "curing_agent_ratio",
    ]
    for col in measured_candidates:
        if col in df.columns:
            out[col] = _numeric(df[col])
    if {"cement_kg_m3", "fly_ash_kg_m3"}.issubset(out.columns):
        out["binder_kg_m3"] = out["cement_kg_m3"] + out["fly_ash_kg_m3"]
    if "w_b" in df.columns:
        out["w_b"] = _numeric(df["w_b"])
    elif {"water_kg_m3", "binder_kg_m3"}.issubset(out.columns):
        out["w_b"] = out["water_kg_m3"] / out["binder_kg_m3"].replace(0, np.nan)
    if "fa_ratio" in df.columns:
        out["fa_ratio"] = _numeric(df["fa_ratio"])
    elif {"fly_ash_kg_m3", "binder_kg_m3"}.issubset(out.columns):
        out["fa_ratio"] = out["fly_ash_kg_m3"] / out["binder_kg_m3"].replace(0, np.nan)
    if {"sp_kg_m3", "binder_kg_m3"}.issubset(out.columns):
        out["sp_binder_ratio"] = out["sp_kg_m3"] / out["binder_kg_m3"].replace(0, np.nan)

    feature_columns = [
        col
        for col in [
            "phi",
            "sp_percent",
            "w_b",
            "fa_ratio",
            "binder_kg_m3",
            "sp_binder_ratio",
            "Vp_L",
            "Vw_L",
            "cement_kg_m3",
            "fly_ash_kg_m3",
            "water_kg_m3",
            "sp_kg_m3",
            "d50_um",
            "psd_width",
            "specific_surface_m2kg",
            "temperature_c",
            "mixing_time_min",
            "mixing_speed_rpm",
            "mixing_energy",
            "rest_time_min",
            "curing_agent_ratio",
        ]
        if col in out.columns
    ]
    meta = {
        "source_schema": "lian2025_table6_full",
        "feature_columns": feature_columns,
        "target_column": TARGET_COLUMN,
        "target_unit": "Pa",
        "leakage_columns": sorted(LEAKAGE_COLUMNS),
        "notes": [
            "Full Materials 18, 2983 Table 6 formulation layout.",
            "Measured formulation fields are available for data audit and modeling.",
            "binder_kg_m3 and sp_binder_ratio are deterministic formulation features derived from measured masses.",
            "If particle/process columns are present in this CSV, they are treated as documented proxy/default fields for feature compatibility and must be described as assumptions.",
            "plastic_viscosity_pa_s is not used as an input because it is a rheology outcome measured with yield stress.",
            "No shear-rate curve is reported, so Herschel-Bulkley/Bingham residual losses are not strictly supported.",
        ],
    }
    return out, meta


def _from_zhou(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = pd.DataFrame(index=df.index)
    out[INDEX_COLUMN] = _ensure_id(df)
    out["phi"] = _numeric(df["phi"])
    out["d_s_um"] = _numeric(df["d_s_um"])
    if "powder" in df.columns:
        out["powder"] = df["powder"].astype(str)
    out[TARGET_COLUMN] = _numeric(df["tau_Pa"])
    if "m1_true" in df.columns:
        out["m1_true_reference"] = _numeric(df["m1_true"])
    if "m1_lf" in df.columns:
        out["m1_lf_reference"] = _numeric(df["m1_lf"])
    meta = {
        "source_schema": "zhou1999",
        "feature_columns": ["phi", "d_s_um"] + (["powder"] if "powder" in out.columns else []),
        "target_column": TARGET_COLUMN,
        "target_unit": "Pa",
        "leakage_columns": sorted(LEAKAGE_COLUMNS),
        "notes": [
            "Temporary Zhou 1999 Al2O3 suspension dataset.",
            "Original target column tau_Pa is normalized to yield_stress.",
            "m1_true/m1_lf are auxiliary generation/reference fields, not model inputs.",
        ],
    }
    return out, meta


def _from_synthetic(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = df.copy()
    out[INDEX_COLUMN] = _ensure_id(out)
    out[TARGET_COLUMN] = _numeric(out[TARGET_COLUMN])
    excluded = set(LEAKAGE_COLUMNS) | {
        INDEX_COLUMN,
        "sample_id",
        "source",
        "data_fidelity",
        "mix_id",
        "phi_percent",
    }
    feature_columns = [
        col
        for col in out.columns
        if col not in excluded
        and not col.endswith("_reference")
        and pd.api.types.is_numeric_dtype(pd.to_numeric(out[col], errors="coerce"))
    ]
    meta = {
        "source_schema": "synthetic_yield_lian2025_low_fidelity",
        "feature_columns": feature_columns,
        "target_column": TARGET_COLUMN,
        "target_unit": "Pa",
        "leakage_columns": sorted(LEAKAGE_COLUMNS | {"source", "data_fidelity"}),
        "notes": [
            "Physics-guided synthetic low-fidelity yield-stress dataset.",
            "Generated from a reproducible code generator anchored to Lian et al. (2025) Materials 18, 2983 Table 6.",
            "phi_max_eff_reference is a generation/reference column and must not be used as a model input.",
            "Use real Table 6 anchor data only for secondary anchor validation, not for OOF folds on synthetic training data.",
        ],
    }
    return out, meta


def _from_generated_process(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Normalize the generated process-control yield-stress dataset.

    This dataset has already been converted from the Chinese xlsx workbook to
    canonical process columns by knowledge.yield_generated_data. Keep its
    fidelity/source metadata, but never use those metadata fields as features.
    """

    out = df.copy()
    out[INDEX_COLUMN] = _ensure_id(out)
    out[TARGET_COLUMN] = _numeric(out[TARGET_COLUMN])
    excluded = set(LEAKAGE_COLUMNS) | {
        INDEX_COLUMN,
        "sample_id",
        "source",
        "data_fidelity",
        "is_augmented",
    }
    for col in out.columns:
        if col not in excluded:
            out[col] = pd.to_numeric(out[col], errors="ignore")
    feature_columns = [
        col
        for col in out.columns
        if col not in excluded
        and not col.endswith("_reference")
        and pd.api.types.is_numeric_dtype(pd.to_numeric(out[col], errors="coerce"))
    ]
    meta = {
        "source_schema": "generated_yield_process_202607",
        "feature_columns": feature_columns,
        "target_column": TARGET_COLUMN,
        "target_unit": "Pa",
        "leakage_columns": sorted(LEAKAGE_COLUMNS | {"source", "data_fidelity", "is_augmented"}),
        "notes": [
            "Generated/augmented process-control yield-stress dataset normalized from Chinese xlsx columns.",
            "source, data_fidelity, and is_augmented are metadata fields, not model inputs.",
            "Process-physics engineered features are added by knowledge.yield_process_features and are included here when present.",
        ],
    }
    return out, meta


def _from_generated_process_chinese(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Normalize Chinese process-control CSV columns into canonical names."""
    out = pd.DataFrame(index=df.index)
    out[INDEX_COLUMN] = _ensure_id(df)
    for src, dst in GENERATED_PROCESS_COLUMN_MAP.items():
        if src in df.columns:
            out[dst] = _numeric(df[src])

    if "phi" not in out.columns:
        out["phi"] = 0.4
    if "data_fidelity" in df.columns:
        out["data_fidelity"] = df["data_fidelity"].astype(str)
    if "source" in df.columns:
        out["source"] = df["source"].astype(str)
    if "is_augmented" in df.columns:
        out["is_augmented"] = df["is_augmented"]

    excluded = set(LEAKAGE_COLUMNS) | {
        INDEX_COLUMN,
        "sample_id",
        "source",
        "data_fidelity",
        "is_augmented",
    }
    feature_columns = [
        col
        for col in out.columns
        if col not in excluded
        and not col.endswith("_reference")
        and pd.api.types.is_numeric_dtype(pd.to_numeric(out[col], errors="coerce"))
    ]
    meta = {
        "source_schema": "generated_yield_process_chinese_csv",
        "feature_columns": feature_columns,
        "target_column": TARGET_COLUMN,
        "target_unit": "Pa",
        "leakage_columns": sorted(LEAKAGE_COLUMNS | {"source", "data_fidelity", "is_augmented"}),
        "column_map": {src: dst for src, dst in GENERATED_PROCESS_COLUMN_MAP.items() if src in df.columns},
        "notes": [
            "Chinese process-control yield-stress CSV normalized into canonical process columns.",
            "phi is filled as 0.4 when not measured in the raw file.",
            "source, data_fidelity, and is_augmented are metadata fields, not model inputs.",
        ],
    }
    return out, meta


def _from_stepwise_process(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Normalize already-standardized 12-step process-control yield data."""

    out = df.copy()
    out[INDEX_COLUMN] = _ensure_id(out)
    out[TARGET_COLUMN] = _numeric(out[TARGET_COLUMN])
    excluded = set(LEAKAGE_COLUMNS) | {
        INDEX_COLUMN,
        "sample_id",
        "raw_batch_id",
        "base_hf_id",
        "source",
        "data_fidelity",
        "is_augmented",
    }
    for col in out.columns:
        if col not in excluded:
            out[col] = pd.to_numeric(out[col], errors="ignore")
    feature_columns = [
        col
        for col in out.columns
        if col not in excluded
        and not col.endswith("_reference")
        and pd.api.types.is_numeric_dtype(pd.to_numeric(out[col], errors="coerce"))
    ]
    stepwise_columns = [col for col in out.columns if str(col).startswith("step_")]
    engineered_columns = [col for col in out.columns if str(col).startswith("proc_")]
    meta = {
        "source_schema": "yield_stepwise_process_20260804",
        "feature_columns": feature_columns,
        "stepwise_columns": stepwise_columns,
        "engineered_feature_columns": engineered_columns,
        "target_column": TARGET_COLUMN,
        "target_unit": "Pa",
        "leakage_columns": sorted(LEAKAGE_COLUMNS | {"raw_batch_id", "base_hf_id", "source", "data_fidelity", "is_augmented"}),
        "notes": [
            "Already-standardized 12-step process-control yield-stress dataset.",
            "raw_batch_id and base_hf_id are grouping metadata for LF/HF audits, not model inputs.",
            "step_* columns preserve the original 12-step process sequence.",
            "proc_* columns are deterministic process features derived without target values.",
        ],
    }
    return out, meta


def _from_generic(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    target_aliases = ["yield_stress", "yield_stress_pa", "tau0", "tau0_pa", "tau_Pa", "Tau0_Pa", "屈服值", "屈服应力"]
    target = next((col for col in target_aliases if col in df.columns), None)
    if target is None:
        raise ValueError(f"Unsupported yield CSV schema; target column not found. Tried: {target_aliases}")

    out = df.copy()
    out[INDEX_COLUMN] = _ensure_id(out)
    if target != TARGET_COLUMN:
        out[TARGET_COLUMN] = _numeric(out[target])
    else:
        out[TARGET_COLUMN] = _numeric(out[TARGET_COLUMN])

    rename_map = {
        "Phi": "phi",
        "固体体积分数": "phi",
        "d50": "d50",
        "D50": "d50",
        "粒径中位径": "d50",
        "sigma_d": "sigma_d",
        "PSD_width": "sigma_d",
        "粒径分布宽度": "sigma_d",
        "Emix": "mixing_energy",
        "emix": "mixing_energy",
        "累积混合功": "mixing_energy",
        "T": "temperature",
        "温度": "temperature",
    }
    out = out.rename(columns={k: v for k, v in rename_map.items() if k in out.columns})
    for col in out.columns:
        if col not in {INDEX_COLUMN, TARGET_COLUMN}:
            out[col] = pd.to_numeric(out[col], errors="ignore")
    feature_columns = [
        col
        for col in out.columns
        if col not in {INDEX_COLUMN, TARGET_COLUMN}
        and col not in LEAKAGE_COLUMNS
        and col not in {"source", "data_fidelity", "is_augmented", "raw_batch_id", "base_hf_id"}
        and not col.endswith("_reference")
    ]
    meta = {
        "source_schema": "generic_yield",
        "feature_columns": feature_columns,
        "target_column": TARGET_COLUMN,
        "target_unit": "Pa",
        "leakage_columns": sorted(LEAKAGE_COLUMNS),
        "notes": ["Generic yield-stress schema inferred from column aliases."],
    }
    return out, meta


def normalize_yield_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    cols = set(df.columns)
    if {"yield_stress", "phi", "sp_percent", "cement_kg_m3", "water_kg_m3"}.issubset(cols):
        out, meta = _from_lian_table6_full(df)
    elif {"yield_stress", "step_01_slurry_temp_c", "step_12_torque_slope"}.issubset(cols):
        out, meta = _from_stepwise_process(df)
    elif {"yield_stress", "slurry_temp_c", "internal_pressure_kpa", "coarse_ap_mass_kg", "fine_ap_mass_kg"}.issubset(cols):
        out, meta = _from_generated_process(df)
    elif {"屈服应力", "药浆温度", "锅内压力", "粗AP料斗下料量设定值", "细AP料斗下料量设定值"}.issubset(cols):
        out, meta = _from_generated_process_chinese(df)
    elif {"yield_stress", "phi", "mixing_energy", "phi_max_eff_reference"}.issubset(cols):
        out, meta = _from_synthetic(df)
    elif {"Phi", "SP_percent", "Tau0_Pa"}.issubset(cols):
        out, meta = _from_lian(df)
    elif {"phi", "d_s_um", "tau_Pa"}.issubset(cols):
        out, meta = _from_zhou(df)
    else:
        out, meta = _from_generic(df)

    out = out.copy()
    out[TARGET_COLUMN] = _numeric(out[TARGET_COLUMN], default=np.nan)
    before = len(out)
    out = out[np.isfinite(out[TARGET_COLUMN].to_numpy(dtype=float)) & (out[TARGET_COLUMN] >= 0)].reset_index(drop=True)
    meta["n_raw"] = int(before)
    meta["n_retained"] = int(len(out))
    meta["n_dropped"] = int(before - len(out))
    return out, meta


def load_yield_dataframe(path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    errors: list[str] = []
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            df = pd.read_csv(path, encoding=encoding)
            break
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc}")
    else:
        raise ValueError(
            "Could not read CSV with utf-8-sig/utf-8/gb18030/gbk. "
            + "; ".join(errors[:2])
        )
    meta = {"data_path": str(path)}
    out, schema_meta = normalize_yield_dataframe(df)
    meta.update(schema_meta)
    return out, meta
