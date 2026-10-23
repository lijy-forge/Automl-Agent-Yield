#!/usr/bin/env python3
"""Prepare the new Chinese yield CSVs for the classmate multi-fidelity PINN scripts.

The normalized new CSVs keep the real process stage numbers, e.g. forward
1/2/3/4/6/7/9/10/11 and reverse 4/10. The classmate script expects a compact
schema with forward 1..9 and reverse 1..2. This script performs only that
schema adaptation plus a deterministic HF/LF split; it does not alter labels.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


SOURCE_ROOT = Path("agent_workspace/data/generated_yield_stress_multifidelity_20260727")
HIGH_IN = SOURCE_ROOT / "high_fidelity_normalized.csv"
LOW_IN = SOURCE_ROOT / "low_fidelity_normalized.csv"
OUT_ROOT = Path("agent_workspace/data/yield_predict_new_csv_classmate_split_20260803")
SEED = 20260803

FORWARD_STAGE_MAP = {
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 6,
    6: 7,
    7: 9,
    8: 10,
    9: 11,
}
REVERSE_STAGE_MAP = {
    1: 4,
    2: 10,
}

DIRECT_COLUMNS = [
    "slurry_temp_c",
    "water_tank_temp_c",
    "jacket_temp_c",
    "internal_pressure_kpa",
    "coarse_ap_mass_kg",
    "fine_ap_mass_kg",
    "rdx_mass_kg",
]

EXTRA_COLUMNS = [
    "source_workshop_temp_c",
    "source_workshop_humidity_pct",
    "source_file",
]


def _expected_columns() -> list[str]:
    columns = list(DIRECT_COLUMNS)
    for dest in range(1, 10):
        columns.extend([f"forward_rpm_{dest}", f"forward_time_{dest}_min"])
    for dest in range(1, 3):
        columns.extend([f"reverse_rpm_{dest}", f"reverse_time_{dest}_min"])
    columns.append("tau_y_final_pa")
    columns.extend(EXTRA_COLUMNS)
    return columns


def _require_columns(df: pd.DataFrame, required: list[str], label: str) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _to_classmate_schema(df: pd.DataFrame, label: str) -> pd.DataFrame:
    required = list(DIRECT_COLUMNS) + ["yield_stress", "room_temp_c", "room_humidity_pct", "source"]
    for source_stage in FORWARD_STAGE_MAP.values():
        required.extend([f"forward_rpm_{source_stage}", f"forward_time_{source_stage}_min"])
    for source_stage in REVERSE_STAGE_MAP.values():
        required.extend([f"reverse_rpm_{source_stage}", f"reverse_time_{source_stage}_min"])
    _require_columns(df, required, label)

    out = pd.DataFrame(index=df.index)
    for column in DIRECT_COLUMNS:
        out[column] = df[column]
    for dest_stage, source_stage in FORWARD_STAGE_MAP.items():
        out[f"forward_rpm_{dest_stage}"] = df[f"forward_rpm_{source_stage}"]
        out[f"forward_time_{dest_stage}_min"] = df[f"forward_time_{source_stage}_min"]
    for dest_stage, source_stage in REVERSE_STAGE_MAP.items():
        out[f"reverse_rpm_{dest_stage}"] = df[f"reverse_rpm_{source_stage}"]
        out[f"reverse_time_{dest_stage}_min"] = df[f"reverse_time_{source_stage}_min"]

    out["tau_y_final_pa"] = df["yield_stress"]
    out["source_workshop_temp_c"] = df["room_temp_c"]
    out["source_workshop_humidity_pct"] = df["room_humidity_pct"]
    out["source_file"] = df["source"]
    if "sample_id" in df.columns:
        out.insert(0, "sample_id", df["sample_id"])
    return out[[column for column in ["sample_id", *_expected_columns()] if column in out.columns]]


def _rank_stratified_indices(y: pd.Series, n_pick: int, seed: int) -> list[int]:
    if n_pick <= 0:
        return []
    if n_pick >= len(y):
        return list(y.index)
    rng = np.random.default_rng(seed)
    ordered = y.sort_values().index.to_numpy()
    bins = np.array_split(ordered, n_pick)
    picked = [int(rng.choice(bin_indices)) for bin_indices in bins if len(bin_indices) > 0]
    return picked


def _split_hf(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    test_idx = _rank_stratified_indices(df["tau_y_final_pa"], 5, SEED)
    remaining = df.drop(index=test_idx)
    eval_idx = _rank_stratified_indices(remaining["tau_y_final_pa"], 3, SEED + 1)
    train = remaining.drop(index=eval_idx)
    return {
        "train": train.sort_index().reset_index(drop=True),
        "eval": df.loc[eval_idx].sort_index().reset_index(drop=True),
        "test": df.loc[test_idx].sort_index().reset_index(drop=True),
        "full": df.sort_index().reset_index(drop=True),
    }


def _split_lf(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    test_idx = _rank_stratified_indices(df["tau_y_final_pa"], 10, SEED + 2)
    train = df.drop(index=test_idx)
    return {
        "train": train.sort_index().reset_index(drop=True),
        "test": df.loc[test_idx].sort_index().reset_index(drop=True),
        "full": df.sort_index().reset_index(drop=True),
    }


def _summary(df: pd.DataFrame) -> dict[str, float | int]:
    y = df["tau_y_final_pa"].astype(float)
    return {
        "n": int(len(df)),
        "min": float(y.min()),
        "max": float(y.max()),
        "mean": float(y.mean()),
        "std": float(y.std(ddof=1)) if len(df) > 1 else 0.0,
    }


def _write_split(split: dict[str, pd.DataFrame], root: Path, full_name: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    split["train"].to_csv(root / "train.csv", index=False)
    split["test"].to_csv(root / "test.csv", index=False)
    split["full"].to_csv(root / full_name, index=False)
    if "eval" in split:
        split["eval"].to_csv(root / "eval.csv", index=False)


def main() -> int:
    hf_raw = pd.read_csv(HIGH_IN)
    lf_raw = pd.read_csv(LOW_IN)
    hf = _to_classmate_schema(hf_raw, "high_fidelity")
    lf = _to_classmate_schema(lf_raw, "low_fidelity")

    hf_split = _split_hf(hf)
    lf_split = _split_lf(lf)
    _write_split(lf_split, OUT_ROOT / "low_fidelity", "dataset_full_real_180.csv")
    _write_split(hf_split, OUT_ROOT / "high_fidelity", "dataset_full_real_20.csv")

    report = {
        "purpose": "Schema adaptation for classmate multi-fidelity PINN scripts on the new Chinese CSV data.",
        "source": {
            "high_fidelity_normalized_csv": str(HIGH_IN),
            "low_fidelity_normalized_csv": str(LOW_IN),
        },
        "output": {
            "root": str(OUT_ROOT),
            "high_fidelity_dir": str(OUT_ROOT / "high_fidelity"),
            "low_fidelity_dir": str(OUT_ROOT / "low_fidelity"),
        },
        "stage_mapping": {
            "forward_dest_to_source": FORWARD_STAGE_MAP,
            "reverse_dest_to_source": REVERSE_STAGE_MAP,
            "note": "Destination stage numbers are compact aliases required by the classmate script; source stage numbers retain the physical process origin.",
        },
        "target": {
            "source_column": "yield_stress",
            "output_column": "tau_y_final_pa",
        },
        "split_policy": {
            "seed": SEED,
            "method": "rank-stratified deterministic target split",
            "hf": {"train": 12, "eval": 3, "test": 5},
            "lf": {"train": 170, "test": 10},
        },
        "summary": {
            "hf_full": _summary(hf_split["full"]),
            "hf_train": _summary(hf_split["train"]),
            "hf_eval": _summary(hf_split["eval"]),
            "hf_test": _summary(hf_split["test"]),
            "lf_full": _summary(lf_split["full"]),
            "lf_train": _summary(lf_split["train"]),
            "lf_test": _summary(lf_split["test"]),
        },
    }
    (OUT_ROOT / "schema_split_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))
    print(f"Wrote split data to {OUT_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
