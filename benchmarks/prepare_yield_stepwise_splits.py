#!/usr/bin/env python3
"""Create paired and group-holdout splits for 12-step LF/HF yield data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_DATA_DIR = Path("agent_workspace/data/yield_stepwise_multifidelity_20260804")
DEFAULT_OUT_DIR = DEFAULT_DATA_DIR / "splits"
SEED = 20260804
TARGET = "yield_stress"
GROUP = "base_hf_id"


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


def _summary(df: pd.DataFrame) -> dict[str, float | int]:
    y = pd.to_numeric(df[TARGET], errors="coerce")
    return {
        "n": int(len(df)),
        "min": float(y.min()) if len(y) else float("nan"),
        "max": float(y.max()) if len(y) else float("nan"),
        "mean": float(y.mean()) if len(y) else float("nan"),
        "std": float(y.std(ddof=1)) if len(y) > 1 else 0.0,
        "n_groups": int(df[GROUP].nunique()) if GROUP in df.columns else 0,
    }


def _split_hf(hf: pd.DataFrame, seed: int) -> dict[str, pd.DataFrame]:
    test_idx = _rank_stratified_indices(hf[TARGET], 5, seed)
    remaining = hf.drop(index=test_idx)
    eval_idx = _rank_stratified_indices(remaining[TARGET], 3, seed + 1)
    train = remaining.drop(index=eval_idx)
    return {
        "train": train.sort_index().reset_index(drop=True),
        "eval": hf.loc[eval_idx].sort_index().reset_index(drop=True),
        "test": hf.loc[test_idx].sort_index().reset_index(drop=True),
        "full": hf.sort_index().reset_index(drop=True),
    }


def _split_lf_paired(lf: pd.DataFrame, seed: int) -> dict[str, pd.DataFrame]:
    test_idx = _rank_stratified_indices(lf[TARGET], 10, seed + 2)
    train = lf.drop(index=test_idx)
    return {
        "train": train.sort_index().reset_index(drop=True),
        "test": lf.loc[test_idx].sort_index().reset_index(drop=True),
        "full": lf.sort_index().reset_index(drop=True),
    }


def _split_lf_group_holdout(lf: pd.DataFrame, hf_test: pd.DataFrame) -> dict[str, pd.DataFrame]:
    holdout_groups = set(hf_test[GROUP].astype(str))
    mask = lf[GROUP].astype(str).isin(holdout_groups)
    train = lf.loc[~mask]
    test = lf.loc[mask]
    return {
        "train": train.sort_index().reset_index(drop=True),
        "test": test.sort_index().reset_index(drop=True),
        "full": lf.sort_index().reset_index(drop=True),
    }


def _write_family(split: dict[str, pd.DataFrame], root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for name, frame in split.items():
        frame.to_csv(root / f"{name}.csv", index=False, encoding="utf-8-sig")


def _leakage_audit(hf_split: dict[str, pd.DataFrame], lf_split: dict[str, pd.DataFrame]) -> dict[str, Any]:
    hf_test_groups = set(hf_split["test"][GROUP].astype(str))
    lf_train_groups = set(lf_split["train"][GROUP].astype(str))
    overlap = sorted(hf_test_groups & lf_train_groups)
    return {
        "hf_test_groups": sorted(hf_test_groups),
        "lf_train_groups": sorted(lf_train_groups),
        "n_hf_test_groups": int(len(hf_test_groups)),
        "n_lf_train_groups": int(len(lf_train_groups)),
        "hf_test_groups_in_lf_train": overlap,
        "n_hf_test_groups_in_lf_train": int(len(overlap)),
        "strict_group_holdout_passed": bool(len(overlap) == 0),
    }


def _report(mode: str, hf_split: dict[str, pd.DataFrame], lf_split: dict[str, pd.DataFrame], seed: int) -> dict[str, Any]:
    return {
        "mode": mode,
        "seed": int(seed),
        "target": TARGET,
        "group_column": GROUP,
        "summary": {
            "hf_train": _summary(hf_split["train"]),
            "hf_eval": _summary(hf_split["eval"]),
            "hf_test": _summary(hf_split["test"]),
            "hf_full": _summary(hf_split["full"]),
            "lf_train": _summary(lf_split["train"]),
            "lf_test": _summary(lf_split["test"]),
            "lf_full": _summary(lf_split["full"]),
        },
        "leakage_audit": _leakage_audit(hf_split, lf_split),
        "interpretation": {
            "paired": "Allows LF variants from HF-test groups in LF train; useful for paired LF-assisted engineering use cases.",
            "group_holdout": "Excludes LF variants from HF-test groups in LF train; stricter externalization check.",
        }[mode],
    }


def create_splits(data_dir: Path, out_dir: Path, seed: int) -> dict[str, Any]:
    hf = pd.read_csv(data_dir / "high_fidelity_stepwise_normalized.csv", encoding="utf-8-sig")
    lf = pd.read_csv(data_dir / "low_fidelity_stepwise_normalized.csv", encoding="utf-8-sig")
    for label, frame in [("high_fidelity", hf), ("low_fidelity", lf)]:
        missing = [col for col in [TARGET, GROUP] if col not in frame.columns]
        if missing:
            raise ValueError(f"{label} is missing split columns: {missing}")

    hf_split = _split_hf(hf, seed)
    modes: dict[str, dict[str, pd.DataFrame]] = {
        "paired": _split_lf_paired(lf, seed),
        "group_holdout": _split_lf_group_holdout(lf, hf_split["test"]),
    }

    reports: dict[str, Any] = {}
    for mode, lf_split in modes.items():
        root = out_dir / f"{mode}_seed_{seed}"
        _write_family(hf_split, root / "high_fidelity")
        _write_family(lf_split, root / "low_fidelity")
        report = _report(mode, hf_split, lf_split, seed)
        report["paths"] = {
            "root": str(root),
            "high_fidelity": str(root / "high_fidelity"),
            "low_fidelity": str(root / "low_fidelity"),
        }
        (root / "split_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        reports[mode] = report

    index_report = {
        "data_dir": str(data_dir),
        "out_dir": str(out_dir),
        "seed": int(seed),
        "modes": {
            mode: {
                "root": report["paths"]["root"],
                "strict_group_holdout_passed": report["leakage_audit"]["strict_group_holdout_passed"],
                "n_hf_test_groups_in_lf_train": report["leakage_audit"]["n_hf_test_groups_in_lf_train"],
                "hf_test": report["summary"]["hf_test"],
                "lf_train": report["summary"]["lf_train"],
            }
            for mode, report in reports.items()
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"split_index_seed_{seed}.json").write_text(json.dumps(index_report, ensure_ascii=False, indent=2), encoding="utf-8")
    return index_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Create paired/group-holdout splits for stepwise LF/HF yield data.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    result = create_splits(Path(args.data_dir), Path(args.out_dir), int(args.seed))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
