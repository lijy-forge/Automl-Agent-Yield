#!/usr/bin/env python3
"""Stability tweak for bilinear m1-level gate.

Runs a new variant where only the bilinear gate parameters have scaled gradients
(effective smaller learning rate), then compares against existing 10-seed runs.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


NEW_VARIANT = {
    "name": "bilinear_m1_slow_gate",
    "root": "agent_workspace/external_baselines/gate_bilinear_m1_slow_gate",
    "description": "Bilinear m1-level gate with gate-parameter gradient scale 0.35.",
}

BASELINE_RESULTS = [
    {
        "name": "original_mlp_gate",
        "path": Path("agent_workspace/runs/yield_pinn_neural_gate_ablation_20260803/original_mlp_gate/original_mlp_gate_result.json"),
    },
    {
        "name": "bilinear_m1_level_gate",
        "path": Path("agent_workspace/runs/yield_pinn_bilinear_m1_combo_20260803/bilinear_m1_level_gate/bilinear_m1_level_gate_result.json"),
    },
    {
        "name": "bilinear_interaction_gate",
        "path": Path("agent_workspace/runs/yield_pinn_neural_gate_ablation_20260803/bilinear_interaction_gate/bilinear_interaction_gate_result.json"),
    },
]


def _fmt(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except Exception:
        return str(value)


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _r2_mean(per_seed: list[dict[str, Any]]) -> float:
    return float(np.mean([float(row["r2"]) for row in per_seed]))


def run_new(out_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    variant_out = out_dir / NEW_VARIANT["name"]
    result_path = variant_out / f"{NEW_VARIANT['name']}_result.json"
    if result_path.exists() and not args.force:
        return _load(result_path)

    cmd = [
        sys.executable,
        "run_yield_pinn_branch_ablation_experiment.py",
        "--variant-root",
        NEW_VARIANT["root"],
        "--variant-name",
        NEW_VARIANT["name"],
        "--out-dir",
        str(variant_out),
        "--seeds",
        *[str(seed) for seed in args.seeds],
        "--lf-max-epochs",
        str(args.lf_max_epochs),
        "--hf-max-epochs",
        str(args.hf_max_epochs),
        "--lf-patience",
        str(args.lf_patience),
        "--hf-patience",
        str(args.hf_patience),
    ]
    subprocess.run(cmd, check=True)
    return _load(result_path)


def summarize(results: list[dict[str, Any]], out_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    best_existing = min(
        [row for row in results if row["variant"] != NEW_VARIANT["name"]],
        key=lambda row: float(row["rmse_mean"]),
    )
    best_by_seed = {int(row["seed"]): row for row in best_existing["per_seed"]}
    enriched = []
    for result in results:
        better = 0
        deltas = []
        for row in result["per_seed"]:
            seed = int(row["seed"])
            if seed in best_by_seed:
                delta = float(row["rmse"]) - float(best_by_seed[seed]["rmse"])
                deltas.append(delta)
                if delta < 0:
                    better += 1
        enriched.append(
            {
                **result,
                "r2_mean": _r2_mean(result["per_seed"]),
                "rmse_delta_vs_best_existing_mean": float(np.mean(deltas)) if deltas else 0.0,
                "seeds_better_than_best_existing": None if result["variant"] == best_existing["variant"] else better,
            }
        )
    ranked = sorted(enriched, key=lambda row: (float(row["rmse_mean"]), float(row["seed_ensemble_rmse"])))
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "protocol": {
            "data": "provided 180 LF / 20 HF split",
            "seeds": args.seeds,
            "lf_max_epochs": args.lf_max_epochs,
            "hf_max_epochs": args.hf_max_epochs,
            "new_variant": NEW_VARIANT,
            "baselines": [{**b, "path": str(b["path"])} for b in BASELINE_RESULTS],
            "question": "Does slowing the bilinear gate optimizer improve seed stability without losing much accuracy?",
        },
        "best_existing": best_existing,
        "ranked": ranked,
        "winner": ranked[0],
    }
    (out_dir / "bilinear_m1_stability_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def write_md(payload: dict[str, Any], out_dir: Path) -> None:
    best_existing = payload["best_existing"]["variant"]
    lines = [
        "# Bilinear m1-Level Gate Stability Summary",
        "",
        f"- Question: {payload['protocol']['question']}",
        f"- Best existing baseline: `{best_existing}`",
        "",
        "| Rank | Variant | RMSE mean | RMSE std | MAE mean | R2 mean | Ensemble RMSE | Ensemble R2 | Delta vs best existing | Better seeds vs best existing |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for idx, row in enumerate(payload["ranked"], start=1):
        better = "NA" if row["seeds_better_than_best_existing"] is None else str(row["seeds_better_than_best_existing"])
        lines.append(
            f"| {idx} | `{row['variant']}` | {_fmt(row['rmse_mean'])} | {_fmt(row['rmse_std'])} | "
            f"{_fmt(row['mae_mean'])} | {_fmt(row['r2_mean'])} | {_fmt(row['seed_ensemble_rmse'])} | "
            f"{_fmt(row['seed_ensemble_r2'])} | {_fmt(row['rmse_delta_vs_best_existing_mean'])} | {better} |"
        )
    lines.extend(
        [
            "",
            "## Readout",
            "",
            f"- Winner by 10-seed mean RMSE: `{payload['winner']['variant']}`.",
            "- The new variant only scales bilinear gate gradients; the architecture and physical layer are unchanged.",
            "- Lower RMSE std means better seed stability; lower RMSE mean means better average accuracy.",
        ]
    )
    (out_dir / "bilinear_m1_stability_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=f"agent_workspace/runs/yield_pinn_bilinear_m1_stability_{datetime.now().strftime('%Y%m%d_%H%M')}")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--lf-max-epochs", type=int, default=400)
    parser.add_argument("--hf-max-epochs", type=int, default=800)
    parser.add_argument("--lf-patience", type=int, default=60)
    parser.add_argument("--hf-patience", type=int, default=150)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = [_load(b["path"]) for b in BASELINE_RESULTS]
    print(f"\n=== Running {NEW_VARIANT['name']} ===", flush=True)
    results.append(run_new(out_dir, args))
    payload = summarize(results, out_dir, args)
    write_md(payload, out_dir)
    print(f"\nWrote summary to {out_dir / 'bilinear_m1_stability_summary.md'}", flush=True)
    print(f"Winner: {payload['winner']['variant']} RMSE={payload['winner']['rmse_mean']:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
