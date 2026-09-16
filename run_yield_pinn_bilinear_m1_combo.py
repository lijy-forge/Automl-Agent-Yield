#!/usr/bin/env python3
"""Run and summarize bilinear neural gate combined with m1-level fusion.

The new variant combines two previously separated positive findings:

1. bilinear_interaction_gate: better neural gate at the original z-level fusion.
2. m1_level_gate: better fusion target by fusing branch raw_m1 estimates.

This script trains only the new combined variant, then compares it with the
existing 10-seed baseline result files from the two earlier ablation runs.
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
    "name": "bilinear_m1_level_gate",
    "root": "agent_workspace/external_baselines/gate_bilinear_m1_level",
    "description": "Bilinear neural gate produces branch weights; the weights fuse three branch raw_m1 estimates.",
}

BASELINE_RESULTS = [
    {
        "name": "original_mlp_gate",
        "path": Path("agent_workspace/runs/yield_pinn_neural_gate_ablation_20260803/original_mlp_gate/original_mlp_gate_result.json"),
        "description": "Original z-level MLP gate.",
    },
    {
        "name": "m1_level_gate",
        "path": Path("agent_workspace/runs/yield_pinn_gate_fusion_ablation_20260803/m1_level_gate/m1_level_gate_result.json"),
        "description": "MLP gate, but fusion target is branch raw_m1 estimates.",
    },
    {
        "name": "bilinear_interaction_gate",
        "path": Path("agent_workspace/runs/yield_pinn_neural_gate_ablation_20260803/bilinear_interaction_gate/bilinear_interaction_gate_result.json"),
        "description": "Bilinear neural gate at the original z-level fusion position.",
    },
]


def _fmt(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except Exception:
        return str(value)


def _aggregate_r2(per_seed: list[dict[str, Any]]) -> dict[str, float]:
    values = [float(row["r2"]) for row in per_seed]
    return {"mean": float(np.mean(values)), "std": float(np.std(values))}


def _load_result(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required baseline result is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def run_new_variant(out_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    variant_out = out_dir / NEW_VARIANT["name"]
    result_path = variant_out / f"{NEW_VARIANT['name']}_result.json"
    if result_path.exists() and not args.force:
        return _load_result(result_path)

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
    return _load_result(result_path)


def build_summary(results: list[dict[str, Any]], out_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    original = next(row for row in results if row["variant"] == "original_mlp_gate")
    original_by_seed = {int(row["seed"]): row for row in original["per_seed"]}
    single_baselines = [row for row in results if row["variant"] in {"m1_level_gate", "bilinear_interaction_gate"}]
    best_single = min(single_baselines, key=lambda row: float(row["rmse_mean"]))
    best_single_by_seed = {int(row["seed"]): row for row in best_single["per_seed"]}

    enriched = []
    for result in results:
        r2 = _aggregate_r2(result["per_seed"])
        deltas_original = []
        deltas_best_single = []
        better_original = 0
        better_best_single = 0
        for row in result["per_seed"]:
            seed = int(row["seed"])
            if seed in original_by_seed:
                delta = float(row["rmse"]) - float(original_by_seed[seed]["rmse"])
                deltas_original.append(delta)
                if delta < 0:
                    better_original += 1
            if seed in best_single_by_seed:
                delta = float(row["rmse"]) - float(best_single_by_seed[seed]["rmse"])
                deltas_best_single.append(delta)
                if delta < 0:
                    better_best_single += 1
        enriched.append(
            {
                **result,
                "r2_mean": r2["mean"],
                "r2_std": r2["std"],
                "rmse_delta_vs_original_mean": float(np.mean(deltas_original)) if deltas_original else 0.0,
                "rmse_delta_vs_best_single_mean": float(np.mean(deltas_best_single)) if deltas_best_single else 0.0,
                "seeds_better_than_original": None if result["variant"] == "original_mlp_gate" else better_original,
                "seeds_better_than_best_single": None if result["variant"] == best_single["variant"] else better_best_single,
            }
        )

    ranked = sorted(enriched, key=lambda row: (float(row["rmse_mean"]), float(row["seed_ensemble_rmse"])))
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "protocol": {
            "data": "provided 180 LF / 20 HF split",
            "hf_test": "held-out provided HF test.csv",
            "seeds": args.seeds,
            "lf_max_epochs": args.lf_max_epochs,
            "hf_max_epochs": args.hf_max_epochs,
            "lf_patience": args.lf_patience,
            "hf_patience": args.hf_patience,
            "new_variant": NEW_VARIANT,
            "baselines": [
                {**baseline, "path": str(baseline["path"])}
                for baseline in BASELINE_RESULTS
            ],
            "combination_question": "Does bilinear neural gate still help when the fusion target is moved from z representations to branch raw_m1 estimates?",
        },
        "ranked": ranked,
        "winner": ranked[0],
        "best_single_baseline": best_single,
    }
    (out_dir / "bilinear_m1_combo_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def write_markdown(payload: dict[str, Any], out_dir: Path) -> None:
    protocol = payload["protocol"]
    best_single = payload["best_single_baseline"]["variant"]
    lines = [
        "# Bilinear Gate + m1-Level Fusion Combo Summary",
        "",
        "## Protocol",
        "",
        f"- Data: `{protocol['data']}`",
        f"- HF test: `{protocol['hf_test']}`",
        f"- Seeds: `{protocol['seeds']}`",
        f"- Epoch budget: LF `{protocol['lf_max_epochs']}`, HF `{protocol['hf_max_epochs']}`",
        f"- Question: {protocol['combination_question']}",
        f"- Best single-change baseline before combo: `{best_single}`",
        "",
        "## Results",
        "",
        "| Rank | Variant | RMSE mean | RMSE std | MAE mean | R2 mean | Ensemble RMSE | Ensemble R2 | Delta vs original | Better than original | Delta vs best single | Better than best single |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for idx, row in enumerate(payload["ranked"], start=1):
        better_original = "NA" if row["seeds_better_than_original"] is None else str(row["seeds_better_than_original"])
        better_single = "NA" if row["seeds_better_than_best_single"] is None else str(row["seeds_better_than_best_single"])
        lines.append(
            f"| {idx} | `{row['variant']}` | {_fmt(row['rmse_mean'])} | {_fmt(row['rmse_std'])} | "
            f"{_fmt(row['mae_mean'])} | {_fmt(row['r2_mean'])} | {_fmt(row['seed_ensemble_rmse'])} | "
            f"{_fmt(row['seed_ensemble_r2'])} | {_fmt(row['rmse_delta_vs_original_mean'])} | "
            f"{better_original} | {_fmt(row['rmse_delta_vs_best_single_mean'])} | {better_single} |"
        )

    winner = payload["winner"]
    lines.extend(
        [
            "",
            "## Readout",
            "",
            f"- Winner by 10-seed mean RMSE: `{winner['variant']}`.",
            "- Negative deltas mean lower RMSE than the comparison row.",
            "- This comparison separates the new combo from the two single-change baselines: m1-level fusion alone and bilinear z-level gate alone.",
            "- HF test is small, so mean RMSE, RMSE std, and seed ensemble RMSE should be read together.",
        ]
    )
    (out_dir / "bilinear_m1_combo_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=f"agent_workspace/runs/yield_pinn_bilinear_m1_combo_{datetime.now().strftime('%Y%m%d_%H%M')}")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--lf-max-epochs", type=int, default=400)
    parser.add_argument("--hf-max-epochs", type=int, default=800)
    parser.add_argument("--lf-patience", type=int, default=60)
    parser.add_argument("--hf-patience", type=int, default=150)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for baseline in BASELINE_RESULTS:
        results.append(_load_result(baseline["path"]))

    print(f"\n=== Running {NEW_VARIANT['name']} ===", flush=True)
    results.append(run_new_variant(out_dir, args))

    payload = build_summary(results, out_dir, args)
    write_markdown(payload, out_dir)
    print(f"\nWrote summary to {out_dir / 'bilinear_m1_combo_summary.md'}", flush=True)
    print(f"Winner: {payload['winner']['variant']} RMSE={payload['winner']['rmse_mean']:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
