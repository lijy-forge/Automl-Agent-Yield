#!/usr/bin/env python3
"""Neural gate architecture ablation at the original z-representation fusion layer.

This keeps the classmate model's original fusion position:

    z_condition, z_composition, z_history -> gate -> fused z -> raw_m1 -> physics layer

Only the neural network used to produce the three gate weights changes.
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


VARIANTS = [
    {
        "name": "original_mlp_gate",
        "root": "agent_workspace/external_baselines/gate_original_learned",
        "description": "Original feature-level MLP gate: concat z vectors -> MLP -> 3 weights.",
    },
    {
        "name": "branchwise_neural_gate",
        "root": "agent_workspace/external_baselines/gate_branchwise_neural",
        "description": "Each branch z vector gets its own small neural scorer, then softmax gives 3 weights.",
    },
    {
        "name": "cross_attention_gate",
        "root": "agent_workspace/external_baselines/gate_cross_attention",
        "description": "Treat the three branch z vectors as tokens and use one tiny attention layer to score them.",
    },
    {
        "name": "bilinear_interaction_gate",
        "root": "agent_workspace/external_baselines/gate_bilinear_interaction",
        "description": "Use low-rank neural pair interactions among branch z vectors before scoring each branch.",
    },
    {
        "name": "residual_neural_gate",
        "root": "agent_workspace/external_baselines/gate_residual_neural",
        "description": "Start near uniform gate and let a bounded neural correction adjust the three weights.",
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


def run_variant(variant: dict[str, str], out_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    variant_out = out_dir / variant["name"]
    result_path = variant_out / f"{variant['name']}_result.json"
    if result_path.exists() and not args.force:
        return json.loads(result_path.read_text(encoding="utf-8"))

    cmd = [
        sys.executable,
        "run_yield_pinn_branch_ablation_experiment.py",
        "--variant-root",
        variant["root"],
        "--variant-name",
        variant["name"],
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
    return json.loads(result_path.read_text(encoding="utf-8"))


def build_summary(results: list[dict[str, Any]], out_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    original = next(row for row in results if row["variant"] == "original_mlp_gate")
    original_by_seed = {int(row["seed"]): row for row in original["per_seed"]}
    enriched = []

    for result in results:
        r2 = _aggregate_r2(result["per_seed"])
        better = 0
        deltas = []
        for row in result["per_seed"]:
            seed = int(row["seed"])
            if seed in original_by_seed:
                delta = float(row["rmse"]) - float(original_by_seed[seed]["rmse"])
                deltas.append(delta)
                if delta < 0:
                    better += 1
        enriched.append(
            {
                **result,
                "r2_mean": r2["mean"],
                "r2_std": r2["std"],
                "rmse_delta_vs_original_mean": float(np.mean(deltas)) if deltas else 0.0,
                "seeds_better_than_original": better if result["variant"] != "original_mlp_gate" else None,
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
            "fixed_fusion_position": "z-representation feature-level fusion",
            "only_changed_component": "neural architecture that produces three gate weights",
        },
        "variants": VARIANTS,
        "ranked": ranked,
        "winner": ranked[0],
    }
    (out_dir / "neural_gate_ablation_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def write_markdown(payload: dict[str, Any], out_dir: Path) -> None:
    protocol = payload["protocol"]
    lines = [
        "# Neural Gate Architecture Ablation Summary",
        "",
        "## Protocol",
        "",
        f"- Data: `{protocol['data']}`",
        f"- HF test: `{protocol['hf_test']}`",
        f"- Seeds: `{protocol['seeds']}`",
        f"- Epoch budget: LF `{protocol['lf_max_epochs']}`, HF `{protocol['hf_max_epochs']}`",
        f"- Fixed fusion position: `{protocol['fixed_fusion_position']}`",
        f"- Controlled variable: `{protocol['only_changed_component']}`",
        "",
        "## Variant Meaning",
        "",
    ]
    for variant in payload["variants"]:
        lines.append(f"- `{variant['name']}`: {variant['description']}")

    lines.extend(
        [
            "",
            "## Results",
            "",
            "| Rank | Variant | RMSE mean | RMSE std | MAE mean | R2 mean | Ensemble RMSE | Ensemble R2 | Delta vs original | Better seeds |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for idx, row in enumerate(payload["ranked"], start=1):
        better = "NA" if row["seeds_better_than_original"] is None else str(row["seeds_better_than_original"])
        lines.append(
            f"| {idx} | `{row['variant']}` | {_fmt(row['rmse_mean'])} | {_fmt(row['rmse_std'])} | "
            f"{_fmt(row['mae_mean'])} | {_fmt(row['r2_mean'])} | {_fmt(row['seed_ensemble_rmse'])} | "
            f"{_fmt(row['seed_ensemble_r2'])} | {_fmt(row['rmse_delta_vs_original_mean'])} | {better} |"
        )

    winner = payload["winner"]
    lines.extend(
        [
            "",
            "## Readout",
            "",
            f"- Winner by 10-seed mean RMSE: `{winner['variant']}`.",
            "- Negative `Delta vs original` means the neural gate improves over the original MLP gate.",
            "- `Better seeds` counts how many matched seeds beat the original MLP gate on RMSE.",
            "- This run keeps the original z-level fusion position, so it isolates gate-network structure rather than m1-level fusion.",
        ]
    )
    (out_dir / "neural_gate_ablation_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=f"agent_workspace/runs/yield_pinn_neural_gate_ablation_{datetime.now().strftime('%Y%m%d_%H%M')}")
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
    for variant in VARIANTS:
        print(f"\n=== Running {variant['name']} ===", flush=True)
        results.append(run_variant(variant, out_dir, args))

    payload = build_summary(results, out_dir, args)
    write_markdown(payload, out_dir)
    print(f"\nWrote summary to {out_dir / 'neural_gate_ablation_summary.md'}", flush=True)
    print(f"Winner: {payload['winner']['variant']} RMSE={payload['winner']['rmse_mean']:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
