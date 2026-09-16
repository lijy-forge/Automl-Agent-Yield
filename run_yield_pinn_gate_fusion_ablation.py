#!/usr/bin/env python3
"""Gate/fusion ablation for the classmate LF/HF yield model.

This runner keeps the original provided LF/HF split and training script fixed.
Only the model fusion head changes across variants, so the comparison answers:

    Which three-branch fusion strategy is most useful for this small dataset?
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
        "name": "original_learned_gate",
        "root": "agent_workspace/external_baselines/gate_original_learned",
        "description": "Original sample-wise learned MLP gate.",
    },
    {
        "name": "global_gate",
        "root": "agent_workspace/external_baselines/gate_global_gate",
        "description": "One learned global three-branch weight vector shared by all samples.",
    },
    {
        "name": "concat_fusion",
        "root": "agent_workspace/external_baselines/gate_concat_fusion",
        "description": "No gate; concatenate thermal/composition/history branch embeddings then predict m1_eff.",
    },
    {
        "name": "m1_level_gate",
        "root": "agent_workspace/external_baselines/gate_m1_level_gate",
        "description": "Each branch predicts raw m1_eff, then a learned gate fuses the three m1_eff estimates.",
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
    original = next(row for row in results if row["variant"] == "original_learned_gate")
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
                "seeds_better_than_original": better if result["variant"] != "original_learned_gate" else None,
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
            "only_changed_component": "GatedFusionHead / branch fusion strategy",
        },
        "variants": VARIANTS,
        "ranked": ranked,
        "winner": ranked[0],
    }
    (out_dir / "gate_fusion_ablation_summary.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    return payload


def write_markdown(payload: dict[str, Any], out_dir: Path) -> None:
    protocol = payload["protocol"]
    lines = [
        "# Gate/Fusion Ablation Summary",
        "",
        "## Protocol",
        "",
        f"- Data: `{protocol['data']}`",
        f"- HF test: `{protocol['hf_test']}`",
        f"- Seeds: `{protocol['seeds']}`",
        f"- Epoch budget: LF `{protocol['lf_max_epochs']}`, HF `{protocol['hf_max_epochs']}`",
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
            "- Negative `Delta vs original` means the variant improves over the original learned gate.",
            "- `Better seeds` counts how many matched seeds beat the original learned gate on RMSE.",
            "- HF test has only a few samples, so seed stability and ensemble RMSE should be read together.",
        ]
    )
    (out_dir / "gate_fusion_ablation_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=f"agent_workspace/runs/yield_pinn_gate_fusion_ablation_{datetime.now().strftime('%Y%m%d_%H%M')}")
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
    print(f"\nWrote summary to {out_dir / 'gate_fusion_ablation_summary.md'}", flush=True)
    print(f"Winner: {payload['winner']['variant']} RMSE={payload['winner']['rmse_mean']:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
