#!/usr/bin/env python3
"""Compare the current yield AutoML runs with an expert PINN baseline."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from knowledge.yield_expert_pinn_baseline import evaluate_expert_pinn_oof


DEFAULT_DATA_PATH = "agent_workspace/data/generated_yield_stress/generated_yield_stress_data_engineered.csv"
DEFAULT_AUTOML_RUN_DIRS = [
    "agent_workspace/runs/yield_multiseed_normal_seed0_20260720",
    "agent_workspace/runs/yield_multiseed_normal_seed1_20260720",
    "agent_workspace/runs/yield_multiseed_normal_seed2_20260720",
]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_automl_row(run_dir: Path) -> dict[str, Any]:
    report_path = run_dir / "metrics" / "free_search_report.json"
    if not report_path.exists():
        return {
            "run_dir": str(run_dir),
            "status": "missing_report",
            "error": f"missing {report_path}",
        }
    report = _read_json(report_path)
    champion = report.get("champion") or {}
    metrics = champion.get("oof_metrics") or {}
    spec = champion.get("spec") or {}
    return {
        "run_dir": str(run_dir),
        "status": "ok",
        "seed": int(report.get("random_state", len(str(run_dir)))),
        "champion": champion.get("name"),
        "champion_rmse": float(metrics.get("rmse")),
        "champion_mae": float(metrics.get("mae")),
        "champion_r2": float(metrics.get("r2")),
        "champion_mape": float(metrics.get("mape")),
        "baseline_rmse": report.get("best_fixed_baseline_oof_rmse"),
        "pool_basis": (report.get("champion_selection") or {}).get("pool_basis"),
        "model_origin": spec.get("model_origin"),
        "model_family": spec.get("model_family"),
        "mechanism": spec.get("mechanism"),
        "fusion_mode": spec.get("fusion_mode"),
    }


def load_automl_rows(run_dirs: list[str]) -> list[dict[str, Any]]:
    rows = [_extract_automl_row(Path(item)) for item in run_dirs]
    return sorted(rows, key=lambda row: int(row.get("seed", 10**9)))


def _aggregate_automl(rows: list[dict[str, Any]], *, seeds: set[int] | None = None) -> dict[str, Any]:
    ok = [row for row in rows if row.get("status") == "ok"]
    if seeds is not None:
        ok = [row for row in ok if int(row.get("seed", -1)) in seeds]
    rmse = [float(row["champion_rmse"]) for row in ok if row.get("champion_rmse") is not None]
    baseline = [float(row["baseline_rmse"]) for row in ok if row.get("baseline_rmse") is not None]
    return {
        "n_ok": len(ok),
        "seeds": sorted(seeds) if seeds is not None else sorted(int(row.get("seed", -1)) for row in ok),
        "rmse_mean": float(np.mean(rmse)) if rmse else None,
        "rmse_std": float(np.std(rmse)) if rmse else None,
        "baseline_rmse_mean": float(np.mean(baseline)) if baseline else None,
        "baseline_rmse_std": float(np.std(baseline)) if baseline else None,
    }


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def _write_markdown(out_dir: Path, payload: dict[str, Any]) -> Path:
    automl_rows = payload["automl"]["rows"]
    aggregate_seeds = set(payload["automl"]["aggregate"].get("seeds") or [])
    pinn_rows = payload["expert_pinn"]["seed_results"]
    pinn_by_seed = {int(row["seed"]): row for row in pinn_rows}

    lines: list[str] = []
    lines.append("# AutoML vs Expert PINN Baseline Comparison")
    lines.append("")
    lines.append(f"Generated at: `{payload['generated_at']}`")
    lines.append("")
    lines.append("## Protocol")
    lines.append("")
    protocol = payload["expert_pinn"]["protocol"]
    profile = payload["expert_pinn"]["data_profile"]
    lines.append(f"- Data: `{payload['data_path']}`")
    lines.append(f"- Samples/features: {profile['n_samples']} / {profile['n_features']} for PINN")
    lines.append(f"- CV: {protocol['n_splits']}-fold KFold, seeds={protocol['seeds']}")
    lines.append("- PINN early stopping uses an inner train-fold split; outer OOF validation targets are not used for early stopping.")
    lines.append(f"- `phi` range/std: {_fmt(profile['phi_min'])} to {_fmt(profile['phi_max'])}, std={_fmt(profile['phi_std'], 6)}")
    lines.append("")
    lines.append("## Seed-Level Results")
    lines.append("")
    lines.append("| Seed | AutoML Champion | AutoML RMSE | Expert PINN RMSE | Delta PINN-AutoML | AutoML Model | AutoML Fusion |")
    lines.append("|---:|---|---:|---:|---:|---|---|")
    for row in automl_rows:
        if row.get("status") != "ok":
            lines.append(f"| NA | `{row.get('run_dir')}` | NA | NA | NA | NA | NA |")
            continue
        seed = int(row["seed"])
        pinn = pinn_by_seed.get(seed)
        pinn_rmse = (pinn or {}).get("metrics", {}).get("rmse")
        delta = None if pinn_rmse is None else float(pinn_rmse) - float(row["champion_rmse"])
        lines.append(
            "| "
            + " | ".join(
                [
                    str(seed),
                    f"`{row.get('champion')}`",
                    _fmt(row.get("champion_rmse")),
                    _fmt(pinn_rmse),
                    _fmt(delta),
                    str(row.get("model_family")),
                    str(row.get("fusion_mode")),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## Aggregate")
    lines.append("")
    lines.append(f"Aggregate is computed on matched seeds: `{sorted(aggregate_seeds)}`.")
    lines.append("")
    auto_agg = payload["automl"]["aggregate"]
    pinn_agg = payload["expert_pinn"]["aggregate"]
    delta_mean = (
        float(pinn_agg["rmse_mean"]) - float(auto_agg["rmse_mean"])
        if pinn_agg.get("rmse_mean") is not None and auto_agg.get("rmse_mean") is not None
        else None
    )
    lines.append("| Method | RMSE Mean | RMSE Std | MAE Mean | R2 Mean |")
    lines.append("|---|---:|---:|---:|---:|")
    lines.append(
        f"| AutoML champion | {_fmt(auto_agg.get('rmse_mean'))} | {_fmt(auto_agg.get('rmse_std'))} | NA | NA |"
    )
    lines.append(
        f"| Expert PINN | {_fmt(pinn_agg.get('rmse_mean'))} | {_fmt(pinn_agg.get('rmse_std'))} | "
        f"{_fmt(pinn_agg.get('mae_mean'))} | {_fmt(pinn_agg.get('r2_mean'))} |"
    )
    lines.append("")
    lines.append(f"Mean RMSE delta (PINN - AutoML): `{_fmt(delta_mean)}`.")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.extend(payload["interpretation"])
    lines.append("")

    path = out_dir / "automl_vs_expert_pinn_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def build_interpretation(payload: dict[str, Any]) -> list[str]:
    auto = payload["automl"]["aggregate"]
    pinn = payload["expert_pinn"]["aggregate"]
    lines: list[str] = []
    if auto.get("rmse_mean") is not None and pinn.get("rmse_mean") is not None:
        delta = float(pinn["rmse_mean"]) - float(auto["rmse_mean"])
        if delta > 0:
            lines.append(
                f"- Under this same OOF protocol, AutoML is better on mean RMSE by {_fmt(delta)}."
            )
        elif delta < 0:
            lines.append(
                f"- Under this same OOF protocol, the expert PINN is better on mean RMSE by {_fmt(abs(delta))}."
            )
        else:
            lines.append("- Under this same OOF protocol, AutoML and expert PINN tie on mean RMSE.")
    lines.append(
        "- This is a controlled baseline comparison, not proof of final external generalization; the current generated dataset has constant phi and no real holdout anchor."
    )
    lines.append(
        "- The expert PINN uses a strong hand-designed three-branch physical architecture; AutoML uses generated/searched mechanisms, models, and fusion modes."
    )
    lines.append(
        "- If this baseline is kept, it should be reported as an expert fixed baseline rather than as an AutoML-generated candidate."
    )
    return lines


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DEFAULT_DATA_PATH)
    parser.add_argument("--automl-run-dir", action="append", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--max-epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else Path("agent_workspace/runs") / (
        "yield_expert_pinn_compare_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    automl_run_dirs = args.automl_run_dir or DEFAULT_AUTOML_RUN_DIRS
    automl_rows = load_automl_rows(automl_run_dirs)

    pinn = evaluate_expert_pinn_oof(
        args.data,
        seeds=args.seeds,
        n_splits=args.n_splits,
        hidden_dim=args.hidden_dim,
        max_epochs=args.max_epochs,
        patience=args.patience,
        lr=args.lr,
        out_dir=out_dir,
    )

    matched_seeds = {int(row["seed"]) for row in pinn["seed_results"]}
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "data_path": args.data,
        "automl": {
            "rows": automl_rows,
            "aggregate": _aggregate_automl(automl_rows, seeds=matched_seeds),
        },
        "expert_pinn": pinn,
    }
    payload["interpretation"] = build_interpretation(payload)

    (out_dir / "automl_vs_expert_pinn_report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    md_path = _write_markdown(out_dir, payload)
    print(f"Wrote comparison JSON: {out_dir / 'automl_vs_expert_pinn_report.json'}")
    print(f"Wrote comparison Markdown: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
