#!/usr/bin/env python3
"""Process-history branch ablation on the classmate model (torch, amla env).

Retrains ONE variant over the provided split + N seeds and reports HF-test metrics.
Run once per variant (original / variant2_simplify / variant3_dropbranch) so each
variant's `multi_fidelity` package is imported in its own process (no collisions).

Variants (only the MODEL file differs; training protocol identical):
  - original          : 3 branches, history = 22 raw stages + 11 aggregates
  - variant2_simplify : history = 11 aggregates only (drop raw stage sequence)
  - variant3_dropbranch: 2 branches (thermal+composition); history removed from
                         the prediction/fusion path (true 2-branch, not zh=0)
"""
from __future__ import annotations
import argparse, importlib.util, json, subprocess, sys
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from run_yield_pinn_residual_compare import load_pinn_frame

DEFAULT_LF = Path("/Users/lijiayao/Downloads/yield_predict_180lf_20hf_real_lf(1)/low_fidelity")
DEFAULT_HF = Path("/Users/lijiayao/Downloads/yield_predict_180lf_20hf_real_lf(1)/high_fidelity")

_rmse = lambda a, b: float(np.sqrt(mean_squared_error(a, b)))


def import_variant_run_module(variant_root: Path):
    script = variant_root / "multi_fidelity" / "src" / "experiments" / "run_yield_predict_experiment.py"
    root_text = str(variant_root.resolve())
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    spec = importlib.util.spec_from_file_location(f"variant_run_{variant_root.name}", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, script


def train_seed(script: Path, variant_root: Path, out_dir: Path, seed: int, args) -> tuple[Path, float]:
    results = out_dir / f"seed{seed}" / "results"
    models = out_dir / f"seed{seed}" / "models"
    cmd = [sys.executable, str(script.resolve()),
           "--lf-data", str(args.lf_dir), "--hf-data", str(args.hf_dir),
           "--out-dir", str(results), "--models-dir", str(models),
           "--phi-value", "0.4", "--hidden-dim", "16", "--seed", str(seed),
           "--lf-max-epochs", str(args.lf_max_epochs), "--hf-max-epochs", str(args.hf_max_epochs),
           "--lf-patience", str(args.lf_patience), "--hf-patience", str(args.hf_patience)]
    import os
    full_env = {**os.environ, "PYTHONPATH": str(variant_root.resolve()),
                "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                "MPLCONFIGDIR": "/tmp/matplotlib-yield"}
    ckpt = models / "multifidelity.pth"
    last = None
    for attempt in range(4):  # intermittent torch subprocess-spawn flakiness -> retry
        proc = subprocess.run(cmd, capture_output=True, text=True, env=full_env)
        if proc.returncode == 0 and ckpt.exists():
            break
        last = (proc.returncode, proc.stdout[-1500:], proc.stderr[-1500:])
    else:
        raise RuntimeError(f"seed {seed} training failed after retries rc={last[0]}\n"
                           f"--- STDOUT ---\n{last[1]}\n--- STDERR ---\n{last[2]}")
    pressure_ref = float(json.loads((results / "summary.json").read_text())["pressure_ref_kpa"])
    return ckpt, pressure_ref


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant-root", required=True)
    ap.add_argument("--variant-name", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--lf-dir", default=str(DEFAULT_LF))
    ap.add_argument("--hf-dir", default=str(DEFAULT_HF))
    ap.add_argument("--lf-max-epochs", type=int, default=400)
    ap.add_argument("--hf-max-epochs", type=int, default=800)
    ap.add_argument("--lf-patience", type=int, default=60)
    ap.add_argument("--hf-patience", type=int, default=150)
    args = ap.parse_args()
    args.lf_dir = Path(args.lf_dir).resolve()
    args.hf_dir = Path(args.hf_dir).resolve()

    variant_root = Path(args.variant_root)
    out_dir = Path(args.out_dir).resolve()  # absolute: subprocess cwd may differ
    out_dir.mkdir(parents=True, exist_ok=True)
    run_module, script = import_variant_run_module(variant_root)

    per_seed = []
    all_pred = None
    y_ref = None
    for seed in args.seeds:
        ckpt, pref = train_seed(script, variant_root, out_dir, seed, args)
        te = load_pinn_frame(run_module, ckpt, args.hf_dir / "test.csv", pressure_ref_kpa=pref, phi_value=0.4)
        y, pred = te["y"], te["pinn_pred"]
        per_seed.append({"seed": seed, "rmse": _rmse(y, pred),
                         "mae": float(mean_absolute_error(y, pred)), "r2": float(r2_score(y, pred))})
        all_pred = pred if all_pred is None else np.vstack([all_pred, pred])
        y_ref = y
        print(f"[{args.variant_name}] seed {seed}: RMSE={per_seed[-1]['rmse']:.4f} R2={per_seed[-1]['r2']:.4f}", flush=True)

    rmses = [r["rmse"] for r in per_seed]
    maes = [r["mae"] for r in per_seed]
    ens = all_pred.mean(axis=0) if all_pred.ndim > 1 else all_pred
    payload = {
        "variant": args.variant_name, "n_seeds": len(args.seeds),
        "per_seed": per_seed,
        "rmse_mean": float(np.mean(rmses)), "rmse_std": float(np.std(rmses)),
        "mae_mean": float(np.mean(maes)), "mae_std": float(np.std(maes)),
        "seed_ensemble_rmse": _rmse(y_ref, ens), "seed_ensemble_r2": float(r2_score(y_ref, ens)),
        "epochs": {"lf": args.lf_max_epochs, "hf": args.hf_max_epochs},
        "data_dirs": {"low_fidelity": str(args.lf_dir), "high_fidelity": str(args.hf_dir)},
    }
    (out_dir / f"{args.variant_name}_result.json").write_text(json.dumps(payload, indent=2))
    print(f"\n[{args.variant_name}] RMSE {payload['rmse_mean']:.4f} ± {payload['rmse_std']:.4f} | "
          f"ensemble {payload['seed_ensemble_rmse']:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
