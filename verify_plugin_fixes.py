"""Verify the plugin-harness P0/P1 review fixes — NO LLM / NO web search needed.

Run from the repo root:
    PYTHONPATH=. python3 verify_plugin_fixes.py

Part 1 exercises the deterministic harness directly and asserts the four fixes:
  #1 same-model mechanism ablation
  #2 benchmark-selected proxy re-run on FULL data + honest alignment
  #3 spec-vs-implementation feature-declaration audit
  #4 tightened research_confidence gate
Part 2 prints the honest result table of the most recent full run (yield_plugin_s*),
so you can eyeball the scientific verdict yourself.
"""
import json
import shutil
from pathlib import Path

from knowledge.yield_candidate_benchmark import execute_plugin_candidate_artifacts
from operation_agent.yield_plugin_contract import validate_plugin_feature_declaration

DATA = "agent_workspace/data/yield_synthetic/synthetic_yield_v1.csv"
ANCHOR = "agent_workspace/data/yield_synthetic/real_table6_anchor.csv"
OUT = Path("agent_workspace/runs/_verify_tmp")

ok = True
def check(label, cond, extra=""):
    global ok
    ok = ok and bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}" + (f"   ({extra})" if extra else ""))

# Benchmark report whose SELECTED proxy is ridge+packing_sp (mimics a real run).
BR = {
    "selected_strategy_id": "cand_ridge",
    "selected_strategy_benchmark": {
        "strategy_id": "cand_ridge", "proxy_model": "ridge", "feature_mode": "packing_sp",
        "evaluated_base_feature_columns": [],
        "oof_metrics": {"rmse": 0.097, "r2": 0.95}, "anchor_validation": {"r2": 0.69},
    },
    "benchmark_protocol": {"random_state": 42},
}
# A valid RF plugin: declares raw+packing, builds only packing features.
RF_PLUGIN = '''
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
CANDIDATE_SPEC = {"candidate_id":"rf_packing_only","feature_families":["raw","packing"],
    "physics_hparams":{"phi_max_policy":"fold_local"},"model_family":"rf","constraints":["nonnegative_prediction"]}
def add_features(df, fit_context):
    out = df.copy(); state = {} if fit_context is None else fit_context
    phi = df["phi"].astype(float).values; gap = np.maximum(0.62 - phi, 1e-6)
    out["bench_packing_gap"] = gap; out["bench_phi_over_gap"] = phi / gap
    return out, state
def make_model(random_state):
    return RandomForestRegressor(n_estimators=120, random_state=random_state, n_jobs=-1)
'''

print("PART 1 — deterministic harness fixes (no LLM):")
shutil.rmtree(OUT, ignore_errors=True); OUT.mkdir(parents=True)
res = execute_plugin_candidate_artifacts(DATA, {}, BR, OUT, RF_PLUGIN, anchor_path=ANCHOR, n_splits=5, source_origin="llm")
m = json.loads((OUT / "metrics/metrics.json").read_text())
ab = m.get("mechanism_ablation") or {}
bsp = m.get("benchmark_selected_proxy_result") or {}
ba = m.get("benchmark_alignment") or {}
iv = m.get("implementation_validation") or {}
cg = iv.get("confidence_gate") or {}

check("#1 ablation uses SAME model for with/without", ab.get("same_model_protocol") is True
      and ab.get("without_mechanism_model") == ab.get("with_mechanism_model"),
      f"{ab.get('without_mechanism_model')} vs {ab.get('with_mechanism_model')}")
check("#2 benchmark proxy re-run on FULL data (not cached subsample)",
      bsp.get("source") == "candidate_benchmark_selected_proxy_reevaluated_full_data")
check("#2 real proxy identity preserved (ridge+packing_sp)",
      bsp.get("proxy_model") == "ridge" and bsp.get("feature_mode") == "packing_sp")
check("#2 alignment honestly flags plugin != benchmark proxy",
      ba.get("benchmark_proxy_model") == "ridge" and ba.get("implemented_model") == "rf"
      and ba.get("implemented_matches_benchmark") is False)
check("#4 confidence gate is recorded + not over-optimistic",
      isinstance(cg, dict) and cg and m.get("research_confidence") != "high",
      f"confidence={m.get('research_confidence')}")
# #3 declaration audit
check("#3 flags features built but NOT declared",
      any("sp_saturation" in r for r in validate_plugin_feature_declaration(
          'out["bench_sp_decay"]=1; out["bench_sp_centered"]=2', {"feature_families": ["raw", "packing"]})))
check("#3 no false positive for a raw sp_percent read",
      validate_plugin_feature_declaration('x = df["sp_percent"].mean()', {"feature_families": ["raw"]}) == [])
shutil.rmtree(OUT, ignore_errors=True)

print(f"\nPART 1 RESULT: {'ALL PASS' if ok else 'SOME FAILED'}")

# ---- Part 2: eyeball the most recent real run ----
print("\nPART 2 — honest verdict of the most recent full run:")
runs = sorted(Path("agent_workspace/runs").glob("yield_plugin_s*/metrics/metrics.json"),
              key=lambda p: p.stat().st_mtime, reverse=True)
if not runs:
    print("  (no yield_plugin_s* run found — run the full pipeline to populate one)")
else:
    mp = runs[0]
    run_root = mp.parent.parent
    # Per-round outcomes (the manager revision loop runs operation several times;
    # a round falls back to the reference plugin when the LLM was unavailable).
    rounds = sorted(run_root.glob("run_result_manager_round_*.json"))
    if rounds:
        print(f"  per-round outcomes in {run_root.name}:")
        for rf in rounds:
            d = json.loads(rf.read_text())
            print(f"    {rf.name}: candidate_source={d.get('candidate_source')}  fallback={d.get('operation_fallback')}  rcode={d.get('rcode')}")
        print("  (rounds tagged plugin_reference_fallback = LLM plugin unavailable that round;")
        print("   the harness ran the deterministic reference plugin instead — rcode still 0.)")
    m = json.loads(mp.read_text())
    def f(x): return f"{x:.4f}" if isinstance(x, (int, float)) else "n/a"
    o = m.get("oof_metrics") or {}; a = m.get("anchor_validation") or {}
    ab = m.get("mechanism_ablation") or {}; bsp = m.get("benchmark_selected_proxy_result") or {}
    arms = m.get("reference_arms") or {}
    print(f"  run: {mp.parent.parent.name}")
    print(f"  candidate_source={m.get('candidate_source')}  operation_fallback={m.get('operation_fallback')}")
    print(f"  LLM plugin      OOF_RMSE={f(o.get('rmse'))}  ANCHOR_R2={f(a.get('r2'))}")
    print(f"  same-model abl  delta_rmse={f(ab.get('delta_rmse'))}  mechanism_improves={ab.get('mechanism_improves_metric')}  (same_model={ab.get('same_model_protocol')})")
    print(f"  benchmark proxy {bsp.get('proxy_model')}+{bsp.get('feature_mode')}  OOF_RMSE={f((bsp.get('oof_metrics') or {}).get('rmse'))}  ANCHOR_R2={f((bsp.get('anchor_validation') or {}).get('r2'))}")
    for k, v in arms.items():
        print(f"  ref {k:22s} ANCHOR_R2={f((v.get('anchor_validation') or {}).get('r2'))}")
    print(f"  >>> VERDICT: {m.get('research_model_status')} / confidence={m.get('research_confidence')}")

raise SystemExit(0 if ok else 1)
