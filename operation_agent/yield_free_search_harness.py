"""Artifact writer + light acceptance for the direction-④ free-search mode.

Following the Step-4 precedent (the plugin mode got its OWN light verify rather
than reusing the freeform verifier), the free-search paradigm gets its own thin
artifact writer and acceptance check instead of forcing the plugin verifier onto
a differently-shaped result. The deterministic search/scoring lives in
knowledge/yield_joint_search.py; this module only persists the champion and
audits that the run is fresh, honest, and reproducible.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from knowledge.yield_schema import INDEX_COLUMN, TARGET_COLUMN, load_yield_dataframe

# keyword -> canonical model family understood by _make_model in yield_joint_search.
# `pinn` is checked FIRST because physics-informed-NN descriptions also contain
# generic "neural"/"network" tokens that would otherwise map to plain `mlp`.
_FAMILY_KEYWORDS = (
    ("branched_hidden16", ("branched_hidden16", "branched hidden16", "three-branch", "three branch", "grouped branch", "branched mlp")),
    ("mlp3_hidden16", ("mlp3_hidden16", "mlp3 hidden16", "three-layer mlp", "3-layer mlp", "16 16 8")),
    ("mlp2_hidden64", ("mlp2_hidden64", "mlp2 hidden64", "hidden=64", "64 -> 64")),
    ("mlp2_hidden16", ("mlp2_hidden16", "mlp2 hidden16", "hidden=16", "16 -> 16")),
    ("pinn", ("pinn", "physics-informed", "physics informed", "physics-informed neural",
              "physics neural", "physics residual", "physics-residual", "physics regular",
              "physically informed", "physics constrained", "physics-constrained")),
    ("extratrees", ("extra tree", "extratrees", "extra-trees", "extremely randomized")),
    ("rf", ("random forest", "randomforest", "forest", " rf", "tree ensemble", "bagging")),
    ("hgb", ("hist", "gradient boost", "gradientboost", "hgb", "boosted", "gbm", "boosting",
             "xgb", "xgboost", "extreme gradient", "lightgbm", "catboost")),
    ("svr", ("svr", "support vector", "svm")),
    ("ridge", ("ridge", "linear", "lasso", "elasticnet", "ols", "regularized linear")),
    ("kernel", ("kernel", "gaussian process", "gpr", "rbf", "matern")),
    ("mlp", ("mlp", "neural", "perceptron", "deep", "network")),
)
_DEFAULT_FAMILIES = ("rf", "hgb", "ridge")
# strong families the fixed baseline always uses; ALWAYS included as a floor so a
# round whose LLM model proposals omit them cannot lose to the baseline by omission.
_FLOOR_FAMILIES = ("rf", "hgb", "ridge")
_LATENT_ARCH_FAMILIES = ("mlp2_hidden16", "mlp2_hidden64", "mlp3_hidden16", "branched_hidden16")
# advanced models ALWAYS offered to the search (as fixed arms) so complex/physics-
# informed candidates are exercised every run, not only when the LLM happens to
# propose them. Without this, a round that proposes only rf/hgb/mlp/ridge never
# evaluates PINN.
_ALWAYS_FAMILIES = ("pinn", *_LATENT_ARCH_FAMILIES)


def _budget_mode() -> str:
    mode = str(os.environ.get("YIELD_SEARCH_BUDGET") or "normal").strip().lower()
    return mode if mode in {"quick", "normal", "full", "unlimited"} else "normal"


def _always_families_for_budget() -> tuple[str, ...]:
    """Representative advanced families by runtime budget.

    quick/normal keep the route visible without running every neural variant;
    full keeps the complete A/B/C/D comparison.
    """
    mode = _budget_mode()
    if mode == "quick":
        return ("mlp2_hidden16",)
    if mode == "normal":
        return ("pinn", "mlp2_hidden16", "branched_hidden16")
    return _ALWAYS_FAMILIES


def family_from_model_spec(spec: dict[str, Any]) -> str:
    """Map a searched candidate_model dict onto a canonical family string."""
    text = " ".join(
        str(spec.get(k, "")) for k in ("family", "name", "modeling_idea", "id")
    ).lower()
    for fam, kws in _FAMILY_KEYWORDS:
        if any(kw in text for kw in kws):
            return fam
    return "hgb"


def model_families_from_candidates(candidate_models: list[dict[str, Any]]) -> list[str]:
    """Deduped canonical families from the searched candidate_models, UNION the
    baseline floor families (rf/hgb). The floor guarantees the free search always
    contains at least as strong a model as the fixed baseline, so it can never
    lose to the baseline merely because a round's LLM proposals omitted rf/hgb."""
    fams: list[str] = []
    for spec in candidate_models or []:
        fam = family_from_model_spec(spec if isinstance(spec, dict) else {})
        if fam not in fams:
            fams.append(fam)
    if not fams:
        fams = list(_DEFAULT_FAMILIES)
    for extra in (*_FLOOR_FAMILIES, *_always_families_for_budget()):
        if extra not in fams:
            fams.append(extra)
    return fams


def _json_clean(obj: Any) -> Any:
    """Drop private numpy-array keys (_oof_pred/_anchor_pred/_champion_row) and make JSON-safe."""
    if isinstance(obj, dict):
        return {k: _json_clean(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, (list, tuple)):
        return [_json_clean(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _write_predict_script(run_dir: Path, feature_columns: list[str]) -> Path:
    """Write a small standalone prediction entrypoint for the saved champion."""
    repo_root = Path(__file__).resolve().parents[1]
    script_path = run_dir / "predict.py"
    script = f'''#!/usr/bin/env python3
"""Predict yield_stress with this run's saved champion model.

Usage:
  python predict.py --input new_samples.csv --output predictions.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


FEATURE_COLUMNS = {json.dumps([str(c) for c in feature_columns], ensure_ascii=False, indent=2)}
MODEL_PATH = Path(__file__).resolve().parent / "trained_models" / "champion.joblib"
REPO_ROOT = Path({str(repo_root)!r})


def _ensure_repo_path() -> None:
    candidates = [REPO_ROOT, *Path(__file__).resolve().parents]
    for root in candidates:
        if (root / "knowledge" / "yield_schema.py").exists():
            text = str(root)
            if text not in sys.path:
                sys.path.insert(0, text)
            return


def _read_table(path: Path):
    import pandas as pd

    if path.suffix.lower() in {{".xlsx", ".xls"}}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def _feature_frame(df):
    import pandas as pd

    X = df.copy()
    for col in FEATURE_COLUMNS:
        if col not in X.columns:
            X[col] = 0.0
    return X.reindex(columns=FEATURE_COLUMNS).apply(pd.to_numeric, errors="coerce").fillna(0.0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input CSV/XLSX with the model feature columns.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    args = parser.parse_args()

    _ensure_repo_path()
    import joblib

    input_path = Path(args.input)
    output_path = Path(args.output)
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Saved champion model not found: {{MODEL_PATH}}")

    df = _read_table(input_path)
    model = joblib.load(MODEL_PATH)
    pred = model.predict(_feature_frame(df))

    out = df.copy()
    out["yield_stress_predicted"] = pred
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"wrote {{len(out)}} predictions to {{output_path}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''
    script_path.write_text(script, encoding="utf-8")
    try:
        script_path.chmod(0o755)
    except Exception:
        pass
    return script_path


def _row_rmse(row: dict[str, Any] | None) -> float | None:
    if not isinstance(row, dict):
        return None
    try:
        value = (row.get("oof_metrics") or {}).get("rmse")
        return None if value is None else float(value)
    except Exception:
        return None


def _row_summary(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(row, dict):
        return None
    spec = row.get("spec") or {}
    return {
        "name": row.get("name"),
        "model_family": spec.get("model_family"),
        "model_origin": spec.get("model_origin"),
        "mechanism": spec.get("mechanism"),
        "fusion_mode": spec.get("fusion_mode"),
        "fusion_id": spec.get("fusion_id"),
        "hparams": spec.get("hparams"),
        "oof_rmse": _row_rmse(row),
        "oof_r2": (row.get("oof_metrics") or {}).get("r2"),
        "baseline_status": row.get("baseline_status"),
    }


def build_contribution_diagnostics(search: dict[str, Any]) -> dict[str, Any]:
    """Explain whether the gain came from the model or the mechanism wiring.

    This is diagnostic only: it does not change champion selection.
    """
    ranked = [r for r in (search.get("ranked") or []) if isinstance(r, dict)]
    viable = [r for r in ranked if _row_rmse(r) is not None]
    fixed = [r for r in viable if (r.get("spec") or {}).get("model_origin") == "fixed_family"]
    llm = [r for r in viable if (r.get("spec") or {}).get("model_origin") == "llm_model"]
    best_fixed = min(fixed, key=lambda r: _row_rmse(r) or 1e99, default=None)
    best_llm = min(llm, key=lambda r: _row_rmse(r) or 1e99, default=None)

    llm_by_family: dict[str, list[dict[str, Any]]] = {}
    for row in llm:
        fam = str((row.get("spec") or {}).get("model_family") or "")
        llm_by_family.setdefault(fam, []).append(row)
    best_llm_by_family = []
    for fam, rows in sorted(llm_by_family.items()):
        best = min(rows, key=lambda r: _row_rmse(r) or 1e99)
        best_llm_by_family.append({
            "model_family": fam,
            "n_rows": len(rows),
            "best": _row_summary(best),
        })
    best_llm_by_family.sort(key=lambda x: (x.get("best") or {}).get("oof_rmse") or 1e99)

    grouped: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = {}
    for row in viable:
        spec = row.get("spec") or {}
        hparams = spec.get("hparams") or {}
        try:
            hp_sig = json.dumps(hparams, ensure_ascii=False, sort_keys=True)
        except Exception:
            hp_sig = repr(hparams)
        key = (
            str(spec.get("model_family") or ""),
            str(spec.get("model_origin") or ""),
            hp_sig,
        )
        mode = str(spec.get("fusion_mode") or "unknown")
        current = grouped.setdefault(key, {}).get(mode)
        if current is None or (_row_rmse(row) or 1e99) < (_row_rmse(current) or 1e99):
            grouped[key][mode] = row

    paired_ablation = []
    for (family, origin, hp_sig), by_mode in grouped.items():
        raw = by_mode.get("raw_ml")
        if raw is None:
            continue
        mechanism_rows = [row for mode, row in by_mode.items() if mode != "raw_ml"]
        if not mechanism_rows:
            continue
        best_mech = min(mechanism_rows, key=lambda r: _row_rmse(r) or 1e99)
        raw_rmse = _row_rmse(raw)
        mech_rmse = _row_rmse(best_mech)
        paired_ablation.append({
            "model_family": family,
            "model_origin": origin,
            "hparams_signature": hp_sig,
            "raw": _row_summary(raw),
            "best_mechanism_combo": _row_summary(best_mech),
            "mechanism_gain_rmse": (
                None if raw_rmse is None or mech_rmse is None else float(raw_rmse - mech_rmse)
            ),
        })
    paired_ablation.sort(
        key=lambda x: (
            -float(x.get("mechanism_gain_rmse") or -1e99),
            (x.get("best_mechanism_combo") or {}).get("oof_rmse") or 1e99,
        )
    )
    paired_by_competitive_rmse = sorted(
        paired_ablation,
        key=lambda x: (x.get("best_mechanism_combo") or {}).get("oof_rmse") or 1e99,
    )

    best_fixed_rmse = _row_rmse(best_fixed)
    best_llm_rmse = _row_rmse(best_llm)
    return {
        "purpose": "Separate model contribution from mechanism/fusion contribution using already-ranked OOF rows.",
        "best_fixed": _row_summary(best_fixed),
        "best_llm": _row_summary(best_llm),
        "best_llm_minus_best_fixed_rmse": (
            None if best_fixed_rmse is None or best_llm_rmse is None else float(best_llm_rmse - best_fixed_rmse)
        ),
        "best_llm_by_family": best_llm_by_family,
        "paired_ablation_by_same_model": paired_ablation[:20],
        "paired_ablation_by_competitive_rmse": paired_by_competitive_rmse[:20],
        "interpretation_rules": [
            "If mechanism_gain_rmse is near zero, the mechanism did not materially improve that model.",
            "If best_llm_minus_best_fixed_rmse is positive, generated models are still behind the fixed model floor.",
            "Use holdout/anchor before claiming true generalization.",
        ],
    }


def finalize_and_write(
    run_dir: str | Path,
    free_result: dict[str, Any],
    data_path: str | Path,
    candidate_builder,
    *,
    searched_mechanisms: list[dict[str, Any]] | None = None,
    searched_models: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Refit the champion on full data and persist the standard free-search artifacts.

    Writes:
      metrics/free_search_report.json  — full ranked table + pair menu + guardrail
      metrics/predictions.csv          — champion OOF predictions (+ compat columns)
      logs/mechanism_report.json       — which searched mechanisms ran / won / ablation
      trained_models/champion.joblib   — champion refit on full data
    Returns an operation-result dict (rcode/candidate_source/champion/...).
    """
    run_dir = Path(run_dir)
    (run_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "trained_models").mkdir(parents=True, exist_ok=True)

    search = free_result.get("search") or {}
    champion = free_result.get("champion")
    champ_row = search.get("_champion_row") or {}
    feature_columns = search.get("feature_columns") or []

    df, _meta = load_yield_dataframe(data_path)
    y = pd.to_numeric(df[TARGET_COLUMN], errors="coerce").to_numpy(float)
    eval_protocol = search.get("evaluation_protocol") or {}
    fidelity_column = str(eval_protocol.get("fidelity_column") or "data_fidelity")
    high_mask = None
    if (
        eval_protocol.get("type") == "high_fidelity_oof"
        and fidelity_column in df.columns
    ):
        labels = df[fidelity_column].astype(str).str.strip().str.lower()
        high_mask = labels.isin({"high_fidelity", "hf", "high", "real_hf"}).to_numpy(dtype=bool)

    model_path = run_dir / "trained_models" / "champion.joblib"
    predict_script_path = run_dir / "predict.py"
    refit_ok = False
    if champion is not None:
        try:
            import joblib

            built = candidate_builder(dict(champion.get("spec") or {}))
            if not isinstance(built, list):
                built = [built]
            # the builder may expand a pair into several hparam configs; pick the one
            # matching the champion (by its full name, which encodes the hparam tag).
            match = next((c for c in built if getattr(c, "name", None) == champion.get("name")),
                         built[0] if built else None)
            est = match.factory()
            champ_spec = champion.get("spec") or {}
            champ_mode = str(champ_spec.get("fusion_mode") or "")
            fit_columns = list(feature_columns)
            if champ_mode == "multi_fidelity_base_residual" and fidelity_column in df.columns:
                fit_columns = fit_columns + [fidelity_column]
                fit_df = df[fit_columns].copy()
                fit_y = y
            elif high_mask is not None:
                fit_df = df.loc[high_mask, fit_columns].copy()
                fit_y = y[high_mask]
            else:
                fit_df = df[fit_columns].copy()
                fit_y = y
            est.fit(fit_df, fit_y)
            joblib.dump(est, model_path)
            refit_ok = True
            predict_script_path = _write_predict_script(run_dir, feature_columns)
        except Exception as exc:  # keep artifacts + report even if refit fails
            free_result.setdefault("build_errors", []).append(
                {"stage": "finalize_refit", "error": f"{type(exc).__name__}: {exc}"}
            )

    # predictions.csv from the champion's OOF predictions (already fold-local)
    oof = champ_row.get("_oof_pred")
    oof_indices = champ_row.get("_oof_indices")
    if oof is not None and len(oof) == len(df):
        preds = pd.DataFrame({
            "yield_stress_actual": y,
            "yield_stress_predicted": np.asarray(oof, dtype=float),
            "y_true": y,
            "y_pred": np.asarray(oof, dtype=float),
        })
        if INDEX_COLUMN in df.columns:
            preds.insert(0, INDEX_COLUMN, df[INDEX_COLUMN].to_numpy())
        if fidelity_column in df.columns:
            preds.insert(1 if INDEX_COLUMN in preds.columns else 0, fidelity_column, df[fidelity_column].to_numpy())
        preds["prediction_scope"] = "all_rows_oof"
        preds.to_csv(run_dir / "metrics" / "predictions.csv", index=False)
    elif oof is not None and oof_indices is not None and len(oof) == len(oof_indices):
        idx = np.asarray(oof_indices, dtype=int)
        preds = pd.DataFrame({
            "yield_stress_actual": y[idx],
            "yield_stress_predicted": np.asarray(oof, dtype=float),
            "y_true": y[idx],
            "y_pred": np.asarray(oof, dtype=float),
        })
        if INDEX_COLUMN in df.columns:
            preds.insert(0, INDEX_COLUMN, df.iloc[idx][INDEX_COLUMN].to_numpy())
        if fidelity_column in df.columns:
            insert_at = 1 if INDEX_COLUMN in preds.columns else 0
            preds.insert(insert_at, fidelity_column, df.iloc[idx][fidelity_column].to_numpy())
        preds["source_row_index"] = idx
        preds["prediction_scope"] = "high_fidelity_oof"
        preds.to_csv(run_dir / "metrics" / "predictions.csv", index=False)

    contribution_diagnostics = build_contribution_diagnostics(search)

    report = _json_clean({
        "stage": "free_search",
        "champion": champion,
        "champion_selection": search.get("champion_selection"),
        "research_champion": search.get("research_champion"),
        "champion_pool_basis": search.get("champion_pool_basis"),
        "n_splits": search.get("n_splits") or free_result.get("n_splits"),
        "random_state": search.get("random_state") or free_result.get("random_state"),
        "base_random_state": free_result.get("base_random_state"),
        "n_viable": search.get("n_viable"),
        "best_fixed_baseline_oof_rmse": search.get("best_fixed_baseline_oof_rmse"),
        "baseline_improvement_threshold": search.get("baseline_improvement_threshold"),
        "min_anchor_r2": search.get("min_anchor_r2"),
        "ranked": search.get("ranked"),
        "failed": search.get("failed"),
        "pair_menu": free_result.get("pair_menu"),
        "build_errors": free_result.get("build_errors"),
        "materialization_plan": free_result.get("materialization_plan"),
        "budget": free_result.get("budget"),
        "timing": free_result.get("timing"),
        "joint_search_timing": search.get("timing"),
        "n_built_before_budget": free_result.get("n_built_before_budget"),
        "n_built": free_result.get("n_built"),
        "feature_columns": feature_columns,
        "evaluation_protocol": eval_protocol,
        "searched_models": searched_models or [],
        "contribution_diagnostics": contribution_diagnostics,
    })
    # project-facing recommendation list ("ideas to try" + champion + tradeoffs)
    recommendations = build_recommendations(search)
    budget = free_result.get("budget") or {}
    if budget.get("truncated"):
        recommendations.setdefault("caveats", []).append(
            f"本轮使用 {budget.get('mode')} 预算，候选从 "
            f"{budget.get('n_candidates_after_execution_dedupe', budget.get('n_candidates_after_name_dedupe'))} 截到 "
            f"{budget.get('n_candidates_selected')}；汇报或最终确认建议再跑 full/multi-seed。"
        )
    elif budget.get("duplicate_execution_count"):
        recommendations.setdefault("caveats", []).append(
            f"本轮去掉 {budget.get('duplicate_execution_count')} 个执行等价的重复候选；"
            "这些候选只是名字或来源不同，训练路径相同。"
        )
    report["recommendations"] = recommendations
    (run_dir / "metrics" / "free_search_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_dir / "metrics" / "recommendations.md").write_text(
        render_recommendations_md(recommendations), encoding="utf-8")
    (run_dir / "metrics" / "recommendations.json").write_text(
        json.dumps(recommendations, ensure_ascii=False, indent=2), encoding="utf-8")

    # mechanism_report: raw arm vs the champion's mechanism arm (honest ablation)
    ranked = search.get("ranked") or []
    raw_best = min((r for r in ranked if not (r.get("spec") or {}).get("mechanism")),
                   key=lambda r: (r.get("oof_metrics") or {}).get("rmse", 1e9), default=None)
    champ_mech = (champion or {}).get("spec", {}).get("mechanism") if champion else None
    by_fusion: dict[str, dict[str, Any]] = {}
    for row in ranked:
        spec = row.get("spec") or {}
        mode = str(spec.get("fusion_mode") or "unknown")
        current = by_fusion.get(mode)
        row_rmse = (row.get("oof_metrics") or {}).get("rmse")
        cur_rmse = ((current or {}).get("oof_metrics") or {}).get("rmse") if current else None
        row_rmse = 1e9 if row_rmse is None else float(row_rmse)
        cur_rmse = 1e9 if cur_rmse is None else float(cur_rmse)
        if current is None or row_rmse < cur_rmse:
            by_fusion[mode] = row

    mech_report = _json_clean({
        "searched_mechanisms": searched_mechanisms or [],
        "searched_models": searched_models or [],
        "champion_mechanism": champ_mech,
        "champion_fusion_mode": (champion or {}).get("spec", {}).get("fusion_mode") if champion else None,
        "champion_is_mechanism_arm": bool(champ_mech),
        "raw_arm_best_oof_rmse": (raw_best or {}).get("oof_metrics", {}).get("rmse") if raw_best else None,
        "champion_oof_rmse": (champion or {}).get("oof_metrics", {}).get("rmse") if champion else None,
        "best_by_fusion_mode": {
            mode: {
                "name": row.get("name"),
                "mechanism": (row.get("spec") or {}).get("mechanism"),
                "model_family": (row.get("spec") or {}).get("model_family"),
                "oof_rmse": (row.get("oof_metrics") or {}).get("rmse"),
                "anchor_r2": (row.get("anchor_metrics") or {}).get("r2"),
            }
            for mode, row in by_fusion.items()
        },
        "contribution_diagnostics": contribution_diagnostics,
        "mechanism_vs_raw_note": (
            "champion beats the raw arm" if (champion and raw_best
              and (champion.get("oof_metrics") or {}).get("rmse", 1e9) < (raw_best.get("oof_metrics") or {}).get("rmse", 1e9))
            else "raw arm is at least as good — mechanism adds no OOF gain on this data (honest)"),
    })
    (run_dir / "logs" / "mechanism_report.json").write_text(
        json.dumps(mech_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return {
        "rcode": 0 if champion is not None else 1,
        "candidate_source": "free_search_champion",
        "operation_fallback": False,
        "champion": champion,
        "champion_refit_saved": refit_ok,
        "predict_script": str(predict_script_path) if predict_script_path.exists() else "",
        "free_search_report": report,
        "action_result": (
            f"Free search picked champion {champion['name']} "
            f"(pool={search.get('champion_pool_basis')})." if champion
            else "Free search produced no champion."),
        "error_logs": [],
    }


def build_recommendations(search: dict[str, Any], *, top_n: int = 6) -> dict[str, Any]:
    """Turn the ranked search into a project-facing recommendation list: the
    champion, the best in-distribution pick, the best-generalizing pick, and a
    top-N shortlist each with a plain rationale + caveats. This is the 'ideas to
    try' deliverable — it does NOT replace the champion, it explains the options.
    """
    ranked = search.get("ranked") or []
    base = search.get("best_fixed_baseline_oof_rmse")

    def oof(r):
        return (r.get("oof_metrics") or {}).get("rmse")

    def anchor(r):
        return (r.get("anchor_metrics") or {}).get("r2")

    viable = [r for r in ranked if r.get("viable")]
    beats = [r for r in viable if r.get("beats_baseline")]
    pool = beats or viable or ranked
    best_oof = min(pool, key=lambda r: oof(r) if oof(r) is not None else 1e9) if pool else None
    best_gen = max((r for r in pool if anchor(r) is not None), key=anchor, default=None)
    # the champion is whatever the harness actually crowned (guardrail-aware), matched
    # by name — NOT simply the lowest-OOF row (which is best_in_distribution).
    champ_name = (search.get("champion") or {}).get("name")
    champion = next((r for r in ranked if r.get("name") == champ_name), None) or (pool[0] if pool else None)

    def why(r):
        bits = []
        if best_oof is not None and r["name"] == best_oof["name"]:
            bits.append("内分布(OOF)最优")
        if best_gen is not None and r["name"] == best_gen["name"]:
            bits.append(f"外部泛化(anchor)最优 R²={anchor(r):.3f}")
        if base is not None:
            if r.get("beats_baseline"):
                bits.append(f"显著超固定基线(基线OOF={base:.4f})")
            elif r.get("ties_baseline") or r.get("baseline_status") == "ties":
                bits.append(f"与固定基线打平(基线OOF={base:.4f})")
        mode = r["spec"].get("fusion_mode")
        if mode == "mechanism_features":
            bits.append("融合方式=机理特征拼接")
        elif mode == "mechanism_residual":
            bits.append("融合方式=机理残差修正")
        elif mode == "hidden_parameter_physics":
            bits.append("融合方式=隐变量m1_eff物理层")
        elif mode == "raw_ml":
            bits.append("融合方式=raw ML")
        if r["spec"].get("mechanism"):
            bits.append("机理臂")
        else:
            bits.append("raw(无机理)")
        o, a = oof(r), anchor(r)
        if o is not None and a is not None:
            og = (r.get("oof_metrics") or {}).get("r2")
            if og is not None:
                bits.append(f"泛化gap(OOF-anchor R²)={og - a:+.3f}")
        return "；".join(bits)

    def entry(r):
        return {"combo": r["name"], "mechanism": r["spec"].get("mechanism") or "(raw)",
                "model_family": r["spec"].get("model_family"), "hparams": r["spec"].get("hparams"),
                "fusion_mode": r["spec"].get("fusion_mode"), "fusion_id": r["spec"].get("fusion_id"),
                "oof_rmse": oof(r), "anchor_r2": anchor(r),
                "beats_baseline": r.get("beats_baseline"),
                "ties_baseline": r.get("ties_baseline"),
                "baseline_status": r.get("baseline_status"),
                "baseline_rmse_delta": r.get("baseline_rmse_delta"),
                "why": why(r)}

    research_info = search.get("research_champion")
    research_row = None
    if isinstance(research_info, dict) and research_info.get("name"):
        research_row = next((r for r in ranked if r.get("name") == research_info.get("name")), None)

    return {
        "champion": entry(champion) if champion else None,
        "champion_selection": search.get("champion_selection"),
        "best_in_distribution": entry(best_oof) if best_oof else None,
        "best_generalization": entry(best_gen) if best_gen else None,
        "research_champion": entry(research_row) if research_row else None,
        "shortlist": [entry(r) for r in pool[:top_n]],
        "baseline_oof_rmse": base,
        "baseline_improvement_threshold": search.get("baseline_improvement_threshold"),
        "caveats": [
            "anchor 是校准/一致性 anchor，非盲测——绝对值保守解读，机理 vs raw 的相对 gap 更可信。",
            "composition-only 数据下机理在 OOF 上≈raw；机理价值主要体现在 anchor 泛化。",
            "OOF 最优与泛化最优常不是同一个——按项目更看重内分布还是外推来选。",
        ],
    }


def render_recommendations_md(recs: dict[str, Any]) -> str:
    def line(tag, e):
        if not e:
            return f"- **{tag}**：无\n"
        hp = f" `{e['hparams']}`" if e.get("hparams") else ""
        fm = f"，融合={e.get('fusion_mode')}" if e.get("fusion_mode") else ""
        a = f"{e['anchor_r2']:.3f}" if e.get("anchor_r2") is not None else "n/a"
        return (f"- **{tag}**：`{e['combo']}`{hp}{fm} — OOF={e['oof_rmse']:.4f}, anchorR²={a}\n"
                f"  - 理由：{e['why']}\n")
    md = ["# 机理×模型×融合方式 搜索推荐清单\n",
          f"固定基线 OOF = {recs.get('baseline_oof_rmse')}\n",
          "## 冠军 / 关键推荐\n",
          line("冠军(闭环选出)", recs.get("champion")),
          line("内分布最优", recs.get("best_in_distribution")),
          line("外部泛化最优", recs.get("best_generalization")),
          line("研究冠军", recs.get("research_champion"))]
    selection = recs.get("champion_selection") or {}
    if selection.get("explanation"):
        md.extend([
            "\n## 冠军选择说明\n",
            f"{selection.get('explanation')}\n",
        ])
    md.append("\n## Top 候选（值得一试）\n")
    for i, e in enumerate(recs.get("shortlist") or [], 1):
        hp = f" `{e['hparams']}`" if e.get("hparams") else ""
        fm = f"，融合={e.get('fusion_mode')}" if e.get("fusion_mode") else ""
        a = f"{e['anchor_r2']:.3f}" if e.get("anchor_r2") is not None else "n/a"
        md.append(f"{i}. `{e['combo']}`{hp}{fm} — OOF={e['oof_rmse']:.4f}, anchorR²={a} — {e['why']}\n")
    md.append("\n## 注意事项\n")
    for c in recs.get("caveats") or []:
        md.append(f"- {c}\n")
    return "".join(md)


def verify_free_search_run(run_dir: str | Path, *, started_at: float, anchor_expected: bool) -> dict[str, Any]:
    """Light acceptance: artifacts fresh + a champion exists + guardrail honesty."""
    run_dir = Path(run_dir)
    reasons: list[str] = []
    warnings: list[str] = []

    required = [run_dir / "metrics" / "free_search_report.json",
                run_dir / "metrics" / "predictions.csv",
                run_dir / "trained_models" / "champion.joblib",
                run_dir / "predict.py"]
    for path in required:
        if not path.exists():
            reasons.append(f"missing artifact: {path.name}")
        elif path.stat().st_mtime + 1.0 < started_at:
            reasons.append(f"stale artifact (not written this run): {path.name}")
    if reasons:
        return {"passed": False, "reasons": reasons, "warnings": warnings}

    try:
        report = json.loads((run_dir / "metrics" / "free_search_report.json").read_text(encoding="utf-8"))
    except Exception as exc:
        return {"passed": False, "reasons": [f"free_search_report.json unreadable: {exc}"], "warnings": warnings}

    champion = report.get("champion")
    if not champion or not champion.get("name"):
        reasons.append("no champion recorded in free_search_report.json")
        return {"passed": False, "reasons": reasons, "warnings": warnings}

    pool_basis = report.get("champion_pool_basis")
    if anchor_expected:
        if champion.get("anchor_metrics") is None:
            warnings.append("anchor expected but champion has no anchor_metrics.")
        # honesty gate: do not crown an unviable candidate as a success unless the
        # run explicitly reports that NOTHING passed the guardrail.
        if champion.get("viable") is False and pool_basis != "no_candidate_passed_guardrail":
            reasons.append("champion marked non-viable but pool_basis does not admit guardrail failure.")

    passed = not reasons
    return {"passed": passed, "reasons": reasons, "warnings": warnings}
