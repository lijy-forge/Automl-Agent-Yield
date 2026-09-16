"""Guardrails for LLM-generated yield-stress AutoML scripts.

These guardrails do not prescribe a final model. They enforce a small
engineering and scientific contract so generated code stays runnable,
auditable, and comparable.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
from dataclasses import dataclass, field
from glob import glob
from pathlib import Path
from typing import Any


YIELD_TASK_NAME = "yield_stress_regression"


@dataclass
class YieldVerificationResult:
    passed: bool
    reasons: list[str]
    repair_guidance: str
    # Tiered guardrail: hard `reasons` block the run (correctness + anti-leakage);
    # `warnings` are non-blocking metadata/consistency/reporting notes recorded
    # for audit but not used to fail the OperationAgent retry loop.
    warnings: list[str] = field(default_factory=list)


def is_yield_task(task: str | None, user_requirements: dict[str, Any] | None = None) -> bool:
    text = str(task or "").lower()
    if isinstance(user_requirements, dict):
        text += " " + str(user_requirements).lower()
    return (
        "yield_stress_regression" in text
        or "yield stress" in text
        or "yield-stress" in text
        or "屈服" in text
    )


def build_yield_contract_instruction(run_dir: str | None = None, data_path: str | None = None) -> str:
    run_dir = run_dir or os.getenv("YIELD_RUN_DIR") or "agent_workspace/runs/yield_search"
    data_path = data_path or os.getenv("YIELD_DATA_PATH") or (
        "/Users/lijiayao/my-project/Yield-value-prediction/data/lian2025/high_fidelity/all_400.csv"
    )
    test_path = os.getenv("YIELD_TEST_PATH") or (
        "/Users/lijiayao/my-project/Yield-value-prediction/data/lian2025/high_fidelity/table6.csv"
    )
    anchor_path = os.getenv("YIELD_ANCHOR_PATH") or test_path
    synthetic_report_path = os.getenv("YIELD_SYNTHETIC_REPORT_PATH") or ""
    return f"""

# Yield-stress AutoML generation contract
This task must remain LLM-generated. The model architecture, feature
engineering, target transform, physics/regularization terms, optimizer,
scheduler, and hyperparameters may be freely designed by the LLM.

However, the generated script must satisfy this contract:
- Read the training CSV from: {data_path}
- If useful, evaluate on the fixed temporary test CSV: {test_path}
- If synthetic data are active and this real anchor CSV exists, run secondary
  anchor validation on: {anchor_path}
- If synthetic data are active, copy or summarize this synthetic audit report
  into RUN_DIR/logs/synthetic_data_report.json: {synthetic_report_path}
- Save every artifact under RUN_DIR = Path(os.environ.get("YIELD_RUN_DIR", "{run_dir}")).resolve().
- Import and use `from knowledge.yield_schema import load_yield_dataframe` for schema normalization.
- Import and use `from knowledge.yield_baselines import run_fixed_yield_baselines`
  for stable baseline comparisons.
- Do not redefine local functions named load_yield_dataframe or
  run_fixed_yield_baselines. These names must come from the project imports
  above so leakage handling and fixed baselines remain stable.
- run_fixed_yield_baselines returns a dict-like bundle. Use
  baseline_bundle["baseline_results"] as the fixed baseline rows. Do not iterate
  over the top-level dict as if it were a list.
- The normalized target column is `yield_stress`.
- Keep the normalized target column `yield_stress` available for y extraction.
- Never use target/leakage/reference columns as FEATURES: yield_stress, Tau0_Pa, tau_Pa, phi_max, m1_true, m1_lf, target, label.
- Correct pattern:
  df, meta = load_yield_dataframe(path)
  y = df["yield_stress"].to_numpy(...)
  feature_df = df.drop(columns=[existing leakage/reference/id/target columns], errors="ignore")
  X = feature_df[feature_columns]
- Do not drop `yield_stress` from the main dataframe before extracting y.
- Fit preprocessing only on train folds/splits. Use validation/test data only for evaluation.
- Use one fitted feature/preprocessing schema for train/validation/anchor
  transforms. Do not independently one-hot encode or engineer train and anchor
  into different column spaces.
- Always run 5-fold OOF on the active training dataset. Do not report metrics
  from training predictions as final performance.
- Select exactly one secondary evaluation. If an anchor CSV exists, use
  anchor_validation as the secondary evaluation; otherwise use repeated K-fold
  or a physics-aware split only when justified by data size/schema.
- In metrics/metrics.json, include evaluation_protocol and, when possible, evaluation_rationale.
- In metrics/metrics.json, include primary_evaluation="5-fold OOF" and
  secondary_evaluation. When anchor_validation is used, include
  anchor_validation metrics with R2/RMSE/MAE/MAPE when mathematically possible.
- Compare baseline models and record selected_model plus candidate_models or
  baseline_results in metrics/metrics.json or logs/mechanism_report.json. At
  minimum include a no-skill mean baseline, a linear/Ridge baseline, an
  unconstrained tree baseline, and a monotonic tree baseline only when sklearn
  HistGradientBoostingRegressor supports monotonic_cst and the feature semantics
  justify a monotonic direction. Do not force superplasticizer-style monotonicity
  onto curing-agent/hardener columns.
- Fixed baselines are reference controls only. They must not be selected as the
  final research model. Compare the best generated/research candidate against
  the best fixed baseline and save best_baseline, best_baseline_metric,
  best_research_model, best_research_metric, selection_metric or
  selection_metric_name, selection_metric_direction, beats_best_baseline, and
  research_model_status. best_baseline_metric and best_research_metric must be
  values of the same metric named by selection_metric; do not put R2 values into
  RMSE fields or RMSE values into R2 fields. If the research model does not beat
  the best fixed baseline, keep the research model as selected_model but set
  beats_best_baseline=false and research_model_status="underperforms_baseline"
  or "completed_with_warning".
- If the CandidateAgent report includes candidate_hybrid_strategies, inspect the selected strategy first. If data_support is supported or safely partial, implement its model+mechanism combination in code. If not implemented, record rejected_hybrid_strategies with strategy id, model id, mechanism ids, and reason in logs/mechanism_report.json.
- Save planned_model and actual_model in metrics/metrics.json or
  logs/mechanism_report.json. If they differ, save a non-empty fallback_reason
  explaining package availability, unsupported data columns, numerical
  instability, or another concrete reason.
- Do not read phi_max, phi_max_eff, phi_max_eff_reference, m1_true, m1_lf, or any
  *_reference column as a sample-level input feature or as a derived mechanism
  feature. If a hidden/global fitted physical parameter is used, fit it only
  from the training fold/split and document the leakage-safe fitting procedure.
- Compute finite R2, RMSE, MAE, and MAPE when mathematically possible.
- Report every MAPE value as a percentage, not a fraction. If using sklearn
  mean_absolute_percentage_error, multiply the returned value by 100 before
  writing metrics JSON so units match fixed baselines.
- Do not write NaN or Infinity into JSON artifacts. If a metric is not
  applicable, use null/None or omit the optional metric field.
- Predictions must be finite and non-negative. If raw predictions can be negative, save both raw and clipped predictions.
- Save predictions/yield_predictions.csv with columns:
  sample_id, prediction_source, yield_stress_actual, yield_stress_predicted.
- Save metrics/metrics.json with finite numeric metrics.
- Before json.dump/json.dumps, convert numpy and pandas scalar/array objects to
  plain Python JSON types. Include a recursive to_jsonable helper that handles
  np.integer, np.floating, np.bool_, np.ndarray, lists, tuples, and dicts, and
  converts NaN/Infinity to None/null, then call
  json.dump(to_jsonable(metrics), ...).
- Save at least one model artifact under trained_models/ (.pt, .pth, .pkl, or .joblib).
- Save preprocessing/preprocessing.json or a scaler/preprocessor artifact.
- Save preprocessing/preprocessing_audit.json. It must include raw_feature_columns,
  final_feature_columns used by the selected final model, optional
  engineered_feature_columns considered, train_feature_count, anchor_feature_count
  when anchor validation exists, anchor_aligned_to_train_schema,
  fold_local_preprocessing, final_model_fit_scope, and leakage_columns_excluded.
  train_feature_count and anchor_feature_count must match len(final_feature_columns).
- Save logs/mechanism_report.json. This file is mandatory.
- If synthetic data are used, save logs/synthetic_data_report.json. This file
  must state source paper/DOI, generation formulas, assumptions, and limitations.
- Print readable training progress to stdout before the final success line:
  dataset shape/features, candidate model names, fold or split metrics, selected
  model, final training status, optional fixed-test metrics, and artifact paths.

Mechanism report requirements:
- If the model uses any mechanistic equation, physics-informed loss, hidden physical variable, monotonicity rule, or physics-guided synthetic data, list every mechanism with:
  name, paper_or_source, exact_formula_or_relationship, columns_used, implementation_role, and why_applicable.
- When external search snippets are available, each used mechanism must also include
  source_ids and source_evidence. source_ids must reference searched source_id
  values from search_report.json. source_evidence should list at least one
  searched title/link/DOI/provider plus the exact claim used.
- If no mechanism is used, save an empty mechanisms list and set `mechanism_used` to false with a short justification.
- Do not claim a paper supports a formula unless the formula or relationship is actually used in the generated code.
- A selected hybrid strategy only counts as `mechanism_used=true` if the generated code actually uses its mechanism as a feature, baseline, loss, constraint, residual target, or fold-local fitted parameter. Otherwise set `mechanism_used=false` and document why the hybrid was rejected.
- If candidate_report.json includes yodel_packing_audit.must_report_if_not_selected=true and the selected/implemented model is not YODEL/packing/fmax-style, logs/mechanism_report.json must include rejected_yodel_packing_candidate with the candidate id, support status, and a concrete data, performance, or numerical-stability reason.
- When rejected_yodel_packing_candidate is required, save it as a structured
  object with candidate_id, reason, and benchmark_evidence. A free-text
  yodel_packing_audit_note is not enough.
- If mechanism_used=true, run and save a same-protocol mechanism ablation
  comparing without_mechanism and with_mechanism. Include delta_rmse, delta_r2,
  and mechanism_improves_metric. If there is no improvement, state that the
  mechanism was audited but did not improve the selected metric.
- If mechanism_used=true, save mechanism_effect_size in metrics or
  mechanism_report with relative_rmse_improvement, relative_rmse_improvement_percent,
  and claim_strength. relative_rmse_improvement must be a fraction such as
  0.013 for 1.3%; relative_rmse_improvement_percent must be 1.3. Use
  claim_strength="marginal" when relative RMSE improvement is below 1%; do not
  present marginal gains as strong mechanistic validation.
- Save selected_hybrid_strategy and benchmark_alignment in metrics or
  mechanism_report. benchmark_alignment must include selected_strategy_id,
  benchmark_proxy_model, benchmark_feature_mode, implemented_model,
  implemented_feature_mode, alignment_status, and deviation_reason when the
  implemented model/features differ from benchmark_audit.
- Save implementation_validation in metrics or mechanism_report. It must compare
  the benchmark-selected strategy with the final implemented model using:
  benchmark_selected_strategy, benchmark_oof_rmse, actual_oof_rmse,
  benchmark_anchor_r2 when available, actual_anchor_r2 when available,
  generalization_gap, and validation_status. If actual anchor R2 is negative or
  the final implementation is materially worse than the benchmark proxy, mark
  validation_status as degraded/failed_generalization/completed_with_warning,
  not validated.
- Run the benchmark-selected proxy as an actual candidate in the generated
  script. Save benchmark_selected_proxy_result in metrics or mechanism_report
  with strategy_id, proxy_model, feature_mode, OOF metrics, and anchor metrics
  when anchor validation exists. Additional models may be tried, but the selected
  proxy must be evaluated instead of only approximated by a different family.
- Save research_confidence in metrics/metrics.json. For synthetic training data
  with anchor validation, anchor R2 < 0 or underperforming the best fixed
  baseline must map to research_confidence="low".

Print exactly "Yield pipeline completed successfully" only after all mandatory artifacts are saved.
"""


def _fresh_files(run_dir: Path, subdir: str, patterns: list[str], started_at: float) -> list[Path]:
    directory = run_dir / subdir
    if not directory.is_dir():
        return []
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(Path(path) for path in glob(str(directory / pattern)))
    return [
        path
        for path in paths
        if path.is_file() and path.stat().st_size > 0 and path.stat().st_mtime >= started_at - 1.0
    ]


def _load_json(paths: list[Path]) -> tuple[Path | None, dict[str, Any] | None]:
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, dict):
            return path, payload
    return None, None


def _flatten_numbers(value: Any) -> list[float]:
    if isinstance(value, dict):
        out: list[float] = []
        for child in value.values():
            out.extend(_flatten_numbers(child))
        return out
    if isinstance(value, (list, tuple)):
        out = []
        for child in value:
            out.extend(_flatten_numbers(child))
        return out
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        number = float(value)
        return [number] if math.isfinite(number) else []
    return []


def _has_finite_metric_numbers(metrics: dict[str, Any]) -> bool:
    nums = _flatten_numbers(metrics)
    return bool(nums) and all(math.isfinite(v) for v in nums)


def _candidate_model_count(metrics: dict[str, Any]) -> int:
    for key in ("baseline_results", "candidate_models", "model_results"):
        value = metrics.get(key)
        if isinstance(value, list):
            return len(value)
        if isinstance(value, dict):
            return len(value)
    return 0


def _metric_number(entry: Any, names: tuple[str, ...]) -> float | None:
    if not isinstance(entry, dict):
        return None
    for name in names:
        value = entry.get(name)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
    return None


def _named_metric_rows(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        nested = value.get("baseline_results")
        if isinstance(nested, list):
            return [item for item in nested if isinstance(item, dict)]
        rows: list[dict[str, Any]] = []
        for name, item in value.items():
            if isinstance(item, dict):
                row = dict(item)
                row.setdefault("name", name)
                rows.append(row)
        return rows
    return []


def _best_rmse(rows: list[dict[str, Any]]) -> tuple[str | None, float | None]:
    best_name: str | None = None
    best_value: float | None = None
    for row in rows:
        value = _metric_number(row, ("oof_rmse", "mean_rmse", "rmse", "RMSE"))
        if value is None:
            continue
        if best_value is None or value < best_value:
            best_value = value
            best_name = str(row.get("name") or row.get("model") or row.get("model_name") or "")
    return best_name, best_value


def _selection_metric_kind(metric_name: str) -> str | None:
    lowered = str(metric_name or "").strip().lower()
    if "mape" in lowered:
        return "mape"
    if "rmse" in lowered:
        return "rmse"
    if "mae" in lowered:
        return "mae"
    if "r2" in lowered or "r^2" in lowered or "r_squared" in lowered or "r-squared" in lowered:
        return "r2"
    return None


def _metric_aliases(metric_kind: str) -> tuple[str, ...]:
    if metric_kind == "rmse":
        return ("oof_rmse", "mean_rmse", "rmse", "RMSE")
    if metric_kind == "mae":
        return ("oof_mae", "mean_mae", "mae", "MAE")
    if metric_kind == "mape":
        return ("oof_mape", "mean_mape", "mape", "MAPE")
    if metric_kind == "r2":
        return ("oof_r2", "mean_r2", "r2", "R2", "r_squared", "R_squared")
    return ()


def _metric_direction(metric_kind: str) -> str:
    return "higher_is_better" if metric_kind == "r2" else "lower_is_better"


def _direction_matches(reported: str, expected: str) -> bool:
    lowered = str(reported or "").strip().lower().replace("-", "_").replace(" ", "_")
    if expected == "lower_is_better":
        return "lower" in lowered or "min" in lowered
    return "higher" in lowered or "max" in lowered


def _best_metric(rows: list[dict[str, Any]], metric_kind: str) -> tuple[str | None, float | None]:
    aliases = _metric_aliases(metric_kind)
    if not aliases:
        return None, None
    higher = _metric_direction(metric_kind) == "higher_is_better"
    best_name: str | None = None
    best_value: float | None = None
    for row in rows:
        value = _metric_number(row, aliases)
        if value is None:
            continue
        if best_value is None or (value > best_value if higher else value < best_value):
            best_value = value
            best_name = str(row.get("name") or row.get("model") or row.get("model_name") or "")
    return best_name, best_value


def _metric_close(reported: float | None, expected: float | None, metric_kind: str) -> bool:
    if reported is None or expected is None:
        return False
    tolerance = 0.05 if metric_kind == "r2" else 0.05 * max(1.0, abs(expected))
    return abs(float(reported) - float(expected)) <= max(1e-6, tolerance)


def _selection_metric_consistency_reasons(metrics: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    baseline_rows = _named_metric_rows(metrics.get("baseline_results"))
    candidate_rows = _named_metric_rows(metrics.get("candidate_models") or metrics.get("model_results"))
    if not baseline_rows and not candidate_rows:
        return reasons

    selection_metric = str(metrics.get("selection_metric_name") or metrics.get("selection_metric") or "").strip()
    if not selection_metric:
        return ["Metrics JSON must include selection_metric or selection_metric_name so best_baseline_metric and best_research_metric are auditable."]
    metric_kind = _selection_metric_kind(selection_metric)
    if metric_kind is None:
        return [
            "selection_metric must clearly identify RMSE, MAE, MAPE, or R2; otherwise best_baseline_metric and best_research_metric cannot be audited."
        ]

    expected_direction = _metric_direction(metric_kind)
    reported_direction = str(metrics.get("selection_metric_direction") or metrics.get("metric_direction") or "").strip()
    if not reported_direction:
        reasons.append("Metrics JSON must include selection_metric_direction, e.g. lower_is_better for RMSE/MAE/MAPE or higher_is_better for R2.")
    elif not _direction_matches(reported_direction, expected_direction):
        reasons.append(
            f"selection_metric={selection_metric} implies {expected_direction}, but selection_metric_direction={reported_direction!r}."
        )

    baseline_name, expected_baseline = _best_metric(baseline_rows, metric_kind)
    research_name, expected_research = _best_metric(candidate_rows, metric_kind)
    reported_baseline = _metric_number(metrics, ("best_baseline_metric",))
    reported_research = _metric_number(metrics, ("best_research_metric",))

    if expected_baseline is not None:
        if reported_baseline is None:
            reasons.append("best_baseline_metric must be numeric and use the same metric as selection_metric.")
        elif not _metric_close(reported_baseline, expected_baseline, metric_kind):
            reasons.append(
                f"selection_metric={selection_metric} but best_baseline_metric={reported_baseline:.6g} "
                f"does not match the best baseline {metric_kind}={expected_baseline:.6g}"
                + (f" ({baseline_name})." if baseline_name else ".")
            )

    if expected_research is not None:
        if reported_research is None:
            reasons.append("best_research_metric must be numeric and use the same metric as selection_metric.")
        elif not _metric_close(reported_research, expected_research, metric_kind):
            reasons.append(
                f"selection_metric={selection_metric} but best_research_metric={reported_research:.6g} "
                f"does not match the best research {metric_kind}={expected_research:.6g}"
                + (f" ({research_name})." if research_name else ".")
            )

    if reported_baseline is not None and reported_research is not None and isinstance(metrics.get("beats_best_baseline"), bool):
        if expected_direction == "lower_is_better":
            expected_beats = reported_research <= reported_baseline
        else:
            expected_beats = reported_research >= reported_baseline
        if bool(metrics.get("beats_best_baseline")) != expected_beats:
            reasons.append("beats_best_baseline conflicts with best_baseline_metric, best_research_metric, and selection_metric_direction.")
    return reasons


def _baseline_audit_reasons(metrics: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    selected = str(metrics.get("selected_model") or "").strip().lower()
    if not selected:
        reasons.append("Metrics JSON must include selected_model.")
    elif selected.startswith("baseline_") or selected in {"dummy", "mean_dummy", "ridge_linear", "fixed_baseline"}:
        reasons.append("selected_model must be a generated research model, not a fixed baseline.")

    baseline_rows = _named_metric_rows(metrics.get("baseline_results"))
    candidate_rows = _named_metric_rows(metrics.get("candidate_models") or metrics.get("model_results"))
    best_baseline_name, best_baseline_rmse = _best_rmse(baseline_rows)
    best_research_name, best_research_rmse = _best_rmse(candidate_rows)

    required = (
        "best_baseline",
        "best_baseline_metric",
        "best_research_model",
        "best_research_metric",
        "beats_best_baseline",
        "research_model_status",
    )
    missing = [key for key in required if key not in metrics]
    if missing:
        reasons.append("Metrics JSON must include baseline audit fields: " + ", ".join(missing) + ".")

    if best_baseline_rmse is not None and best_research_rmse is not None:
        beats = bool(metrics.get("beats_best_baseline"))
        status = str(metrics.get("research_model_status") or "").strip().lower()
        if best_research_rmse > best_baseline_rmse * 1.02:
            if beats:
                reasons.append("beats_best_baseline cannot be true when the best research RMSE is worse than the best baseline RMSE.")
            if status not in {"underperforms_baseline", "completed_with_warning", "needs_research_revision"}:
                reasons.append(
                    "Research model underperforms the best fixed baseline; set research_model_status='underperforms_baseline' or 'completed_with_warning'."
                )
        elif "beats_best_baseline" in metrics and not beats and status not in {
            "beats_baseline",
            "competitive_with_baseline",
            "completed_with_warning",
            "underperforms_baseline",
        }:
            reasons.append("research_model_status must explain the baseline comparison outcome.")

    reported_best_baseline = str(metrics.get("best_baseline") or "").strip()
    if best_baseline_name and reported_best_baseline and best_baseline_name not in reported_best_baseline:
        # Do not fail on naming style differences; require only that a baseline was reported.
        pass
    return reasons


def _planned_actual_audit_reasons(metrics: dict[str, Any], mechanism_report: dict[str, Any] | None) -> list[str]:
    reasons: list[str] = []
    audit_sources = [metrics]
    if isinstance(mechanism_report, dict):
        audit_sources.append(mechanism_report)
        selected_strategy = mechanism_report.get("selected_hybrid_strategy")
        if isinstance(selected_strategy, dict):
            audit_sources.append(selected_strategy)

    planned = ""
    actual = ""
    fallback = ""
    for payload in audit_sources:
        planned = planned or str(payload.get("planned_model") or payload.get("planned_model_family") or "").strip()
        actual = actual or str(payload.get("actual_model") or payload.get("actual_model_family") or "").strip()
        fallback = fallback or str(payload.get("fallback_reason") or payload.get("package_availability") or "").strip()

    if not planned:
        reasons.append("Metrics or mechanism report must include planned_model.")
    if not actual:
        reasons.append("Metrics or mechanism report must include actual_model.")
    if planned and actual and planned.lower() != actual.lower() and not fallback:
        reasons.append("planned_model differs from actual_model; provide a non-empty fallback_reason.")
    return reasons


def _mechanism_ablation_reasons(metrics: dict[str, Any], mechanism_report: dict[str, Any] | None) -> list[str]:
    if not isinstance(mechanism_report, dict) or mechanism_report.get("mechanism_used") is not True:
        return []

    ablation = metrics.get("mechanism_ablation")
    if not isinstance(ablation, dict):
        ablation = mechanism_report.get("mechanism_ablation")
    if not isinstance(ablation, dict):
        ablation = mechanism_report.get("ablation_study")
    if not isinstance(ablation, dict):
        return ["Mechanism-used runs must save mechanism_ablation comparing without_mechanism and with_mechanism under the same OOF protocol."]

    required = {"without_mechanism", "with_mechanism", "delta_rmse", "delta_r2", "mechanism_improves_metric"}
    missing = [key for key in sorted(required) if key not in ablation]
    if missing:
        return ["mechanism_ablation is missing required fields: " + ", ".join(missing) + "."]

    without_rmse = _metric_number(ablation.get("without_mechanism"), ("rmse", "RMSE", "oof_rmse", "mean_rmse"))
    with_rmse = _metric_number(ablation.get("with_mechanism"), ("rmse", "RMSE", "oof_rmse", "mean_rmse"))
    if without_rmse is not None and with_rmse is not None and abs(without_rmse) > 1e-12:
        relative_improvement = (without_rmse - with_rmse) / abs(without_rmse)
        effect = metrics.get("mechanism_effect_size")
        if not isinstance(effect, dict):
            effect = mechanism_report.get("mechanism_effect_size")
        if not isinstance(effect, dict):
            return [
                "Mechanism-used runs must save mechanism_effect_size with relative_rmse_improvement and claim_strength."
            ]
        reported = _metric_number(effect, ("relative_rmse_improvement", "relative_improvement", "rmse_relative_improvement"))
        reported_percent = _metric_number(
            effect,
            (
                "relative_rmse_improvement_percent",
                "relative_improvement_percent",
                "rmse_relative_improvement_percent",
            ),
        )
        claim = str(effect.get("claim_strength") or "").strip().lower()
        if reported is None and reported_percent is None:
            return ["mechanism_effect_size must include relative_rmse_improvement as a fraction or relative_rmse_improvement_percent as a percentage."]
        if reported is not None and reported > 1.0:
            return [
                "mechanism_effect_size.relative_rmse_improvement must be a fraction such as 0.013 for 1.3%; put percent values in relative_rmse_improvement_percent."
            ]
        if reported is not None and abs(reported - relative_improvement) > 0.02:
            return ["mechanism_effect_size.relative_rmse_improvement does not match mechanism_ablation RMSE values."]
        if reported_percent is not None and abs(reported_percent - relative_improvement * 100.0) > 2.0:
            return ["mechanism_effect_size.relative_rmse_improvement_percent does not match mechanism_ablation RMSE values."]
        if relative_improvement <= 0 and ablation.get("mechanism_improves_metric") is True:
            return ["mechanism_improves_metric cannot be true when with_mechanism RMSE is not lower than without_mechanism RMSE."]
        if relative_improvement < 0.01 and claim not in {"marginal", "none", "no_gain", "no_improvement", "negative"}:
            return [
                "Mechanism RMSE improvement is below 1%; mechanism_effect_size.claim_strength must be 'marginal' or weaker."
            ]
    return []


def _research_confidence_reasons(metrics: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    confidence = str(metrics.get("research_confidence") or "").strip().lower()
    if not confidence:
        return ["Metrics JSON must include research_confidence."]

    anchor = metrics.get("anchor_validation") or metrics.get("anchor_validation_metrics") or metrics.get("anchor_metrics")
    anchor_r2 = None
    if isinstance(anchor, dict):
        anchor_r2 = _metric_number(anchor, ("r2", "R2", "anchor_r2"))
    if anchor_r2 is None:
        anchor_r2 = _metric_number(metrics, ("anchor_r2", "anchor_R2"))

    underperforms = str(metrics.get("research_model_status") or "").strip().lower() in {
        "underperforms_baseline",
        "needs_research_revision",
    } or metrics.get("beats_best_baseline") is False
    if ((anchor_r2 is not None and anchor_r2 < 0) or underperforms) and confidence != "low":
        reasons.append("research_confidence must be 'low' when anchor R2 < 0 or the research model underperforms the best baseline.")
    return reasons


def _mape_unit_reasons(metrics: dict[str, Any]) -> list[str]:
    baseline_rows = _named_metric_rows(metrics.get("baseline_results"))
    baseline_mapes = [
        value
        for row in baseline_rows
        for value in (_metric_number(row, ("oof_mape", "mean_mape", "mape", "MAPE")),)
        if value is not None
    ]
    if not baseline_mapes or max(baseline_mapes) <= 5.0:
        return []

    checked: list[tuple[str, float]] = []
    for label, payload in (
        ("oof_average", metrics.get("oof_average")),
        ("oof_metrics", metrics.get("oof_metrics")),
        ("anchor_validation", metrics.get("anchor_validation") or metrics.get("anchor_validation_metrics") or metrics.get("anchor_metrics")),
    ):
        value = _metric_number(payload, ("mape", "MAPE", "oof_mape", "anchor_mape")) if isinstance(payload, dict) else None
        if value is not None:
            checked.append((label, value))
    for row in _named_metric_rows(metrics.get("candidate_models") or metrics.get("model_results")):
        value = _metric_number(row, ("oof_mape", "mean_mape", "mape", "MAPE"))
        if value is not None:
            checked.append((str(row.get("name") or row.get("model") or "candidate"), value))
    fractional = [(label, value) for label, value in checked if 0.0 < value < 1.0]
    if fractional:
        examples = ", ".join(f"{label}={value:.4g}" for label, value in fractional[:4])
        return [
            "MAPE appears to mix percentage and fractional units. Fixed baselines report MAPE as percent; "
            f"these entries look fractional and must be multiplied by 100 or renamed: {examples}."
        ]
    return []


def _strategy_id_from_value(value: Any) -> str:
    if isinstance(value, dict):
        for key in ("strategy_id", "id", "selected_strategy_id"):
            if value.get(key) not in (None, ""):
                return str(value.get(key))
    if isinstance(value, str):
        return value.strip()
    return ""


def _strategy_rejected(strategy_id: str, payloads: list[dict[str, Any]]) -> bool:
    if not strategy_id:
        return False
    for payload in payloads:
        rejected = payload.get("rejected_hybrid_strategies")
        if not isinstance(rejected, list):
            continue
        for item in rejected:
            if isinstance(item, dict) and str(item.get("strategy_id") or item.get("id") or "") == strategy_id:
                return bool(str(item.get("reason") or "").strip())
    return False


def _candidate_alignment_reasons(run_path: Path, metrics: dict[str, Any] | None, mechanism_report: dict[str, Any] | None) -> list[str]:
    if metrics is None:
        return []
    candidate_path = run_path / "candidate_report.json"
    try:
        candidate_report = json.loads(candidate_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    selected = candidate_report.get("selected_combination")
    if not isinstance(selected, dict) or not selected.get("strategy_id"):
        return []
    expected_strategy_id = str(selected.get("strategy_id"))
    payloads = [metrics]
    if isinstance(mechanism_report, dict):
        payloads.append(mechanism_report)

    reported_strategy_id = ""
    for payload in payloads:
        reported_strategy_id = reported_strategy_id or _strategy_id_from_value(payload.get("selected_hybrid_strategy"))

    reasons: list[str] = []
    if not reported_strategy_id and not _strategy_rejected(expected_strategy_id, payloads):
        reasons.append(
            "Metrics or mechanism_report must include selected_hybrid_strategy matching CandidateAgent's benchmark-selected strategy, or reject that strategy with a concrete reason."
        )
    elif reported_strategy_id and reported_strategy_id != expected_strategy_id and not _strategy_rejected(expected_strategy_id, payloads):
        reasons.append(
            f"selected_hybrid_strategy={reported_strategy_id} does not match benchmark-selected strategy {expected_strategy_id}; reject the selected strategy explicitly if using a fallback."
        )

    alignment = None
    for payload in payloads:
        if isinstance(payload.get("benchmark_alignment"), dict):
            alignment = payload.get("benchmark_alignment")
            break
    if alignment is None:
        reasons.append(
            "Metrics or mechanism_report must include benchmark_alignment describing selected_strategy_id, benchmark proxy, implemented model, implemented feature mode, and deviations."
        )
    else:
        aligned_strategy = str(alignment.get("selected_strategy_id") or "")
        if aligned_strategy and aligned_strategy != expected_strategy_id:
            reasons.append(
                f"benchmark_alignment.selected_strategy_id={aligned_strategy} does not match benchmark-selected strategy {expected_strategy_id}."
            )
        status = str(alignment.get("alignment_status") or "").strip().lower()
        if status in {"deviated", "fallback", "partial"} and not str(alignment.get("deviation_reason") or "").strip():
            reasons.append("benchmark_alignment with fallback/partial/deviated status must include deviation_reason.")
    return reasons


def _load_run_json(run_path: Path, filename: str) -> dict[str, Any] | None:
    path = run_path / filename
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _selected_benchmark_row(run_path: Path) -> dict[str, Any] | None:
    audit = _load_run_json(run_path, "candidate_selection_audit.json")
    if isinstance(audit, dict):
        row = audit.get("selected_strategy_benchmark")
        if isinstance(row, dict):
            return row
    benchmark = _load_run_json(run_path, "candidate_benchmark_report.json")
    if isinstance(benchmark, dict):
        selected_id = benchmark.get("selected_strategy_id")
        for row in benchmark.get("strategy_benchmarks", []) or []:
            if isinstance(row, dict) and str(row.get("strategy_id") or "") == str(selected_id or ""):
                return row
    candidate_report = _load_run_json(run_path, "candidate_report.json")
    if isinstance(candidate_report, dict):
        audit = candidate_report.get("candidate_selection_audit")
        if isinstance(audit, dict) and isinstance(audit.get("selected_strategy_benchmark"), dict):
            return audit["selected_strategy_benchmark"]
    return None


def _find_dict_field(payloads: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    for payload in payloads:
        value = payload.get(key)
        if isinstance(value, dict):
            return value
    return None


def _extract_oof_rmse(metrics: dict[str, Any]) -> float | None:
    for key in ("oof_average", "oof_metrics", "primary_metrics", "validation_metrics"):
        value = _metric_number(metrics.get(key), ("rmse", "RMSE", "oof_rmse", "mean_rmse"))
        if value is not None:
            return value
    return _metric_number(metrics, ("oof_rmse", "mean_rmse", "rmse", "RMSE"))


def _extract_anchor_r2(metrics: dict[str, Any]) -> float | None:
    anchor = metrics.get("anchor_validation") or metrics.get("anchor_validation_metrics") or metrics.get("anchor_metrics")
    value = _metric_number(anchor, ("r2", "R2", "anchor_r2")) if isinstance(anchor, dict) else None
    if value is not None:
        return value
    return _metric_number(metrics, ("anchor_r2", "anchor_R2"))


def _row_text(row: dict[str, Any]) -> str:
    chunks: list[str] = []
    for key in (
        "name",
        "model",
        "model_name",
        "family",
        "proxy_model",
        "feature_mode",
        "implemented_feature_mode",
        "strategy_id",
        "selected_strategy_id",
    ):
        value = row.get(key)
        if value not in (None, ""):
            chunks.append(str(value))
    return " ".join(chunks).lower()


def _proxy_result_from_payload(payload: dict[str, Any], expected_strategy: str) -> dict[str, Any] | None:
    for key in (
        "benchmark_selected_proxy_result",
        "benchmark_proxy_result",
        "selected_proxy_result",
        "required_proxy_result",
    ):
        value = payload.get(key)
        if isinstance(value, dict):
            return value
    rows = _named_metric_rows(payload.get("candidate_models") or payload.get("model_results"))
    for row in rows:
        if str(row.get("strategy_id") or row.get("selected_strategy_id") or "") == expected_strategy:
            return row
    for row in rows:
        if row.get("benchmark_selected_proxy") is True or row.get("required_proxy") is True:
            return row
    return None


def _implementation_validation_reasons(
    run_path: Path,
    metrics: dict[str, Any] | None,
    mechanism_report: dict[str, Any] | None,
) -> list[str]:
    if metrics is None:
        return []
    selected_row = _selected_benchmark_row(run_path)
    if not isinstance(selected_row, dict) or not selected_row.get("strategy_id"):
        return []

    payloads = [metrics]
    if isinstance(mechanism_report, dict):
        payloads.append(mechanism_report)
    validation = _find_dict_field(payloads, "implementation_validation")
    if validation is None:
        return [
            "Metrics or mechanism_report must include implementation_validation comparing the actual final implementation against the benchmark-selected candidate."
        ]

    reasons: list[str] = []
    expected_strategy = str(selected_row.get("strategy_id") or "")
    expected_proxy = str(selected_row.get("proxy_model") or "").strip().lower()
    expected_feature_mode = str(selected_row.get("feature_mode") or "").strip().lower()
    required_proxy_result = _proxy_result_from_payload(metrics, expected_strategy)
    if required_proxy_result is None and isinstance(mechanism_report, dict):
        required_proxy_result = _proxy_result_from_payload(mechanism_report, expected_strategy)
    if expected_proxy or expected_feature_mode:
        if not isinstance(required_proxy_result, dict):
            reasons.append(
                "Metrics or mechanism_report must include benchmark_selected_proxy_result for the benchmark-selected proxy model/feature mode as an actually evaluated candidate."
            )
        else:
            proxy_text = _row_text(required_proxy_result)
            if expected_strategy and str(required_proxy_result.get("strategy_id") or required_proxy_result.get("selected_strategy_id") or expected_strategy) != expected_strategy:
                reasons.append("benchmark_selected_proxy_result.strategy_id must match the benchmark-selected strategy.")
            if expected_proxy and expected_proxy not in proxy_text:
                reasons.append(
                    f"benchmark_selected_proxy_result must identify and evaluate the benchmark proxy model '{expected_proxy}'."
                )
            if expected_feature_mode and expected_feature_mode not in proxy_text:
                reasons.append(
                    f"benchmark_selected_proxy_result must identify and evaluate the benchmark feature mode '{expected_feature_mode}'."
                )
            proxy_rmse = _metric_number(required_proxy_result, ("oof_rmse", "rmse", "mean_rmse", "RMSE"))
            if proxy_rmse is None:
                reasons.append("benchmark_selected_proxy_result must include OOF RMSE for the actually executed proxy candidate.")
            if isinstance(selected_row.get("anchor_validation"), dict):
                proxy_anchor = required_proxy_result.get("anchor_validation") or required_proxy_result.get("anchor_metrics")
                proxy_anchor_r2 = _metric_number(proxy_anchor, ("r2", "R2", "anchor_r2")) if isinstance(proxy_anchor, dict) else _metric_number(required_proxy_result, ("anchor_r2", "anchor_R2"))
                if proxy_anchor_r2 is None:
                    reasons.append("benchmark_selected_proxy_result must include anchor R2 when benchmark anchor validation exists.")
    reported_strategy = str(
        validation.get("benchmark_selected_strategy")
        or validation.get("selected_strategy_id")
        or validation.get("strategy_id")
        or ""
    )
    if not reported_strategy:
        reasons.append("implementation_validation must include benchmark_selected_strategy.")
    elif reported_strategy != expected_strategy:
        reasons.append(
            f"implementation_validation.benchmark_selected_strategy={reported_strategy} does not match benchmark-selected strategy {expected_strategy}."
        )

    benchmark_oof = _metric_number(selected_row.get("oof_metrics"), ("rmse", "RMSE", "oof_rmse", "mean_rmse"))
    reported_benchmark_oof = _metric_number(validation, ("benchmark_oof_rmse", "benchmark_selected_oof_rmse"))
    if benchmark_oof is not None:
        if reported_benchmark_oof is None:
            reasons.append("implementation_validation must include benchmark_oof_rmse from candidate_selection_audit.")
        elif not _metric_close(reported_benchmark_oof, benchmark_oof, "rmse"):
            reasons.append("implementation_validation.benchmark_oof_rmse does not match candidate_selection_audit selected benchmark OOF RMSE.")

    actual_oof = _metric_number(validation, ("actual_oof_rmse", "implemented_oof_rmse", "final_oof_rmse"))
    if actual_oof is None:
        reasons.append("implementation_validation must include actual_oof_rmse for the final implemented model.")
        actual_oof = _extract_oof_rmse(metrics)

    benchmark_anchor = selected_row.get("anchor_validation") if isinstance(selected_row.get("anchor_validation"), dict) else None
    benchmark_anchor_r2 = _metric_number(benchmark_anchor, ("r2", "R2", "anchor_r2")) if benchmark_anchor else None
    reported_benchmark_anchor_r2 = _metric_number(validation, ("benchmark_anchor_r2", "benchmark_selected_anchor_r2"))
    if benchmark_anchor_r2 is not None:
        if reported_benchmark_anchor_r2 is None:
            reasons.append("implementation_validation must include benchmark_anchor_r2 when the benchmark used anchor validation.")
        elif abs(reported_benchmark_anchor_r2 - benchmark_anchor_r2) > 0.05:
            reasons.append("implementation_validation.benchmark_anchor_r2 does not match candidate benchmark anchor R2.")

    actual_anchor_r2 = _metric_number(validation, ("actual_anchor_r2", "implemented_anchor_r2", "final_anchor_r2"))
    metrics_anchor_r2 = _extract_anchor_r2(metrics)
    if metrics_anchor_r2 is not None and actual_anchor_r2 is not None and abs(actual_anchor_r2 - metrics_anchor_r2) > 0.05:
        reasons.append("implementation_validation.actual_anchor_r2 does not match metrics anchor_validation R2.")
    if benchmark_anchor_r2 is not None or metrics_anchor_r2 is not None:
        if actual_anchor_r2 is None:
            reasons.append("implementation_validation must include actual_anchor_r2 when anchor validation exists.")
            actual_anchor_r2 = metrics_anchor_r2
        if "generalization_gap" not in validation:
            reasons.append("implementation_validation must include generalization_gap for synthetic-to-anchor evaluation.")

    status = str(validation.get("validation_status") or "").strip().lower()
    if not status:
        reasons.append("implementation_validation must include validation_status.")
    strong_statuses = {"validated", "success", "passed", "aligned_success"}
    weak_statuses = {
        "degraded",
        "failed_generalization",
        "anchor_generalization_failed",
        "completed_with_warning",
        "not_validated",
        "underperforms_baseline",
    }
    if actual_oof is not None and benchmark_oof is not None and actual_oof > benchmark_oof * 1.25 and status in strong_statuses:
        reasons.append("implementation_validation cannot be marked validated when actual_oof_rmse is more than 25% worse than the benchmark proxy.")
    if actual_anchor_r2 is not None:
        if actual_anchor_r2 < 0 and status in strong_statuses:
            reasons.append("implementation_validation cannot be marked validated when actual_anchor_r2 < 0.")
        if benchmark_anchor_r2 is not None and actual_anchor_r2 < benchmark_anchor_r2 - 0.2 and status in strong_statuses:
            reasons.append("implementation_validation cannot be marked validated when actual anchor R2 is substantially worse than benchmark anchor R2.")
        if actual_anchor_r2 < 0 and status not in weak_statuses and status not in strong_statuses:
            reasons.append(
                "implementation_validation.validation_status should explicitly mark anchor generalization failure/degradation when actual_anchor_r2 < 0."
            )
    return reasons


def _anchor_generalization_reasons(metrics: dict[str, Any]) -> list[str]:
    anchor_r2 = _extract_anchor_r2(metrics)
    if anchor_r2 is None or anchor_r2 >= 0:
        return []
    reasons: list[str] = []
    status = str(metrics.get("research_model_status") or "").strip().lower()
    if status in {"completed_successfully", "success", "beats_baseline", "validated"}:
        reasons.append("research_model_status cannot be a success/validated status when anchor R2 < 0; mark completed_with_warning, underperforms_baseline, or needs_research_revision.")
    validation = metrics.get("implementation_validation")
    if isinstance(validation, dict):
        gap = validation.get("generalization_gap")
    else:
        gap = metrics.get("generalization_gap")
    if gap in (None, ""):
        reasons.append("When anchor R2 < 0, metrics must record generalization_gap or implementation_validation.generalization_gap.")
    return reasons


def _prediction_file_ok(paths: list[Path]) -> bool:
    required = {"yield_stress_actual", "yield_stress_predicted"}
    for path in paths:
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                fields = set(reader.fieldnames or [])
                if not required.issubset(fields):
                    continue
                for row in reader:
                    pred = float(row["yield_stress_predicted"])
                    if not math.isfinite(pred) or pred < 0:
                        return False
                return True
        except Exception:
            continue
    return False


def _mechanism_report_ok(report: dict[str, Any]) -> bool:
    if "mechanism_used" not in report:
        return False
    mechanisms = report.get("mechanisms", [])
    if report.get("mechanism_used") is False:
        return isinstance(mechanisms, list)
    if not isinstance(mechanisms, list) or not mechanisms:
        return False
    for item in mechanisms:
        if not isinstance(item, dict):
            return False
        name = str(item.get("name") or "").strip()
        paper = str(item.get("paper_or_source") or item.get("paper") or item.get("source") or "").strip()
        formula = str(
            item.get("exact_formula_or_relationship")
            or item.get("formula_or_relationship")
            or item.get("formula")
            or item.get("relationship")
            or ""
        ).strip()
        columns = item.get("columns_used") or item.get("used_columns") or item.get("required_columns")
        role = str(item.get("implementation_role") or item.get("role") or item.get("code_role") or "").strip()
        applicable = str(item.get("why_applicable") or item.get("applicability_reason") or "").strip()
        if not name or not paper or not formula or not columns or not role or not applicable:
            return False
    return True


def _yodel_decision_reasons(run_path: Path, mechanism_report: dict[str, Any] | None) -> list[str]:
    candidate_path = run_path / "candidate_report.json"
    if not candidate_path.exists():
        return []
    try:
        candidate_report = json.loads(candidate_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    yodel_audit = candidate_report.get("yodel_packing_audit", {}) if isinstance(candidate_report, dict) else {}
    if not isinstance(yodel_audit, dict) or yodel_audit.get("must_report_if_not_selected") is not True:
        return []
    if not isinstance(mechanism_report, dict):
        return ["YODEL/packing audit requires an explicit use-or-reject decision, but mechanism_report.json is missing or invalid."]
    selected = yodel_audit.get("selected") is True
    if selected:
        report_text = json.dumps(mechanism_report, ensure_ascii=False).lower()
        implements_yodel = any(
            token in report_text
            for token in ("yodel", "fmax", "phi_m", "packing", "maximum packing")
        )
        if not implements_yodel:
            return [
                "CandidateAgent selected a YODEL/packing/fmax strategy, but mechanism_report.json does not document an implemented YODEL/packing mechanism."
            ]
        return []

    rejection = mechanism_report.get("rejected_yodel_packing_candidate")
    if not isinstance(rejection, (dict, list)):
        nested_yodel = mechanism_report.get("yodel_packing_audit")
        if isinstance(nested_yodel, dict):
            rejection = nested_yodel.get("rejected_yodel_packing_candidate")
    if isinstance(rejection, list):
        rejection_items = [item for item in rejection if isinstance(item, dict)]
        rejection = rejection_items[0] if rejection_items else None
    if not isinstance(rejection, dict):
        return [
            "CandidateAgent required an explicit YODEL/packing/fmax decision; when benchmark does not select it, mechanism_report.json must include structured rejected_yodel_packing_candidate."
        ]
    missing = [
        key
        for key in ("candidate_id", "reason", "benchmark_evidence")
        if not rejection.get(key)
    ]
    if missing:
        return [
            "rejected_yodel_packing_candidate is missing required fields: " + ", ".join(missing) + "."
        ]
    return []


def _preprocessing_uses_leakage(paths: list[Path]) -> str | None:
    forbidden = {
        "yield_stress",
        "Tau0_Pa",
        "tau_Pa",
        "target",
        "label",
        "phi_max",
        "phi_max_eff",
        "phi_max_eff_reference",
        "m1_true",
        "m1_lf",
    }
    for path in paths:
        if path.suffix.lower() != ".json":
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        columns = payload.get("feature_columns") if isinstance(payload, dict) else None
        if not isinstance(columns, list):
            continue
        bad = [str(col) for col in columns if str(col) in forbidden or str(col).endswith("_reference")]
        if bad:
            return ", ".join(sorted(set(bad)))
    return None


def _preprocessing_audit_payload(run_path: Path, metrics: dict[str, Any] | None) -> dict[str, Any] | None:
    if isinstance(metrics, dict) and isinstance(metrics.get("preprocessing_audit"), dict):
        return metrics["preprocessing_audit"]
    candidates = [
        run_path / "preprocessing" / "preprocessing_audit.json",
        run_path / "preprocessing" / "preprocessing.json",
        run_path / "logs" / "preprocessing_audit.json",
    ]
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _bool_field(payload: dict[str, Any], names: tuple[str, ...]) -> bool | None:
    for name in names:
        if name not in payload:
            continue
        value = payload.get(name)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "yes", "1", "passed", "aligned"}:
                return True
            if lowered in {"false", "no", "0", "failed", "not_aligned"}:
                return False
    return None


def _count_field(payload: dict[str, Any], names: tuple[str, ...]) -> int | None:
    for name in names:
        value = payload.get(name)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return int(value)
        if isinstance(value, float) and math.isfinite(value):
            return int(value)
    return None


def _list_field(payload: dict[str, Any], names: tuple[str, ...]) -> list[str]:
    for name in names:
        value = payload.get(name)
        if isinstance(value, list):
            return [str(item) for item in value]
    return []


def _preprocessing_audit_reasons(run_path: Path, metrics: dict[str, Any] | None) -> list[str]:
    audit = _preprocessing_audit_payload(run_path, metrics)
    if not isinstance(audit, dict):
        return [
            "Generated run must save preprocessing/preprocessing_audit.json documenting the fitted feature schema and train/anchor column alignment."
        ]

    reasons: list[str] = []
    raw_columns = _list_field(audit, ("raw_feature_columns", "input_feature_columns", "source_feature_columns"))
    final_columns = _list_field(
        audit,
        (
            "final_feature_columns",
            "selected_feature_columns",
            "model_feature_columns",
            "feature_columns",
            "train_feature_columns",
            "engineered_feature_columns",
        ),
    )
    if not raw_columns:
        reasons.append("preprocessing_audit must include raw_feature_columns.")
    if not final_columns:
        reasons.append("preprocessing_audit must include engineered_feature_columns or final_feature_columns.")

    forbidden = {
        "yield_stress",
        "Tau0_Pa",
        "tau_Pa",
        "target",
        "label",
        "phi_max",
        "phi_max_eff",
        "phi_max_eff_reference",
        "m1_true",
        "m1_lf",
    }
    bad = [col for col in final_columns if col in forbidden or col.endswith("_reference")]
    if bad:
        reasons.append("preprocessing_audit final feature columns include leakage/reference columns: " + ", ".join(sorted(set(bad))) + ".")

    train_count = _count_field(audit, ("train_feature_count", "n_train_features", "training_feature_count"))
    if train_count is None and final_columns:
        train_count = len(final_columns)
    if train_count is not None and final_columns and train_count != len(final_columns):
        reasons.append(
            f"preprocessing_audit feature count is inconsistent: train_feature_count={train_count}, "
            f"but final/model feature column list has {len(final_columns)} columns."
        )
    anchor_path = os.getenv("YIELD_ANCHOR_PATH")
    has_anchor = bool(anchor_path and Path(anchor_path).exists())
    anchor_count = _count_field(audit, ("anchor_feature_count", "n_anchor_features", "validation_anchor_feature_count"))
    if train_count is None:
        reasons.append("preprocessing_audit must include train_feature_count or final feature columns.")
    if has_anchor:
        if anchor_count is None:
            reasons.append("preprocessing_audit must include anchor_feature_count when anchor validation exists.")
        elif train_count is not None and anchor_count != train_count:
            reasons.append(
                f"Train/anchor feature schema mismatch: train_feature_count={train_count}, anchor_feature_count={anchor_count}."
            )
        aligned = _bool_field(
            audit,
            (
                "anchor_aligned_to_train_schema",
                "anchor_reindexed_to_train_columns",
                "anchor_schema_matches_train",
            ),
        )
        if aligned is not True:
            reasons.append("preprocessing_audit must set anchor_aligned_to_train_schema=true when anchor validation exists.")

    fold_local = _bool_field(
        audit,
        (
            "fold_local_preprocessing",
            "fit_preprocessing_inside_folds",
            "preprocessing_fit_inside_folds",
            "fold_local_feature_engineering",
        ),
    )
    if fold_local is not True:
        reasons.append("preprocessing_audit must set fold_local_preprocessing=true to show OOF preprocessing is fit inside each training fold.")

    fit_scope = str(
        audit.get("final_model_fit_scope")
        or audit.get("feature_schema_fit_scope")
        or audit.get("preprocessing_fit_scope")
        or ""
    ).strip().lower()
    if not fit_scope:
        reasons.append("preprocessing_audit must include final_model_fit_scope or preprocessing_fit_scope.")
    forbidden_fit_scope_patterns = (
        "fit on anchor",
        "fit_on_anchor",
        "fit anchor",
        "fit with anchor",
        "anchor+train",
        "train+anchor",
        "training+anchor",
        "anchor and training",
        "include anchor in fit",
        "including anchor in fit",
        "fit on test",
        "fit_on_test",
        "train+test",
        "test+train",
    )
    if any(pattern in fit_scope for pattern in forbidden_fit_scope_patterns):
        reasons.append("preprocessing_audit fit scope must not include anchor/test data.")
    elif not any(token in fit_scope for token in ("train", "fold", "training")):
        reasons.append("preprocessing_audit fit scope must state that preprocessing was fit on training folds/full training data only.")

    leakage_excluded = _list_field(audit, ("leakage_columns_excluded", "forbidden_columns_excluded", "dropped_leakage_columns"))
    if not leakage_excluded:
        reasons.append("preprocessing_audit must include leakage_columns_excluded.")
    return reasons


def _source_uses_reference_columns(source_code: str) -> str | None:
    """Detect leakage/reference columns used through dataframe indexing.

    Mentioning a forbidden column in an exclusion list is fine; reading it from a
    dataframe to construct a feature is not. This catches the common generated
    pattern df["phi_max_eff_reference"] -> derived mechanism feature.
    """
    forbidden = [
        "phi_max",
        "phi_max_eff",
        "phi_max_eff_reference",
        "m1_true",
        "m1_lf",
    ]
    hits: list[str] = []
    for col in forbidden:
        pattern = rf"\b[a-zA-Z_][a-zA-Z0-9_]*\s*\[\s*['\"]{re.escape(col)}['\"]\s*\]"
        if re.search(pattern, source_code or ""):
            hits.append(col)
    generic_reference = re.findall(
        r"\b[a-zA-Z_][a-zA-Z0-9_]*\s*\[\s*['\"]([A-Za-z0-9_]+_reference)['\"]\s*\]",
        source_code or "",
    )
    hits.extend(generic_reference)
    return ", ".join(sorted(set(hits))) if hits else None


def _source_uses_reference_dot_access(source_code: str) -> str | None:
    forbidden = [
        "phi_max",
        "phi_max_eff",
        "phi_max_eff_reference",
        "m1_true",
        "m1_lf",
    ]
    hits: list[str] = []
    for col in forbidden:
        pattern = rf"\b[a-zA-Z_][a-zA-Z0-9_]*\.{re.escape(col)}\b"
        if re.search(pattern, source_code or ""):
            hits.append(col)
    generic_reference = re.findall(
        r"\b[a-zA-Z_][a-zA-Z0-9_]*\.([A-Za-z0-9_]+_reference)\b",
        source_code or "",
    )
    hits.extend(generic_reference)
    return ", ".join(sorted(set(hits))) if hits else None


def _source_uses_unsupported_hb_loss(source_code: str) -> bool:
    lowered = (source_code or "").lower()
    if not any(token in lowered for token in ("herschel", "bingham", "hb_", "hb loss", "hb physics")):
        return False
    loss_pattern = re.compile(
        r"(def\s+\w*(?:hb|herschel|bingham|physics|constitutive)\w*loss\b|"
        r"\b(?:hb|herschel|bingham|physics|constitutive)_loss\s*=)",
        flags=re.IGNORECASE,
    )
    if not loss_pattern.search(source_code or ""):
        return False
    shear_terms = ("gamma_dot", "shear_rate", "shear rate", "flow_curve", "flow curve", "tau_vs_gamma")
    return not any(term in lowered for term in shear_terms)


def preflight_yield_source_reasons(source_code: str) -> list[str]:
    """Static checks for generated yield code before expensive execution."""
    reasons: list[str] = []
    source_code = source_code or ""
    lowered = source_code.lower()

    if re.search(r"^\s*def\s+load_yield_dataframe\s*\(", source_code, flags=re.MULTILINE):
        reasons.append("Generated code redefines load_yield_dataframe; import and use knowledge.yield_schema.load_yield_dataframe instead.")
    if re.search(r"^\s*def\s+run_fixed_yield_baselines\s*\(", source_code, flags=re.MULTILINE):
        reasons.append("Generated code redefines run_fixed_yield_baselines; import and use knowledge.yield_baselines.run_fixed_yield_baselines instead.")

    direct_optional_imports = [
        "import lightgbm",
        "from lightgbm",
        "import xgboost",
        "from xgboost",
        "import catboost",
        "from catboost",
    ]
    if any(token in lowered for token in direct_optional_imports):
        reasons.append("Generated code must not import lightgbm/xgboost/catboost directly; use sklearn fallback or guarded optional imports.")

    leaked_index_cols = _source_uses_reference_columns(source_code)
    leaked_dot_cols = _source_uses_reference_dot_access(source_code)
    leaked_cols = sorted(
        {
            item.strip()
            for value in (leaked_index_cols, leaked_dot_cols)
            if value
            for item in value.split(",")
            if item.strip()
        }
    )
    if leaked_cols:
        reasons.append(
            "Generated code reads leakage/reference columns as inputs or derived features: "
            f"{', '.join(leaked_cols)}."
        )

    if "histgradientboostingregressor" in lowered and ".feature_importances_" in lowered:
        reasons.append("HistGradientBoostingRegressor has no feature_importances_ attribute in sklearn; use permutation_importance or skip this artifact.")

    if "pd.get_dummies" in lowered and "reindex(columns=" not in lowered and ".align(" not in lowered:
        reasons.append(
            "Generated code uses pd.get_dummies without reindexing validation/anchor features to the training dummy columns; use a fitted encoder or reindex(columns=train_columns, fill_value=0)."
        )

    if _source_uses_unsupported_hb_loss(source_code):
        reasons.append("Herschel-Bulkley/Bingham physics loss requires shear-rate or flow-curve data; reject it or implement only a clearly labeled proxy.")

    return reasons


def _external_search_snippets(run_path: Path) -> list[dict[str, Any]]:
    search_path = run_path / "search_report.json"
    try:
        payload = json.loads(search_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    snippets = payload.get("snippets", []) if isinstance(payload, dict) else []
    return [item for item in snippets if isinstance(item, dict)]


def _mechanism_sources_match_search(report: dict[str, Any], snippets: list[dict[str, Any]]) -> bool:
    if not snippets or report.get("mechanism_used") is not True:
        return True
    valid_ids = {
        str(item.get("source_id", item.get("id", ""))).strip()
        for item in snippets
        if str(item.get("source_id", item.get("id", ""))).strip()
    }
    titles = [str(item.get("title") or "").strip().lower() for item in snippets]
    links = [str(item.get("link") or item.get("url") or item.get("doi") or "").strip().lower() for item in snippets]

    def cited_ids(mechanism: dict[str, Any]) -> set[str]:
        ids: set[str] = set()
        for key in ("source_ids", "source_id", "source_ids_or_links"):
            value = mechanism.get(key)
            if isinstance(value, (list, tuple, set)):
                ids.update(str(item).strip() for item in value if str(item).strip())
            elif value not in (None, ""):
                ids.add(str(value).strip())
        evidence = mechanism.get("source_evidence")
        if isinstance(evidence, list):
            for entry in evidence:
                if isinstance(entry, dict):
                    value = entry.get("source_id", entry.get("id"))
                    if value not in (None, ""):
                        ids.add(str(value).strip())
        return ids

    def cited_text(mechanism: dict[str, Any]) -> str:
        chunks = [str(mechanism.get("paper_or_source") or "")]
        for key in ("source_evidence", "sources", "references"):
            value = mechanism.get(key)
            if isinstance(value, str):
                chunks.append(value)
            elif isinstance(value, list):
                for entry in value:
                    if isinstance(entry, dict):
                        chunks.extend(str(entry.get(field) or "") for field in ("title", "link", "url", "doi", "source", "provider"))
                    else:
                        chunks.append(str(entry))
        return " ".join(chunks).strip().lower()

    for item in report.get("mechanisms", []):
        ids = cited_ids(item)
        if ids and ids.intersection(valid_ids):
            continue
        source = cited_text(item)
        if "http://" in source or "https://" in source or "doi" in source:
            if any(link and link in source for link in links):
                continue
            dois = [str(s.get("doi") or "").strip().lower() for s in snippets]
            if "doi" in source and any(doi and doi in source for doi in dois):
                continue
        if any(link and link in source for link in links):
            continue
        if any(title and (title in source or source in title) for title in titles):
            continue
        return False
    return True


def _repair_guidance(reasons: list[str]) -> str:
    text = "\n".join(reasons).lower()
    guidance = ["Patch the existing generated script with the smallest targeted change."]
    if "schema" in text or "load_yield_dataframe" in text:
        guidance.append("Use knowledge.yield_schema.load_yield_dataframe and its normalized yield_stress target.")
    if "run_fixed_yield_baselines" in text or "baseline" in text:
        guidance.append("Use knowledge.yield_baselines.run_fixed_yield_baselines and save its baseline_results alongside generated model results.")
    if "attributeerror: 'str' object has no attribute 'get'" in text or "baseline_results" in text:
        guidance.append("Treat run_fixed_yield_baselines output as a dict-like bundle. Read fixed rows from baseline_bundle['baseline_results']; do not iterate over the top-level dict.")
    if "redefines" in text or "static check" in text:
        guidance.append("Do not shadow fixed project helpers. Import load_yield_dataframe and run_fixed_yield_baselines from knowledge/ and call them directly.")
    if "feature_importances_" in text:
        guidance.append("HistGradientBoostingRegressor does not expose feature_importances_; use sklearn.inspection.permutation_importance or omit feature importance.")
    if "lightgbm" in text or "xgboost" in text or "catboost" in text or "libomp" in text:
        guidance.append("Do not import lightgbm/xgboost/catboost at top level; use sklearn HistGradientBoosting/RandomForest/GaussianProcess/MLP fallback or wrap optional imports in try/except Exception/OSError.")
    if "leakage" in text or "reference columns" in text:
        guidance.append("Drop Tau0_Pa, tau_Pa, yield_stress, phi_max, m1_true, and m1_lf before feature construction.")
    if "herschel" in text or "bingham" in text:
        guidance.append("Only use Herschel-Bulkley/Bingham as a strict physics loss when shear-rate or flow-curve columns exist; otherwise reject it or label it as a proxy and prefer packing/fmax-style mechanisms.")
    if "mechanism" in text:
        guidance.append("Create logs/mechanism_report.json and list every used mechanism with paper/source, exact formula, source_ids, and source_evidence from search_report.json; or explicitly declare no mechanism used.")
    if "yodel" in text or "packing" in text or "fmax" in text:
        guidance.append("Use the selected YODEL/packing/fmax strategy as a leakage-safe feature/baseline/residual layer, or add rejected_yodel_packing_candidate with a concrete reason in logs/mechanism_report.json.")
    if "prediction" in text or "negative" in text:
        guidance.append("Save predictions/yield_predictions.csv and clip final yield_stress_predicted to non-negative finite values.")
    if "metrics" in text:
        guidance.append("Save finite R2/RMSE/MAE/MAPE values in metrics/metrics.json.")
    if "mape" in text:
        guidance.append("Report every MAPE value as a percentage, matching fixed baselines; multiply sklearn mean_absolute_percentage_error outputs by 100 before saving.")
    if "selection_metric" in text or "best_baseline_metric" in text or "best_research_metric" in text:
        guidance.append("Make selection_metric, selection_metric_direction, best_baseline_metric, and best_research_metric use one consistent metric. For selection_metric='oof_rmse', best_*_metric must be RMSE values, not R2.")
    if "mechanism_effect_size" in text or "claim_strength" in text or "below 1%" in text:
        guidance.append("Save mechanism_effect_size with relative_rmse_improvement as a fraction, relative_rmse_improvement_percent as percent, and claim_strength. Use claim_strength='marginal' when RMSE improvement is below 1%.")
    if "benchmark_alignment" in text or "selected_hybrid_strategy" in text:
        guidance.append("Record selected_hybrid_strategy and benchmark_alignment in metrics or mechanism_report. Explain any deviation from the benchmark-selected CandidateAgent strategy.")
    if "implementation_validation" in text or "generalization_gap" in text:
        guidance.append("Save implementation_validation comparing benchmark_selected_strategy, benchmark_oof_rmse, actual_oof_rmse, benchmark_anchor_r2, actual_anchor_r2, generalization_gap, and validation_status. Mark degraded/failed_generalization when anchor R2 is negative or actual metrics are materially worse.")
    if "benchmark_selected_proxy_result" in text or "benchmark proxy model" in text or "benchmark feature mode" in text:
        guidance.append("Actually evaluate the benchmark-selected proxy as one candidate and save benchmark_selected_proxy_result with strategy_id, proxy_model, feature_mode, OOF RMSE/R2/MAPE, and anchor R2/RMSE/MAPE when anchor exists. Do not replace it only with a different model family.")
    if "json serializable" in text or "bool_" in text or "float32" in text or "int64" in text:
        guidance.append("Add a recursive to_jsonable helper for numpy/pandas types and call json.dump(to_jsonable(metrics), f, indent=2) for every JSON artifact.")
    if "baseline" in text or "candidate" in text:
        guidance.append("Evaluate and record baseline_results/candidate_models for DummyRegressor, linear/Ridge, unconstrained tree, and monotonic HistGradientBoosting when available.")
    if "beats_best_baseline" in text or "underperforms_baseline" in text or "best fixed baseline" in text:
        guidance.append("Do not select a fixed baseline as the research model. Save best_baseline, best_baseline_metric, best_research_model, best_research_metric, beats_best_baseline, and research_model_status; mark underperforms_baseline when the generated model is weaker.")
    if "planned_model" in text or "actual_model" in text or "fallback_reason" in text:
        guidance.append("Save planned_model and actual_model in metrics or mechanism_report. If they differ, add a concrete fallback_reason or implement the planned model family.")
    if "mechanism_ablation" in text or "without_mechanism" in text:
        guidance.append("When mechanism_used=true, evaluate the same model family with and without mechanism features/losses under the same OOF protocol and save mechanism_ablation with delta_rmse, delta_r2, and mechanism_improves_metric.")
    if "preprocessing_audit" in text or "feature schema" in text or "train/anchor" in text or "get_dummies" in text:
        guidance.append("Save preprocessing/preprocessing_audit.json and use one fitted feature schema: fit encoders/scalers/mechanism-feature parameters inside each train fold, then transform validation/anchor by reindexing or encoder.transform so feature counts match.")
    if "research_confidence" in text:
        guidance.append("Set research_confidence from anchor validation and baseline comparison: low when anchor R2 < 0 or the research model underperforms baseline.")
    if "synthetic" in text:
        guidance.append("When using synthetic training data, save logs/synthetic_data_report.json and include source paper, formulas, assumptions, and limitations.")
    if "anchor" in text:
        guidance.append("If YIELD_ANCHOR_PATH exists, evaluate the final model on that anchor CSV and save anchor_validation metrics in metrics/metrics.json.")
    if "oof" in text:
        guidance.append("Run 5-fold OOF on the active training dataset and record primary_evaluation='5-fold OOF'.")
    return "\n".join(f"- {item}" for item in guidance)


def verify_yield_run(
    source_code: str,
    action_result: str,
    return_code: int,
    run_dir: str | os.PathLike[str] | None,
    started_at: float,
) -> YieldVerificationResult:
    reasons: list[str] = []          # HARD gates: correctness + anti-leakage; block the run
    soft: list[str] = []             # SOFT: metadata/consistency/reporting; recorded, non-blocking
    run_path = Path(run_dir or os.getenv("YIELD_RUN_DIR") or "agent_workspace/runs/yield_search")
    lowered_code = (source_code or "").lower()
    lowered_output = (action_result or "").lower()
    reasons.extend(preflight_yield_source_reasons(source_code or ""))

    if return_code != 0:
        reasons.append(f"Generated script exited with non-zero return code {return_code}.")
    if "yield pipeline completed successfully" not in lowered_output:
        reasons.append("Execution output did not print Yield pipeline completed successfully after artifact saves.")
    if "from knowledge.yield_schema import load_yield_dataframe" not in source_code:
        reasons.append("Generated code must import load_yield_dataframe from knowledge.yield_schema; do not redefine a local function with the same name.")
    if "from knowledge.yield_baselines import run_fixed_yield_baselines" not in source_code:
        reasons.append("Generated code must import run_fixed_yield_baselines from knowledge.yield_baselines; do not redefine fixed baselines locally.")
    direct_optional_imports = [
        "import lightgbm",
        "from lightgbm",
        "import xgboost",
        "from xgboost",
        "import catboost",
        "from catboost",
    ]
    if any(token in lowered_code for token in direct_optional_imports):
        reasons.append("Generated code must not import lightgbm/xgboost/catboost at top level; use sklearn fallback or guarded optional imports.")
    if "kfold" not in lowered_code and "cross_val" not in lowered_code and "oof" not in lowered_code:
        reasons.append("Generated code must implement 5-fold OOF evaluation on the active training dataset.")
    leakage_feature_patterns = [
        "feature_cols = [\"phi\", \"sp_percent\", \"phi_max\"]",
        "feature_cols = ['phi', 'sp_percent', 'phi_max']",
        "feature_cols = [\"phi\", \"d_s_um\", \"m1_true\"]",
        "feature_cols = ['phi', 'd_s_um', 'm1_true']",
        "feature_cols = [\"phi\", \"d_s_um\", \"m1_lf\"]",
        "feature_cols = ['phi', 'd_s_um', 'm1_lf']",
    ]
    if any(pattern in lowered_code for pattern in leakage_feature_patterns):
        reasons.append("Generated code includes auxiliary physics/reference columns in feature_cols.")
    leaked_source_cols = _source_uses_reference_columns(source_code or "")
    if leaked_source_cols:
        reasons.append(
            "Generated code reads leakage/reference columns as dataframe inputs or derived features: "
            f"{leaked_source_cols}."
        )

    metrics_path, metrics = _load_json(_fresh_files(run_path, "metrics", ["metrics.json", "*.json"], started_at))
    if metrics is None:
        reasons.append("No fresh metrics JSON artifact was found under metrics/.")
    elif not _has_finite_metric_numbers(metrics):
        reasons.append("Metrics JSON does not contain finite numeric values.")
    elif _candidate_model_count(metrics) < 3:
        reasons.append("Metrics JSON must include baseline comparison for at least three evaluated models/baselines.")
    else:
        primary = str(metrics.get("primary_evaluation") or metrics.get("evaluation_protocol") or "").lower()
        if "oof" not in primary and "5-fold" not in primary and "5 fold" not in primary:
            soft.append("Metrics JSON should record primary_evaluation='5-fold OOF' or an equivalent 5-fold OOF evaluation_protocol.")
        anchor_path = os.getenv("YIELD_ANCHOR_PATH")
        if anchor_path and Path(anchor_path).exists():
            secondary = str(metrics.get("secondary_evaluation") or "").lower()
            if "anchor" not in secondary:
                soft.append("Metrics JSON should record secondary_evaluation='anchor_validation' when YIELD_ANCHOR_PATH exists.")
            has_anchor_metrics = any(
                key in metrics
                for key in ("anchor_validation", "anchor_validation_metrics", "anchor_metrics", "fixed_test_metrics")
            ) or any(key in metrics for key in ("anchor_rmse", "anchor_mae", "anchor_r2", "anchor_mape"))
            if not has_anchor_metrics:
                soft.append("Metrics JSON should include anchor_validation metrics when YIELD_ANCHOR_PATH exists.")

    pred_files = _fresh_files(run_path, "predictions", ["*.csv"], started_at)
    pred_files.extend(_fresh_files(run_path, "reports", ["*pred*.csv", "*prediction*.csv"], started_at))
    if not _prediction_file_ok(pred_files):
        reasons.append("No fresh prediction CSV with non-negative finite yield_stress_predicted was found.")

    model_files = _fresh_files(run_path, "trained_models", ["*.pt", "*.pth", "*.pkl", "*.joblib"], started_at)
    if not model_files:
        reasons.append("No fresh model artifact was found under trained_models/.")

    pre_files = _fresh_files(run_path, "preprocessing", ["*.json", "*.pkl", "*.joblib"], started_at)
    if not pre_files:
        reasons.append("No fresh preprocessing artifact was found under preprocessing/.")
    else:
        leaked = _preprocessing_uses_leakage(pre_files)
        if leaked:
            reasons.append(f"Preprocessing feature_columns include leakage/reference columns: {leaked}.")
        soft.extend(_preprocessing_audit_reasons(run_path, metrics))

    mech_path, mech = _load_json(_fresh_files(run_path, "logs", ["mechanism_report.json"], started_at))
    if mech is None:
        reasons.append("No fresh logs/mechanism_report.json artifact was found.")
    elif not _mechanism_report_ok(mech):
        soft.append("Mechanism report is incomplete; every used mechanism should list paper/source, formula, columns, role, and applicability.")
    elif not _mechanism_sources_match_search(mech, _external_search_snippets(run_path)):
        soft.append(
            "External search returned snippets, but a used mechanism does not cite a searched title/link/DOI/source."
        )
    # Bucket-2 (metadata / cross-file consistency / reporting) → SOFT warnings.
    # These do not affect whether the produced model is correct or leakage-free;
    # requiring the LLM to hand-write all of them self-consistently was the main
    # cause of OperationAgent never passing within the attempt budget.
    soft.extend(_yodel_decision_reasons(run_path, mech))
    if metrics is not None:
        soft.extend(_mape_unit_reasons(metrics))
        soft.extend(_selection_metric_consistency_reasons(metrics))
        soft.extend(_baseline_audit_reasons(metrics))
        soft.extend(_planned_actual_audit_reasons(metrics, mech))
        soft.extend(_research_confidence_reasons(metrics))
        soft.extend(_anchor_generalization_reasons(metrics))
        if mech is not None:
            soft.extend(_mechanism_ablation_reasons(metrics, mech))
        soft.extend(_candidate_alignment_reasons(run_path, metrics, mech))
        soft.extend(_implementation_validation_reasons(run_path, metrics, mech))

    synthetic_report_path = os.getenv("YIELD_SYNTHETIC_REPORT_PATH")
    if synthetic_report_path and Path(synthetic_report_path).exists():
        report_paths = _fresh_files(run_path, "logs", ["synthetic_data_report.json"], started_at)
        if not report_paths:
            existing = run_path / "logs" / "synthetic_data_report.json"
            if not existing.exists() or existing.stat().st_size == 0:
                soft.append("Synthetic training data are active, but no logs/synthetic_data_report.json artifact was found.")

    def _dedupe(items: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for item in items:
            if item in seen:
                continue
            seen.add(item)
            out.append(item)
        return out

    reasons = _dedupe(reasons)
    soft = _dedupe(soft)

    # Best-effort: persist the tiered guardrail outcome for audit / paper evidence.
    try:
        report = {
            "passed": not reasons,
            "hard_reasons": reasons,
            "soft_warnings": soft,
            "policy": "hard=correctness+anti-leakage block; soft=metadata/consistency/reporting recorded only",
        }
        logs_dir = run_path / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        (logs_dir / "yield_verification_audit.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception:
        pass

    if reasons:
        return YieldVerificationResult(False, reasons, _repair_guidance(reasons), soft)
    return YieldVerificationResult(True, [], "", soft)


def verify_yield_plugin_harness_run(
    run_dir: str | None,
    *,
    anchor_expected: bool = False,
    started_at: float | None = None,
) -> YieldVerificationResult:
    """Lightweight acceptance check for the structured plugin harness path.

    Unlike verify_yield_run (built for freeform full scripts), this does NOT
    inspect source imports or in-script CV — the deterministic harness owns
    those. It only confirms the harness produced fresh, self-consistent
    artifacts and honoured the anchor / fallback bookkeeping rules.
    """
    reasons: list[str] = []
    warnings: list[str] = []
    repair = (
        "The structured plugin harness must write fresh metrics/predictions/model/"
        "preprocessing/mechanism artifacts, expose plugin_candidate_result + "
        "reference_arms + benchmark_selected_proxy_result + implementation_validation "
        "+ research_confidence, and record plugin_error + candidate_source="
        "plugin_reference_fallback whenever the LLM plugin was not used."
    )

    if not run_dir:
        return YieldVerificationResult(False, ["Plugin harness run_dir was not provided."], repair, warnings)
    root = Path(run_dir)

    def _fresh(path: Path) -> bool:
        return started_at is None or path.stat().st_mtime + 1.0 >= float(started_at)

    # (1) fresh artifacts exist
    required = {
        "metrics/metrics.json": root / "metrics" / "metrics.json",
        "predictions/yield_predictions.csv": root / "predictions" / "yield_predictions.csv",
        "preprocessing/preprocessing_audit.json": root / "preprocessing" / "preprocessing_audit.json",
        "logs/mechanism_report.json": root / "logs" / "mechanism_report.json",
    }
    for label, path in required.items():
        if not path.exists():
            reasons.append(f"Plugin harness did not write {label}.")
        elif not _fresh(path):
            reasons.append(f"{label} is stale (not written during this run).")
    model_dir = root / "trained_models"
    model_files = list(model_dir.glob("*.joblib")) if model_dir.exists() else []
    if not model_files:
        reasons.append("Plugin harness did not write a model artifact under trained_models/*.joblib.")
    elif not any(_fresh(p) for p in model_files):
        reasons.append("Model artifact under trained_models/ is stale (not written during this run).")

    metrics_path = required["metrics/metrics.json"]
    metrics: Any = None
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception as exc:
            reasons.append(f"metrics.json could not be parsed: {exc}.")
    if not isinstance(metrics, dict):
        reasons.append("metrics.json is missing or not a JSON object.")
        return YieldVerificationResult(False, reasons, repair, warnings)

    # (2) plugin result fields present
    for fld in ("plugin_candidate_result", "reference_arms", "benchmark_selected_proxy_result",
                "implementation_validation", "research_confidence"):
        if fld not in metrics:
            reasons.append(f"metrics.json missing required plugin field '{fld}'.")

    candidate_source = str(metrics.get("candidate_source") or "")
    is_fallback = candidate_source == "plugin_reference_fallback"
    if candidate_source not in ("structured_plugin_harness", "plugin_reference_fallback"):
        reasons.append(f"Unexpected candidate_source '{candidate_source}' for the plugin harness path.")
    if "plugin_candidate_result" in metrics and metrics.get("plugin_candidate_result") is None and not is_fallback:
        reasons.append("plugin_candidate_result is null but candidate_source is not a reference fallback.")

    # (3) fallback bookkeeping
    if is_fallback and not metrics.get("plugin_error"):
        reasons.append("candidate_source=plugin_reference_fallback requires a recorded plugin_error.")

    # (3) anchor rules
    anchor = metrics.get("anchor_validation") if isinstance(metrics.get("anchor_validation"), dict) else None
    if anchor_expected:
        if not anchor:
            reasons.append("Anchor data was provided but metrics.json has no anchor_validation.")
        else:
            anchor_r2 = anchor.get("r2")
            if isinstance(anchor_r2, (int, float)) and not isinstance(anchor_r2, bool) and float(anchor_r2) < 0:
                if str(metrics.get("research_confidence") or "").lower() == "high":
                    reasons.append("Anchor R2 < 0 must not be reported with research_confidence=high.")
                iv = metrics.get("implementation_validation") if isinstance(metrics.get("implementation_validation"), dict) else {}
                if str(iv.get("validation_status") or "") == "aligned_success":
                    reasons.append("Anchor R2 < 0 must not be reported as validation_status=aligned_success.")

    # (4) prediction columns
    pred_path = required["predictions/yield_predictions.csv"]
    if pred_path.exists():
        try:
            with open(pred_path, newline="", encoding="utf-8") as f:
                header = next(csv.reader(f), [])
        except Exception as exc:
            header = []
            warnings.append(f"Could not read prediction header: {exc}.")
        if "yield_stress_predicted" not in header:
            reasons.append("predictions CSV must include a yield_stress_predicted column.")

    passed = not reasons
    return YieldVerificationResult(passed, reasons, "" if passed else repair, warnings)
