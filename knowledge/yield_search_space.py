"""Search-space coverage map for the yield-stress AutoML project.

This module is deliberately separate from the scorer. Its job is to keep the
project honest about what the AutoML search space *claims* to cover, what the
current harness can actually execute, and what remains planned-only or
data-blocked. It prevents ad-hoc "add one more route" drift.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from knowledge.yield_executor_graphs import (
    ALLOWED_GRAPH_OPS,
    EXECUTABLE_GRAPH_MODES,
)
from knowledge.yield_joint_search import hparam_grid
from knowledge.yield_schema import TARGET_COLUMN, load_yield_dataframe


ENGINEERED_FEATURE_REPORT = Path(
    "agent_workspace/data/generated_yield_stress/generated_yield_stress_feature_report.json"
)


SEARCH_SPACE_AXES: dict[str, list[dict[str, Any]]] = {
    "data": [
        {
            "id": "real_hf",
            "name": "Real high-fidelity data",
            "meaning": "真实高保真屈服值数据，用于最终可信评估。",
            "required_for": ["holdout", "multi_fidelity_base_residual", "paper_claims"],
        },
        {
            "id": "augmented_hf",
            "name": "Generated/augmented high-fidelity-like data",
            "meaning": "当前 generated_yield_stress_data.xlsx 归一化/特征工程后的数据。",
            "required_for": ["structure_exploration", "executor_smoke"],
        },
        {
            "id": "low_fidelity",
            "name": "Low-fidelity/synthetic data",
            "meaning": "低保真或合成数据，只能用于结构预训练/探索，不能单独支撑真实泛化结论。",
            "required_for": ["multi_fidelity_base_residual"],
        },
        {
            "id": "anchor_holdout",
            "name": "Anchor/holdout validation",
            "meaning": "外部一致性或留出验证，用于防止只看 OOF。",
            "required_for": ["guardrail"],
        },
    ],
    "feature": [
        {"id": "raw_numeric", "name": "Raw numeric features"},
        {"id": "engineered_physics", "name": "Physics-inspired engineered features"},
        {"id": "thermal_pressure", "name": "Thermal/pressure feature group"},
        {"id": "composition", "name": "Composition/gradation feature group"},
        {"id": "mixing_history", "name": "Mixing/process history feature group"},
    ],
    "mechanism": [
        {"id": "yodel", "name": "YODEL/packing-style mechanism"},
        {"id": "lian_packing", "name": "Lian-style packing relation"},
        {"id": "process_proxy", "name": "Thermal/mixing/process proxy mechanism"},
        {"id": "llm_dynamic_shape", "name": "LLM-generated executable mechanism shape"},
        {"id": "constitutive_flow_curve", "name": "HB/Bingham/Casson flow-curve mechanism"},
    ],
    "target": [
        {"id": "yield_stress", "name": "Direct yield-stress prediction"},
        {"id": "residual", "name": "Residual prediction over a mechanism base"},
        {"id": "latent_phi_m", "name": "Latent effective packing parameter phi_m"},
        {"id": "latent_m1_eff", "name": "Latent/effective scale parameter m1_eff"},
        {"id": "latent_structure_factor", "name": "Latent process/structure state"},
    ],
    "model": [
        {"id": "rf", "name": "Random Forest"},
        {"id": "hgb", "name": "Histogram Gradient Boosting"},
        {"id": "kernel", "name": "Kernel/Gaussian Process"},
        {"id": "ridge", "name": "Ridge/linear"},
        {"id": "svr", "name": "Support Vector Regression"},
        {"id": "mlp", "name": "Generic MLP"},
        {"id": "pinn", "name": "Torch-free physics-residual MLP wrapper"},
        {"id": "mlp2_hidden16", "name": "Reference latent MLP 2-layer hidden=16"},
        {"id": "mlp2_hidden64", "name": "Reference latent MLP 2-layer hidden=64"},
        {"id": "mlp3_hidden16", "name": "Reference latent MLP 3-layer hidden=16"},
        {"id": "branched_hidden16", "name": "Reference grouped three-branch latent MLP"},
        {"id": "llm_model_factory", "name": "LLM-generated bounded estimator factory"},
    ],
    "fusion_training": [
        {"id": "raw_ml", "name": "Raw features to model"},
        {"id": "mechanism_features", "name": "Mechanism-derived features to model"},
        {"id": "mechanism_residual", "name": "Mechanism base plus learned residual"},
        {"id": "two_stage_inverse_grid", "name": "Fold-local latent inversion then X-to-latent model"},
        {"id": "end_to_end_physics_layer", "name": "Model outputs latent; physics layer computes y and y-loss trains model"},
        {"id": "end_to_end_physics_layer_residual", "name": "Physics-layer output plus learned residual correction"},
        {"id": "multi_fidelity_base_residual", "name": "Low-fidelity base plus high-fidelity residual calibration"},
        {"id": "physics_loss_regularizer", "name": "PINN/physics-loss regularization"},
    ],
    "executor_graph": [
        {"id": "x_to_y", "ops": ["select_raw_features", "fit_model", "predict_model"]},
        {
            "id": "x_plus_mechanism_features_to_y",
            "ops": ["fit_mechanism_params", "compute_mechanism_features", "fit_model", "predict_model"],
        },
        {
            "id": "mechanism_base_plus_residual",
            "ops": ["fit_mechanism_params", "compute_mechanism_base", "fit_residual_model", "predict_residual_model", "add_predictions"],
        },
        {
            "id": "x_to_latent_to_physics_output",
            "ops": ["fit_mechanism_params", "infer_latent_targets", "fit_latent_model", "predict_latent_model", "compute_physics_output"],
        },
        {
            "id": "x_to_latent_to_physics_output_plus_residual",
            "ops": ["fit_mechanism_params", "infer_latent_targets", "fit_latent_model", "predict_latent_model", "compute_physics_output", "fit_residual_model", "predict_residual_model", "add_predictions"],
        },
        {
            "id": "end_to_end_latent_physics_layer",
            "ops": ["fit_latent_physics_model", "predict_latent_physics_model", "compute_physics_output"],
        },
        {
            "id": "multi_fidelity_base_residual",
            "ops": ["fit_low_fidelity_base", "fit_high_fidelity_residual", "predict_model", "add_predictions"],
        },
    ],
}


REFERENCE_ROUTES: list[dict[str, Any]] = [
    {
        "id": "raw_ml_baseline",
        "name": "Raw ML baseline",
        "axes": {
            "feature": ["raw_numeric", "engineered_physics"],
            "target": "yield_stress",
            "model": ["rf", "hgb", "kernel", "ridge", "svr", "mlp", "llm_model_factory"],
            "fusion_training": "raw_ml",
            "executor_graph": "x_to_y",
        },
        "expected_status": "executable",
        "why_it_matters": "所有机理路线必须打败或解释无法打败 raw baseline。",
    },
    {
        "id": "mechanism_feature_augmented_ml",
        "name": "Mechanism features + ML",
        "axes": {
            "mechanism": ["yodel", "lian_packing", "process_proxy", "llm_dynamic_shape"],
            "feature": ["raw_numeric", "engineered_physics"],
            "target": "yield_stress",
            "model": ["rf", "hgb", "kernel", "ridge", "svr", "mlp", "llm_model_factory"],
            "fusion_training": "mechanism_features",
            "executor_graph": "x_plus_mechanism_features_to_y",
        },
        "expected_status": "executable",
        "why_it_matters": "当前已跑通的机理增强路线。",
    },
    {
        "id": "mechanism_residual_ml",
        "name": "Mechanism base + residual ML",
        "axes": {
            "mechanism": ["yodel", "lian_packing", "process_proxy", "llm_dynamic_shape"],
            "feature": ["raw_numeric", "engineered_physics"],
            "target": "residual",
            "model": ["rf", "hgb", "kernel", "ridge", "svr", "mlp", "pinn", "llm_model_factory"],
            "fusion_training": "mechanism_residual",
            "executor_graph": "mechanism_base_plus_residual",
        },
        "expected_status": "executable",
        "why_it_matters": "检验机理是否能解释稳定主项，ML 只补误差。",
    },
    {
        "id": "two_stage_latent_inverse",
        "name": "Two-stage latent inversion",
        "axes": {
            "mechanism": ["yodel", "lian_packing", "process_proxy"],
            "target": ["latent_phi_m", "latent_m1_eff", "latent_structure_factor"],
            "model": ["rf", "hgb", "kernel", "ridge", "mlp", "llm_model_factory"],
            "fusion_training": "two_stage_inverse_grid",
            "executor_graph": "x_to_latent_to_physics_output",
        },
        "expected_status": "planned_only",
        "why_it_matters": "稳健、可审计的反演 baseline；不能代表全部反演路线。",
    },
    {
        "id": "end_to_end_latent_physics_layer",
        "name": "End-to-end latent physics layer",
        "axes": {
            "mechanism": ["yodel", "lian_packing", "process_proxy"],
            "target": ["latent_m1_eff", "latent_phi_m", "latent_structure_factor"],
            "model": ["mlp2_hidden16", "mlp2_hidden64", "mlp3_hidden16", "branched_hidden16", "llm_model_factory"],
            "fusion_training": "end_to_end_physics_layer",
            "executor_graph": "end_to_end_latent_physics_layer",
        },
        "expected_status": "executable",
        "why_it_matures_project": "覆盖陈同学 A/B/C/D 启发：不是只做网格反演，也搜索端到端物理层。",
    },
    {
        "id": "latent_physics_plus_residual",
        "name": "Latent physics layer + residual correction",
        "axes": {
            "mechanism": ["yodel", "lian_packing", "process_proxy"],
            "target": ["latent_m1_eff", "latent_phi_m", "residual"],
            "model": ["rf", "hgb", "kernel", "mlp", "branched_hidden16", "llm_model_factory"],
            "fusion_training": "end_to_end_physics_layer_residual",
            "executor_graph": "x_to_latent_to_physics_output_plus_residual",
        },
        "expected_status": "planned_only",
        "why_it_matters": "把物理层输出和数据驱动修正拆开，适合小样本纠偏。",
    },
    {
        "id": "multi_fidelity_base_residual",
        "name": "Multi-fidelity base + HF residual",
        "axes": {
            "data": ["low_fidelity", "real_hf"],
            "feature": ["raw_numeric", "engineered_physics"],
            "target": ["yield_stress", "residual"],
            "model": ["mlp", "branched_hidden16", "rf", "hgb"],
            "fusion_training": "multi_fidelity_base_residual",
            "executor_graph": "multi_fidelity_base_residual",
        },
        "expected_status": "missing_data_or_planned",
        "why_it_matters": "覆盖粘度项目多保真启发；需要清楚 LF/HF 标记和 executor。",
    },
    {
        "id": "physics_loss_regularized_model",
        "name": "Physics-loss/PINN-style regularization",
        "axes": {
            "mechanism": ["constitutive_flow_curve", "yodel", "lian_packing"],
            "target": "yield_stress",
            "model": ["pinn", "mlp", "llm_model_factory"],
            "fusion_training": "physics_loss_regularizer",
        },
        "expected_status": "missing_data_or_planned",
        "why_it_matters": "覆盖 PINN 方向；当前缺剪切曲线/明确可微损失时不能强跑。",
    },
]


def _load_feature_groups() -> dict[str, list[str]]:
    if not ENGINEERED_FEATURE_REPORT.exists():
        return {}
    try:
        payload = json.loads(ENGINEERED_FEATURE_REPORT.read_text(encoding="utf-8"))
    except Exception:
        return {}
    groups = payload.get("feature_groups") if isinstance(payload, dict) else {}
    return groups if isinstance(groups, dict) else {}


def _dataset_capabilities(data_path: str | Path | None = None) -> dict[str, Any]:
    caps: dict[str, Any] = {
        "data_path": str(data_path or ""),
        "exists": bool(data_path and Path(data_path).exists()),
        "n_rows": None,
        "columns": [],
        "feature_groups": _load_feature_groups(),
        "has_target": False,
        "has_phi": False,
        "phi_varies": False,
        "has_flow_curve": False,
        "has_microstructure": False,
        "has_fidelity_label": False,
        "fidelity_values": [],
    }
    if not data_path or not Path(data_path).exists():
        return caps
    try:
        df, meta = load_yield_dataframe(data_path)
    except Exception:
        df = pd.read_csv(data_path)
        meta = {}
    cols = [str(c) for c in df.columns]
    caps["n_rows"] = int(len(df))
    caps["columns"] = cols
    caps["has_target"] = TARGET_COLUMN in df.columns
    caps["has_phi"] = "phi" in df.columns
    if "phi" in df.columns:
        phi = pd.to_numeric(df["phi"], errors="coerce")
        caps["phi_varies"] = bool(phi.nunique(dropna=True) > 1)
    flow_cols = {"shear_rate", "flow_curve", "gamma_dot"}
    micro_cols = {"d50_um", "psd_width", "specific_surface_m2kg", "ssa", "specific_surface"}
    caps["has_flow_curve"] = bool(flow_cols & set(cols))
    caps["has_microstructure"] = bool(micro_cols & set(cols))
    caps["has_fidelity_label"] = "data_fidelity" in df.columns
    if "data_fidelity" in df.columns:
        caps["fidelity_values"] = sorted(str(v) for v in df["data_fidelity"].dropna().unique().tolist())
    caps["schema_feature_count"] = len(meta.get("feature_columns", [])) if isinstance(meta, dict) else None
    return caps


def _implemented_model_families() -> list[str]:
    candidates = [
        "rf",
        "hgb",
        "kernel",
        "ridge",
        "svr",
        "mlp",
        "pinn",
        "extratrees",
        "mlp2_hidden16",
        "mlp2_hidden64",
        "mlp3_hidden16",
        "branched_hidden16",
    ]
    return [fam for fam in candidates if hparam_grid(fam)]


def _status_for_route(route: dict[str, Any], data_caps: dict[str, Any]) -> tuple[str, list[str], str]:
    route_id = str(route.get("id"))
    graph_id = str((route.get("axes") or {}).get("executor_graph") or "")
    reasons: list[str] = []

    executable_graph_ids = {
        "x_to_y": "raw_ml",
        "x_plus_mechanism_features_to_y": "mechanism_features",
        "mechanism_base_plus_residual": "mechanism_residual",
    }
    if graph_id in executable_graph_ids:
        mode = executable_graph_ids[graph_id]
        if mode in EXECUTABLE_GRAPH_MODES:
            return "executable", [f"executor_graph maps to implemented mode '{mode}'."], "current"

    if route_id == "two_stage_latent_inverse":
        if not data_caps.get("has_target"):
            return "missing_data", ["Needs train-fold target to infer latent labels."], "next_candidate"
        if not data_caps.get("has_phi"):
            reasons.append("Needs phi or another mechanism input for current packing physics_layer.")
            return "missing_data", reasons, "next_candidate"
        return "planned_only", ["Graph nodes exist, but interpreter does not yet implement latent inversion nodes."], "next_candidate"

    if route_id == "end_to_end_latent_physics_layer":
        groups = data_caps.get("feature_groups") or {}
        if not {"thermal_pressure", "composition", "mixing_history"} <= set(groups):
            reasons.append("Branched latent model needs thermal_pressure/composition/mixing_history feature groups.")
            return "missing_data_or_planned", reasons, "high_value_planned"
        return "executable", ["End-to-end m1_eff physics-layer interpreter and A/B/C/D latent architectures are implemented."], "current"

    if route_id == "latent_physics_plus_residual":
        return "planned_only", ["Needs latent physics output interpreter plus residual correction wiring."], "planned"

    if route_id == "multi_fidelity_base_residual":
        if not data_caps.get("has_fidelity_label") or len(data_caps.get("fidelity_values") or []) < 2:
            return "missing_data_or_planned", ["Needs clear LF/HF fidelity labels with at least two fidelity levels."], "planned_after_data"
        return "executable", ["LF-base + HF-residual executor graph is implemented; metrics are scoped to HF OOF."], "current"

    if route_id == "physics_loss_regularized_model":
        if not data_caps.get("has_flow_curve"):
            return "missing_data_or_planned", ["Needs shear-rate/flow-curve columns or a defensible differentiable physics residual."], "planned_after_data"
        return "planned_only", ["Needs torch/autograd physics-loss executor."], "planned_after_data"

    return "planned_only", ["Route is listed in search-space map but has no executable interpreter mapping yet."], "planned"


def audit_search_space(data_path: str | Path | None = None) -> dict[str, Any]:
    """Return a machine-readable search-space coverage report."""
    data_caps = _dataset_capabilities(data_path)
    routes: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for route in REFERENCE_ROUTES:
        status, reasons, priority = _status_for_route(route, data_caps)
        counts[status] = counts.get(status, 0) + 1
        routes.append({
            **route,
            "implementation_status": status,
            "status_reasons": reasons,
            "priority_bucket": priority,
        })

    executable = [r["id"] for r in routes if r["implementation_status"] == "executable"]
    high_value_unexecuted = [
        r["id"] for r in routes
        if r["priority_bucket"] in {"high_value_planned", "next_candidate"} and r["implementation_status"] != "executable"
    ]
    return {
        "stage": "search_space_coverage_audit",
        "purpose": "Prevent route drift by tracking executable/planned/missing-data coverage across AutoML axes.",
        "axes": SEARCH_SPACE_AXES,
        "reference_routes": routes,
        "coverage_counts": counts,
        "currently_executable_routes": executable,
        "high_value_unexecuted_routes": high_value_unexecuted,
        "implemented_executor_graph_modes": sorted(EXECUTABLE_GRAPH_MODES),
        "allowed_executor_graph_ops": sorted(ALLOWED_GRAPH_OPS),
        "implemented_model_families": _implemented_model_families(),
        "dataset_capabilities": data_caps,
        "next_recommended_work": [
            "Keep raw/mechanism_features/mechanism_residual as executable baselines.",
            "Keep two_stage_inverse_grid and end_to_end_physics_layer as distinct routes; do not collapse them into one latent route.",
            "Polish executable A/B/C/D latent architectures and compare them under the same OOF/holdout protocol.",
            "Do not claim multi-fidelity or PINN coverage until data/executor blockers are resolved.",
        ],
    }


def render_coverage_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Yield AutoML Search-Space Coverage Audit",
        "",
        "This report separates what is executable today from what is planned or blocked.",
        "",
        "## Summary",
        "",
        f"- Executable routes: {', '.join(report.get('currently_executable_routes') or []) or 'none'}",
        f"- High-value unexecuted routes: {', '.join(report.get('high_value_unexecuted_routes') or []) or 'none'}",
        f"- Coverage counts: {json.dumps(report.get('coverage_counts', {}), ensure_ascii=False)}",
        "",
        "## Dataset Capabilities",
        "",
    ]
    caps = report.get("dataset_capabilities") or {}
    for key in ("data_path", "n_rows", "has_target", "has_phi", "phi_varies", "has_flow_curve", "has_microstructure", "has_fidelity_label", "fidelity_values"):
        lines.append(f"- `{key}`: {caps.get(key)}")
    lines.extend(["", "## Reference Routes", ""])
    for route in report.get("reference_routes", []):
        lines.append(f"### {route.get('id')}")
        lines.append(f"- status: `{route.get('implementation_status')}`")
        lines.append(f"- priority: `{route.get('priority_bucket')}`")
        lines.append(f"- meaning: {route.get('name')}")
        for reason in route.get("status_reasons", []):
            lines.append(f"- reason: {reason}")
        lines.append("")
    lines.extend([
        "## Next Work",
        "",
        *[f"- {item}" for item in report.get("next_recommended_work", [])],
        "",
    ])
    return "\n".join(lines)


def write_coverage_report(
    out_dir: str | Path = "agent_workspace/search_space",
    *,
    data_path: str | Path | None = "agent_workspace/data/generated_yield_stress/generated_yield_stress_data_engineered.csv",
) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = audit_search_space(data_path)
    (out / "yield_search_space_coverage.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out / "yield_search_space_coverage.md").write_text(
        render_coverage_markdown(report),
        encoding="utf-8",
    )
    return report


if __name__ == "__main__":
    report = write_coverage_report()
    print(json.dumps({
        "coverage_counts": report["coverage_counts"],
        "currently_executable_routes": report["currently_executable_routes"],
        "high_value_unexecuted_routes": report["high_value_unexecuted_routes"],
    }, ensure_ascii=False, indent=2))
