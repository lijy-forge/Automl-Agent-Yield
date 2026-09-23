"""Deterministic dataset-role and anchor-compatibility audit helpers.

The project contains several 200-row datasets with different meanings.  This
module keeps their lineage separate from model quality so reports cannot call
an augmented development set an independent business anchor.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


DATASET_ROLE_LABELS = {
    "augmented_development": "200条增强开发集",
    "multifidelity_development": "HF/LF多保真开发集",
    "literature_mechanism_anchor": "文献机理锚点",
    "synthetic_low_fidelity_development": "合成低保真开发集",
    "observed_development": "实测开发数据",
    "user_supplied_unclassified": "用户数据（待确认角色）",
}


def _counts(series: pd.Series | None) -> dict[str, int]:
    if series is None:
        return {}
    return {
        str(key): int(value)
        for key, value in series.dropna().astype(str).value_counts().to_dict().items()
    }


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return None


def summarize_dataset_lineage(
    dataframe: pd.DataFrame,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Classify how a normalized yield dataset may be used in evaluation."""

    metadata = metadata or {}
    schema = str(metadata.get("source_schema") or "unknown")
    fidelity_counts = _counts(dataframe.get("data_fidelity"))
    source_counts = _counts(dataframe.get("source"))
    augmented_values = [
        parsed
        for parsed in (_as_bool(value) for value in dataframe.get("is_augmented", pd.Series(dtype=object)))
        if parsed is not None
    ]
    augmented_count = sum(1 for value in augmented_values if value)
    observed_count = sum(1 for value in augmented_values if not value)
    fidelity_names = {name.strip().lower() for name in fidelity_counts}
    source_names = {name.strip().lower() for name in source_counts}
    row_count = int(len(dataframe))

    is_literature_anchor = bool(
        "real_table6_anchor" in fidelity_names
        or "local_table6_materials_18_02983" in source_names
        and row_count <= 32
    )
    is_synthetic_low_fidelity = bool(
        "synthetic_low_fidelity" in fidelity_names
        or schema == "synthetic_yield_lian2025_low_fidelity"
    )
    has_hf = "high_fidelity" in fidelity_names
    has_lf = "low_fidelity" in fidelity_names

    if is_literature_anchor:
        role = "literature_mechanism_anchor"
        evaluation_role = "secondary_mechanism_consistency_only"
        boundary = "文献实验可用于机理一致性检查；不等于同域业务Holdout。"
    elif has_hf and has_lf:
        role = "multifidelity_development"
        evaluation_role = "grouped_or_paired_multifidelity_evaluation"
        boundary = "必须按base_hf_id/批次分组；paired与group-holdout结果必须分开报告。"
    elif is_synthetic_low_fidelity:
        role = "synthetic_low_fidelity_development"
        evaluation_role = "development_cv_with_literature_anchor"
        boundary = "合成数据可用于流程开发；不可宣称为生产泛化结果。"
    elif row_count and augmented_count == row_count:
        role = "augmented_development"
        evaluation_role = "development_oof_only"
        boundary = "增强数据可用于候选搜索和OOF开发评测；不是独立业务Anchor。"
    elif observed_count and augmented_count == 0:
        role = "observed_development"
        evaluation_role = "role_requires_split_confirmation"
        boundary = "数据为实测记录，但是否为独立Holdout取决于批次/时间切分和训练暴露记录。"
    else:
        role = "user_supplied_unclassified"
        evaluation_role = "role_requires_user_confirmation"
        boundary = "尚未证明该数据未参与训练，不得自动标记为业务Anchor。"

    group_columns = [name for name in ("base_hf_id", "raw_batch_id", "batch_id") if name in dataframe.columns]
    return {
        "dataset_role": role,
        "dataset_role_label": DATASET_ROLE_LABELS[role],
        "evaluation_role": evaluation_role,
        "evaluation_boundary": boundary,
        "source_schema": schema,
        "row_count": row_count,
        "feature_count": len(metadata.get("feature_columns") or []),
        "target_column": metadata.get("target_column") or "yield_stress",
        "target_unit": metadata.get("target_unit") or "",
        "fidelity_counts": fidelity_counts,
        "source_counts": source_counts,
        "augmented_count": int(augmented_count),
        "observed_count": int(observed_count),
        "group_columns": group_columns,
        "independent_business_anchor": False,
    }


def audit_anchor_compatibility(
    training_metadata: dict[str, Any],
    anchor_metadata: dict[str, Any],
    *,
    training_lineage: dict[str, Any] | None = None,
    anchor_lineage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return an explicit compatibility verdict before anchor evaluation."""

    training_lineage = training_lineage or {}
    anchor_lineage = anchor_lineage or {}
    train_features = [str(item) for item in training_metadata.get("feature_columns") or []]
    anchor_features = {str(item) for item in anchor_metadata.get("feature_columns") or []}
    missing_features = [item for item in train_features if item not in anchor_features]
    train_schema = str(training_metadata.get("source_schema") or "unknown")
    anchor_schema = str(anchor_metadata.get("source_schema") or "unknown")
    train_unit = str(training_metadata.get("target_unit") or "")
    anchor_unit = str(anchor_metadata.get("target_unit") or "")
    reasons: list[str] = []

    if missing_features:
        reasons.append(
            f"anchor缺少{len(missing_features)}个训练特征（例：{', '.join(missing_features[:5])}）"
        )
    if train_unit and anchor_unit and train_unit != anchor_unit:
        reasons.append(f"目标单位不一致：{train_unit} vs {anchor_unit}")
    if train_schema.startswith("generated_yield_process") and anchor_schema.startswith("lian2025"):
        reasons.append("增强工艺药浆数据与Lian水泥浆体文献数据不属于同一材料/特征域")

    compatible = not reasons
    return {
        "status": "compatible" if compatible else "incompatible",
        "compatible": compatible,
        "training_schema": train_schema,
        "anchor_schema": anchor_schema,
        "training_dataset_role": training_lineage.get("dataset_role"),
        "anchor_dataset_role": anchor_lineage.get("dataset_role"),
        "missing_feature_count": len(missing_features),
        "missing_features": missing_features,
        "reasons": reasons,
        "guardrail_action": "enable_anchor_validation" if compatible else "disable_anchor_validation",
    }
