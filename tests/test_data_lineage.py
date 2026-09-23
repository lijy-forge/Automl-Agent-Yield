from __future__ import annotations

import pandas as pd

from knowledge.yield_data_lineage import audit_anchor_compatibility, summarize_dataset_lineage
from knowledge.yield_schema import load_yield_dataframe


def test_augmented_200_rows_are_development_data_not_business_anchor() -> None:
    dataframe = pd.DataFrame(
        {
            "yield_stress": [69.0] * 200,
            "data_fidelity": ["augmented_hf"] * 200,
            "source": ["generated_yield_stress_data"] * 200,
            "is_augmented": [True] * 200,
        }
    )
    lineage = summarize_dataset_lineage(
        dataframe,
        {
            "source_schema": "generated_yield_process_202607",
            "feature_columns": ["slurry_temp_c", "internal_pressure_kpa"],
            "target_column": "yield_stress",
            "target_unit": "Pa",
        },
    )

    assert lineage["dataset_role"] == "augmented_development"
    assert lineage["row_count"] == 200
    assert lineage["augmented_count"] == 200
    assert lineage["evaluation_role"] == "development_oof_only"
    assert lineage["independent_business_anchor"] is False


def test_multifidelity_200_rows_require_grouped_or_paired_reporting() -> None:
    dataframe = pd.DataFrame(
        {
            "yield_stress": [69.0] * 200,
            "data_fidelity": ["high_fidelity"] * 20 + ["low_fidelity"] * 180,
            "is_augmented": [False] * 20 + [True] * 180,
            "base_hf_id": [f"batch_{index % 20}" for index in range(200)],
        }
    )
    lineage = summarize_dataset_lineage(
        dataframe,
        {"source_schema": "yield_stepwise_process_20260804", "feature_columns": ["step_01"]},
    )

    assert lineage["dataset_role"] == "multifidelity_development"
    assert lineage["fidelity_counts"] == {"low_fidelity": 180, "high_fidelity": 20}
    assert lineage["group_columns"] == ["base_hf_id"]


def test_literature_anchor_is_not_compatible_with_process_feature_space() -> None:
    training_meta = {
        "source_schema": "generated_yield_process_202607",
        "feature_columns": ["slurry_temp_c", "internal_pressure_kpa", "phi"],
        "target_unit": "Pa",
    }
    anchor_meta = {
        "source_schema": "lian2025_table6_full",
        "feature_columns": ["phi", "sp_percent", "w_b"],
        "target_unit": "Pa",
    }
    audit = audit_anchor_compatibility(
        training_meta,
        anchor_meta,
        training_lineage={"dataset_role": "augmented_development"},
        anchor_lineage={"dataset_role": "literature_mechanism_anchor"},
    )

    assert audit["status"] == "incompatible"
    assert audit["guardrail_action"] == "disable_anchor_validation"
    assert audit["missing_features"] == ["slurry_temp_c", "internal_pressure_kpa"]
    assert any("不属于同一材料" in reason for reason in audit["reasons"])


def test_lian_csv_provenance_survives_schema_normalization(tmp_path) -> None:
    csv_path = tmp_path / "table6.csv"
    pd.DataFrame(
        {
            "sample_id": ["table6_01"],
            "phi": [0.458],
            "sp_percent": [0.8],
            "cement_kg_m3": [1130.0],
            "fly_ash_kg_m3": [234.0],
            "water_kg_m3": [533.0],
            "yield_stress": [0.35],
            "data_fidelity": ["real_table6_anchor"],
            "source": ["LOCAL_TABLE6_MATERIALS_18_02983"],
        }
    ).to_csv(csv_path, index=False)

    dataframe, metadata = load_yield_dataframe(csv_path)
    lineage = summarize_dataset_lineage(dataframe, metadata)

    assert dataframe.loc[0, "data_fidelity"] == "real_table6_anchor"
    assert lineage["dataset_role"] == "literature_mechanism_anchor"
