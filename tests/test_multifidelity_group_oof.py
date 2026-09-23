from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from knowledge.yield_joint_search import Candidate, evaluate_candidate, run_joint_search


class _MeanRegressor:
    def fit(self, X, y):
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_, dtype=float)


def _multifidelity_frame(*, include_groups: bool = True) -> pd.DataFrame:
    rows = []
    for group_index in range(4):
        group = f"batch_{group_index:02d}"
        rows.append(
            {
                "sample_id": f"hf_{group_index}",
                "feature": float(group_index),
                "yield_stress": 10.0 + group_index,
                "data_fidelity": "high_fidelity",
                "base_hf_id": group,
            }
        )
        for low_index in range(2):
            rows.append(
                {
                    "sample_id": f"lf_{group_index}_{low_index}",
                    "feature": float(group_index) + 0.1 * low_index,
                    "yield_stress": 9.5 + group_index + 0.1 * low_index,
                    "data_fidelity": "low_fidelity",
                    "base_hf_id": group,
                }
            )
    dataframe = pd.DataFrame(rows)
    if not include_groups:
        dataframe = dataframe.drop(columns=["base_hf_id"])
    return dataframe


def test_multifidelity_group_oof_excludes_lf_from_validation_groups() -> None:
    dataframe = _multifidelity_frame()
    candidate = Candidate(
        "strict-mf",
        _MeanRegressor,
        spec={"fusion_mode": "multi_fidelity_base_residual"},
    )
    result = evaluate_candidate(
        candidate,
        dataframe,
        ["feature"],
        dataframe["yield_stress"].to_numpy(dtype=float),
        n_splits=2,
        random_state=42,
    )

    assert result["status"] == "evaluated"
    protocol = result["evaluation_protocol"]
    assert protocol["type"] == "high_fidelity_group_oof"
    assert protocol["group_column"] == "base_hf_id"
    assert protocol["strict_group_isolation_passed"] is True
    assert protocol["n_shared_groups"] == 4
    for fold in protocol["folds"]:
        assert fold["n_hf_train_groups"] == 2
        assert fold["n_hf_valid_groups"] == 2
        assert fold["n_lf_train"] == 4
        assert fold["n_lf_excluded_for_validation_groups"] == 4
        assert fold["train_valid_group_overlap"] == []
        assert fold["lf_train_validation_group_overlap"] == []


def test_multifidelity_candidate_is_rejected_without_group_metadata() -> None:
    dataframe = _multifidelity_frame(include_groups=False)
    candidate = Candidate(
        "unsafe-mf",
        _MeanRegressor,
        spec={"fusion_mode": "multi_fidelity_base_residual"},
    )
    result = evaluate_candidate(
        candidate,
        dataframe,
        ["feature"],
        dataframe["yield_stress"].to_numpy(dtype=float),
        n_splits=2,
        random_state=42,
    )

    assert result["status"] == "failed"
    assert "requires base_hf_id/raw_batch_id" in result["error"]


def test_joint_search_exposes_grouped_baseline_audit(tmp_path) -> None:
    data_path = tmp_path / "multifidelity.csv"
    _multifidelity_frame().to_csv(data_path, index=False)
    candidate = Candidate(
        "strict-mf",
        _MeanRegressor,
        spec={"fusion_mode": "multi_fidelity_base_residual"},
    )

    result = run_joint_search(
        [candidate],
        data_path,
        n_splits=2,
        random_state=42,
    )

    assert result["evaluation_protocol"]["split_strategy"] == "group_kfold"
    assert result["evaluation_protocol"]["strict_group_isolation"] is True
    assert result["n_evaluated"] == 1
    assert result["ranked"][0]["evaluation_protocol"]["strict_group_isolation_passed"] is True
    baseline = result["fixed_baseline"]
    assert baseline["status"] == "evaluated"
    assert baseline["split_strategy"] == "group_kfold"
    assert baseline["group_column"] == "base_hf_id"
    assert baseline["n_unique_groups"] == 4


def test_llm_only_keeps_fixed_candidates_as_comparison_without_fallback(tmp_path) -> None:
    data_path = tmp_path / "linear.csv"
    x = np.linspace(0.1, 2.0, 24)
    pd.DataFrame(
        {"sample_id": [f"s{i}" for i in range(len(x))], "feature": x, "yield_stress": 2.0 * x + 1.0}
    ).to_csv(data_path, index=False)
    fixed = Candidate(
        "fixed-linear",
        LinearRegression,
        spec={"model_origin": "fixed_family", "fusion_mode": "raw_ml"},
    )
    generated = Candidate(
        "llm-mean",
        _MeanRegressor,
        spec={"model_origin": "llm_model", "fusion_mode": "raw_ml"},
    )

    mixed = run_joint_search([fixed, generated], data_path, n_splits=3, champion_policy="mixed")
    llm_only = run_joint_search([fixed, generated], data_path, n_splits=3, champion_policy="llm_only")

    assert mixed["champion"]["name"] == "fixed-linear"
    assert llm_only["champion"] is None
    assert llm_only["selection_status"] == "no_acceptable_llm_candidate"
    assert llm_only["champion_selection"]["comparison_only_candidate_count"] == 1
    fixed_row = next(row for row in llm_only["ranked"] if row["name"] == "fixed-linear")
    assert fixed_row["champion_eligible"] is False
    assert fixed_row["champion_exclusion_reason"] == "fixed_family_is_comparison_only"


def test_llm_only_selects_generated_model_that_matches_baseline(tmp_path) -> None:
    data_path = tmp_path / "linear.csv"
    x = np.linspace(0.1, 2.0, 24)
    pd.DataFrame(
        {"sample_id": [f"s{i}" for i in range(len(x))], "feature": x, "yield_stress": 2.0 * x + 1.0}
    ).to_csv(data_path, index=False)
    fixed = Candidate(
        "fixed-mean",
        _MeanRegressor,
        spec={"model_origin": "fixed_family", "fusion_mode": "raw_ml"},
    )
    generated = Candidate(
        "llm-linear",
        LinearRegression,
        spec={"model_origin": "llm_model", "fusion_mode": "raw_ml"},
    )

    result = run_joint_search([fixed, generated], data_path, n_splits=3, champion_policy="llm_only")

    assert result["champion"]["name"] == "llm-linear"
    assert result["champion"]["spec"]["model_origin"] == "llm_model"
    assert result["selection_status"] == "selected"
    assert result["champion_selection"]["policy"] == "llm_only"
