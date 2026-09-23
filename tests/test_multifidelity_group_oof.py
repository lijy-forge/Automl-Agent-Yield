from __future__ import annotations

import numpy as np
import pandas as pd

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
